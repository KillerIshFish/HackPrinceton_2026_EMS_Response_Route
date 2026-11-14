"""FastAPI backend for the 911 Caller Location Predictor.

Run from the repo root::

    uv run uvicorn api.server:app --reload --port 8000

The server holds ONE in-memory demo call. Every mutating endpoint returns the
full ``DemoState`` so the web client can simply replace its state.

The call itself lives in ``api/session.py`` (``CallSession``; see its docstring for the
pipeline: interpreter -> evidence matches -> fusion -> residual after looks -> policy -> plan / approach)
and writes call events to ``data/processed/call_log.jsonl`` (``CALL_LOG_PATH`` overrides it).

Network use
-----------
* Anthropic: only when the parser mode is "llm" (Claude Haiku; "agent" is an older name for the
  same interpreter). A new session starts in that mode when a key is configured
  (``default_parser_mode``; ``PARSER_MODE`` overrides it). One request per caller turn in the
  usual case, plus one no-output request when a call starts to write the building's prompt
  cache; replies are cached on disk, see ``parser/agent.py``.
* Vexcel: ``GET /vexcel/tile/...`` proxies aerial tiles (cached on disk) so the
  API key stays on the server. Only tiles around a known building are served.
* Matterport: nothing at request time; scans are fetched by ``ingest.real_building``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.schema import Building, Gates, Posterior, SearchPlan, Situation, TranscriptTurn  # noqa: F401  (TranscriptTurn re-exported)
from ingest import matterport_api, vexcel
from ingest.matterport import list_buildings, load_building
from ingest.real_building import load_meta
from ingest.unified import index as property_index
from ingest.unified import pipeline as unified
from ingest.unified.lift import from_building, from_model3d
from ingest.unified.project import viewer_payload
from parser.extract import LLM_MODEL
from api.session import (  # noqa: F401  (models re-exported for older imports)
    DEMO_HOUR,
    MAX_TEAMS,
    MAX_UTTERANCE_CHARS,
    AskRequest,
    BuildingSummary,
    CallSession,
    DemoState,
    HistoryPoint,
    LLMStatus,
    SituationPatch,
    Utterance,
    building_summaries,
    llm_available,
    scenario_script,
    start_scenario,
)

load_dotenv()

SYNTHETIC_BUILDING_ID = "ptc-100"
REAL_BUILDING_ID = "bundy-olympic"

# Pre-Step-0 name of the session class.
DemoSession = CallSession


def default_building_id() -> str:
    """``DEMO_BUILDING_ID`` if set, else the real building when it has been built, else synthetic."""
    wanted = (os.getenv("DEMO_BUILDING_ID") or "").strip()
    known = {b.id for b in list_buildings()}
    if wanted in known:
        return wanted
    return REAL_BUILDING_ID if REAL_BUILDING_ID in known else SYNTHETIC_BUILDING_ID


MODEL_PARSER_MODES = ("llm", "agent")  # both are the grounded Claude Haiku interpreter (parser.agent)


def default_parser_mode() -> str:
    """Parser a new session starts with: ``PARSER_MODE`` (rules | llm | agent) when set, else
    Claude Haiku (``llm``) when a key is configured, else the offline rules."""
    wanted = (os.getenv("PARSER_MODE") or "").strip().lower()
    if wanted in ("rules", "llm", "agent"):
        return wanted if wanted == "rules" or llm_available() else "rules"
    return "llm" if llm_available() else "rules"


def _prewarm(building: Building, mode: str) -> None:
    """Write the building's prompt cache in the background when a call starts in a model mode,
    so the first caller turn does not pay for it (``parser.agent.prewarm``; one request with
    no output). ``LLM_PREWARM=0`` turns it off."""
    if mode not in MODEL_PARSER_MODES or not llm_available():
        return
    if (os.getenv("LLM_PREWARM") or "1").strip().lower() in ("0", "false", "no", "off"):
        return
    from parser.agent import prewarm

    threading.Thread(target=prewarm, args=(building,), daemon=True, name="llm-prewarm").start()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="911 Caller Location Predictor",
    version="0.2.0",
    description="Room-level caller location from a phone fix and what the caller says; "
                "synthetic building plus real buildings assembled from CoStar, Matterport and Vexcel.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: Optional[CallSession] = None
_session_lock = threading.Lock()


def get_session() -> CallSession:
    global _session
    with _session_lock:
        if _session is None:
            building, mode = load_building(default_building_id()), default_parser_mode()
            _session = CallSession(building, parser_mode=mode)
            _prewarm(building, mode)
        return _session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, Any]:
    """Client configuration. The Matterport SDK key is a browser key by design; no secret is returned."""
    return {
        "matterport_sdk_key": (os.getenv("MATTERPORT_SDK_KEY") or "").strip() or None,
        "llm_available": llm_available(),
        "llm_model": LLM_MODEL,
        "vexcel_available": bool("".join((os.getenv("VEXCEL_API_KEY") or "").split())),
    }


@app.get("/buildings", response_model=list[Building])
def get_buildings() -> list[Building]:
    return list_buildings()


@app.get("/buildings/{building_id}", response_model=Building)
def get_building(building_id: str) -> Building:
    try:
        return load_building(building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{building_id}'")


@app.get("/buildings/{building_id}/meta")
def get_building_meta(building_id: str) -> dict[str, Any]:
    """Provenance, Matterport scans and the aerial site annotation for a building.

    ``site.tiles`` is the Vexcel tile range the client should request from
    ``/vexcel/tile`` (absent for buildings with no imagery).
    """
    try:
        load_building(building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{building_id}'")
    meta = load_meta(building_id)
    tiles = _site_tiles(meta)
    if tiles:
        meta.setdefault("site", {})["tiles"] = tiles
    return meta


@app.get("/lookup")
def lookup(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(8, ge=1, le=25)) -> list[dict]:
    """Properties matching an address, a building name or a property id (indexed; about a millisecond)."""
    try:
        return property_index.lookup(q, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/buildings/generate")
def generate_building(q: str = Query(..., min_length=1, max_length=200), stamp: bool = Query(True),
                      context: bool = Query(True)) -> dict[str, Any]:
    """Generate a building model now from an address, a property id or ``mp:<model id>``.

    Runs the unified pipeline (lookup, roster, layout, unit-type scans, site context) and
    registers the result, so ``/calls/demo/building?id=<building_id>`` can start a call on it.
    """
    try:
        return unified.generate(q, stamp=stamp, context=context)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]) if exc.args else "not found")
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))


_model3d_cache: dict[tuple[str, float], dict] = {}


@app.get("/buildings/{building_id}/model3d")
def get_model3d(building_id: str) -> dict[str, Any]:
    """The 3D viewer payload for any building the app knows.

    Best source first: a model generated by the unified pipeline; for a whole-building
    scan, Track A's surveyed ``Model3D`` (rooms mapped back to this building's room ids);
    otherwise the building's own room outlines extruded.
    """
    try:
        building = load_building(building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{building_id}'")
    path = unified.model_path(building_id)
    stamp = path.stat().st_mtime if path.is_file() else 0.0
    key = (building_id, stamp)
    if key in _model3d_cache:
        return _model3d_cache[key]
    bm = unified.load_model(building_id)
    room_of: dict[str, str] = {}
    source = {"whole_scan": "Matterport scan (surveyed walls and openings)",
              "scan_clusters": "Matterport scan positions (this scan has no wall data)"}.get(
        bm.archetype if bm else "", "generated from the property record")
    if bm is None:
        meta = load_meta(building_id)
        scans = meta.get("matterport_models") or []
        if meta.get("kind") == "full_scan" and scans:
            from ingest.unified.stamp import load_unit_model

            model, _ = load_unit_model(scans[0]["model_id"])
            if model is not None:
                bm = from_model3d(model, None, scans[0]["model_id"])
                room_of = {m["scan_room_id"]: rid for rid, m in (meta.get("rooms") or {}).items() if m.get("scan_room_id")}
                source = "Matterport scan (surveyed walls and openings)"
        if bm is None:
            bm = from_building(building, "surveyed" if meta.get("kind") == "full_scan" else "template")
            source = "room outlines"
    payload = viewer_payload(bm, room_of)
    known = {r.id for r in building.rooms}
    payload["source"] = source
    payload["unmapped"] = sum(1 for s in payload["spaces"] if s["r"] not in known)
    _model3d_cache.clear()
    _model3d_cache[key] = payload
    return payload


@app.get("/buildings/{building_id}/floorplan/{floor}")
def get_floorplan(building_id: str, floor: int) -> FileResponse:
    """Matterport's plan image of one floor (PNG, transparent background).

    Only buildings built from a whole-building scan have these; their placement
    (``origin``, ``resolution``) is in ``meta["floorplans"][floor]``.
    """
    try:
        load_building(building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{building_id}'")
    entry = (load_meta(building_id).get("floorplans") or {}).get(str(floor))
    path = matterport_api.floorplan_dir() / entry["file"] if entry else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No plan image for floor {floor} of '{building_id}'")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


def _site_tiles(meta: dict) -> Optional[dict]:
    imagery = (meta.get("site") or {}).get("imagery") or {}
    if not imagery or not meta.get("center_latlng"):
        return None
    return vexcel.site_tiles(
        meta["center_latlng"], float(imagery.get("radius_m", 90.0)), int(imagery.get("zoom", 20)),
        imagery.get("layer", vexcel.DEFAULT_LAYER),
    )


_tile_windows: dict = {"at": 0.0, "windows": []}
TILE_WINDOW_TTL_S = 5.0


def _tile_allowed(layer: str, z: int, x: int, y: int) -> bool:
    """True when the tile lies in some known building's site window.

    A panel asks for dozens of tiles at once, so the windows are read from the building
    metadata once and reused for a few seconds rather than once per tile.
    """
    now = time.monotonic()
    if now - _tile_windows["at"] > TILE_WINDOW_TTL_S:
        windows = []
        for b in list_buildings():
            t = _site_tiles(load_meta(b.id))
            if t:
                windows.append(t)
        _tile_windows["at"], _tile_windows["windows"] = now, windows
    hit = any(t["layer"] == layer and t["zoom"] == z and t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]
              for t in _tile_windows["windows"])
    if not hit and now - _tile_windows["at"] > 0.5:
        # a building generated a moment ago is not in the cached list yet: look again once
        _tile_windows["at"] = 0.0
        return _tile_allowed(layer, z, x, y)
    return hit


@app.get("/vexcel/tile/{layer}/{z}/{x}/{y}")
def vexcel_tile(layer: str, z: int, x: int, y: int) -> Response:
    """One aerial tile (JPEG). Only tiles inside a known building's site window are served."""
    allowed = _tile_allowed(layer, z, x, y)
    if not allowed:
        raise HTTPException(status_code=404, detail="Tile is outside every building's site window")
    try:
        body = vexcel.fetch_tile(layer, z, x, y)
    except vexcel.VexcelError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return Response(content=body, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/calls/demo/state", response_model=DemoState)
def demo_state() -> DemoState:
    s = get_session()
    with s.lock:
        return s.state()


@app.post("/calls/demo/step", response_model=DemoState)
def demo_step() -> DemoState:
    """Reveal the next scripted call turn (no-op once the script is finished)."""
    s = get_session()
    with s.lock:
        s.step()
        return s.state()


@app.post("/calls/demo/reset", response_model=DemoState)
def demo_reset() -> DemoState:
    s = get_session()
    with s.lock:
        s.reset()
        return s.state()


@app.post("/calls/demo/teams", response_model=DemoState)
def demo_teams(n: int = Query(1, ge=1, le=MAX_TEAMS)) -> DemoState:
    s = get_session()
    with s.lock:
        s.n_teams = n
        return s.state()


@app.post("/calls/demo/building", response_model=DemoState)
def demo_building(id: str = Query(...)) -> DemoState:
    """Start the demo call over on another building (keeps the parser mode)."""
    global _session
    try:
        building = load_building(id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{id}'")
    with _session_lock:
        mode = _session.parser_mode if _session is not None else default_parser_mode()
        _session = CallSession(building, parser_mode=mode)
        s = _session
    _prewarm(building, mode)
    with s.lock:
        return s.state()


@app.post("/calls/demo/parser", response_model=DemoState)
def demo_parser(mode: str = Query(..., pattern="^(rules|llm|agent)$")) -> DemoState:
    """Choose the interpreter: ``rules`` (offline) or ``llm`` (Claude Haiku: the grounded
    interpreter that reads the call against the building's room list; replies are cached and it
    falls back to the rules when the model cannot be reached). ``agent`` is accepted as an older
    name for ``llm``; both run the same interpreter."""
    if mode in ("llm", "agent") and not llm_available():
        raise HTTPException(status_code=409, detail=f"The {mode} parser needs LLM_PROVIDER=anthropic and LLM_API_KEY in .env")
    s = get_session()
    with s.lock:
        s.set_parser(mode)
        return s.state()


@app.post("/calls/demo/utterance", response_model=DemoState)
def demo_utterance(utterance: Utterance) -> DemoState:
    """Add a turn: something the caller said (parsed with the session's interpreter) or a dispatcher turn.

    ``answers`` carries the caller's Yes / No / Not sure reply to a yes/no question.
    """
    text = utterance.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Utterance is empty")
    if utterance.speaker not in ("caller", "dispatcher"):
        raise HTTPException(status_code=422, detail="speaker must be 'caller' or 'dispatcher'")
    s = get_session()
    with s.lock:
        s.say(text, speaker=utterance.speaker, answer=utterance.answers)
        return s.state()


@app.post("/calls/demo/ask", response_model=DemoState)
def demo_ask(req: AskRequest) -> DemoState:
    """Log a (suggested) question the dispatcher asked as a dispatcher turn. It is not suggested again."""
    s = get_session()
    with s.lock:
        s.ask(req)
        return s.state()


@app.post("/calls/demo/situation", response_model=DemoState)
def demo_situation(patch: SituationPatch) -> DemoState:
    """Patch the situation (incident type, confirm / dismiss a suggestion, smoke, crew load, teams...)."""
    s = get_session()
    with s.lock:
        s.patch_situation(patch)
        return s.state()


def _set_evidence_status(evidence_id: str, status: str) -> DemoState:
    s = get_session()
    with s.lock:
        try:
            s.set_evidence_status(evidence_id, status)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown evidence '{evidence_id}'")
        return s.state()


@app.post("/calls/demo/evidence/{evidence_id}/reject", response_model=DemoState)
def demo_evidence_reject(evidence_id: str) -> DemoState:
    """The dispatcher rejects an evidence item; the posterior and plan rebuild without it."""
    return _set_evidence_status(evidence_id, "rejected")


@app.post("/calls/demo/evidence/{evidence_id}/restore", response_model=DemoState)
def demo_evidence_restore(evidence_id: str) -> DemoState:
    """Undo a reject, or reinstate a superseded item."""
    return _set_evidence_status(evidence_id, "active")


@app.post("/calls/demo/gates", response_model=DemoState)
def demo_gates(gates: Gates) -> DemoState:
    """Rooms and entrances closed on the officer's word (replaces both lists). Undoable."""
    s = get_session()
    with s.lock:
        try:
            s.set_gates(gates)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown room or entrance '{exc.args[0]}'")
        return s.state()


@app.post("/calls/demo/undo", response_model=DemoState)
def demo_undo() -> DemoState:
    """Undo the last searched, found or gate action (no-op when there is none)."""
    s = get_session()
    with s.lock:
        s.undo()
        return s.state()


@app.post("/calls/demo/timestamp", response_model=DemoState)
def demo_timestamp(
    name: str = Query(..., pattern="^(wheels_stopped|building_entry|floor_arrival|patient_contact)$"),
) -> DemoState:
    """Log one of the four call timestamps at the server's current time."""
    s = get_session()
    with s.lock:
        s.log_timestamp(name)  # type: ignore[arg-type]
        return s.state()


@app.post("/calls/demo/scenario", response_model=DemoState)
def demo_scenario(id: str = Query(..., min_length=1, max_length=100)) -> DemoState:
    """Start a scripted scenario (``state.scenarios`` lists them). Keeps the parser mode.

    The call starts with the incident type unset (``summary.incident_type`` is what the call
    turns out to be): the interpreter suggests a type from the caller's words and the
    dispatcher confirms it with POST /calls/demo/situation.
    """
    global _session
    try:
        summary, _ = scenario_script(id)
        building = load_building(summary.building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{id}'")
    with _session_lock:
        mode = _session.parser_mode if _session is not None else default_parser_mode()
        _session = start_scenario(id, building, parser_mode=mode)
        s = _session
    _prewarm(building, mode)
    with s.lock:
        return s.state()


@app.get("/posterior", response_model=Posterior)
def get_posterior() -> Posterior:
    s = get_session()
    with s.lock:
        return s.posterior()


@app.get("/plan", response_model=SearchPlan)
def get_plan(teams: Optional[int] = Query(None, ge=1, le=MAX_TEAMS)) -> SearchPlan:
    s = get_session()
    with s.lock:
        return s.plan(teams)


@app.post("/rooms/{room_id}/searched", response_model=DemoState)
def searched_room(room_id: str, mode: str = Query("rapid", pattern="^(rapid|thorough)$")) -> DemoState:
    """The crew searched a room (rapid or thorough). Each call is one look; undoable."""
    s = get_session()
    with s.lock:
        try:
            s.search_room(room_id, mode)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown room '{room_id}'")
        return s.state()


@app.post("/rooms/{room_id}/clear", response_model=DemoState)
def clear_room(room_id: str) -> DemoState:
    """Pre-Step-0 alias of ``/rooms/{id}/searched?mode=rapid``, recorded once per room (idempotent)."""
    s = get_session()
    with s.lock:
        try:
            s.clear_room(room_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown room '{room_id}'")
        return s.state()


@app.post("/rooms/{room_id}/found", response_model=DemoState)
def found_room(room_id: str) -> DemoState:
    """The caller was found in this room: the search ends. Undoable."""
    s = get_session()
    with s.lock:
        try:
            s.mark_found(room_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown room '{room_id}'")
        return s.state()

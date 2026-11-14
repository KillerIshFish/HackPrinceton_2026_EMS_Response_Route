"""FastAPI backend for the 911 Caller Location Predictor.

Run from the repo root::

    uv run uvicorn api.server:app --reload --port 8000

The server holds ONE in-memory demo call. Every mutating endpoint returns the
full ``DemoState`` so the web client can simply replace its state.

Pipeline per request::

    revealed turns -> parser.extract (rules, or Claude Haiku when the session's
                      parser mode is "llm") -> clues
    core.fusion.posterior_probs(prior, phone fix, clues)
    planning.search_order.koopman_update   for every room marked clear
    core.uncertainty.summarize             -> Posterior
    planning.search_order.plan_search      -> SearchPlan (+ baselines)
    planning.approach.recommend_approach   -> entrance / staging / stairwell

Network use
-----------
* Anthropic: only when the parser mode is "llm" (``POST /calls/demo/parser``);
  results are cached on disk, see ``parser/extract.py``.
* Vexcel: ``GET /vexcel/tile/...`` proxies aerial tiles (cached on disk) so the
  API key stays on the server. Only tiles around a known building are served.
* Matterport: nothing at request time; scans are fetched by ``ingest.real_building``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.fusion import posterior_probs
from core.schema import Building, Clue, PhoneFix, Posterior, SearchPlan
from core.uncertainty import summarize
from ingest import matterport_api, vexcel
from ingest.matterport import list_buildings, load_building
from ingest.real_building import load_meta
from parser.extract import LLM_MODEL, extract_clues_with_source, llm_usage
from parser.next_question import next_question
from planning.approach import recommend_approach
from planning.search_order import koopman_update, plan_search
from sim.generate_calls import demo_script_for

load_dotenv()

SYNTHETIC_BUILDING_ID = "ptc-100"
REAL_BUILDING_ID = "bundy-olympic"
DEMO_HOUR = 14  # 2 pm on a weekday: business-hours prior
MAX_TEAMS = 3
MAX_UTTERANCE_CHARS = 600


def default_building_id() -> str:
    """``DEMO_BUILDING_ID`` if set, else the real building when it has been built, else synthetic."""
    wanted = (os.getenv("DEMO_BUILDING_ID") or "").strip()
    known = {b.id for b in list_buildings()}
    if wanted in known:
        return wanted
    return REAL_BUILDING_ID if REAL_BUILDING_ID in known else SYNTHETIC_BUILDING_ID


def llm_available() -> bool:
    return (os.getenv("LLM_PROVIDER") or "").strip().lower() == "anthropic" and bool((os.getenv("LLM_API_KEY") or "").strip())


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TranscriptTurn(BaseModel):
    turn: int
    title: str
    speaker: str
    text: str
    phone_fix: Optional[PhoneFix] = None
    clues: list[Clue] = []
    parser: Optional[str] = None  # "rules" | "llm" | "rules (llm unavailable: ...)" for caller turns
    scripted: bool = True


class HistoryPoint(BaseModel):
    turn: int
    entropy_bits: float
    expected_time_to_find_s: float
    top_room_id: str
    top_room_prob: float


class BuildingSummary(BaseModel):
    id: str
    name: str
    real: bool
    floors: int
    rooms: int


class LLMStatus(BaseModel):
    available: bool
    model: str
    usage: dict[str, Any]


class DemoState(BaseModel):
    building_id: str
    turn: int
    total_turns: int
    done: bool
    transcript: list[TranscriptTurn]
    clues: list[Clue]
    phone_fix: Optional[PhoneFix] = None
    cleared_room_ids: list[str]
    n_teams: int
    posterior: Posterior
    plan: SearchPlan
    approach: dict[str, Any]
    next_question: Optional[dict[str, Any]] = None
    history: list[HistoryPoint]
    parser_mode: str = "rules"
    llm: Optional[LLMStatus] = None
    buildings: list[BuildingSummary] = []


class Utterance(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_UTTERANCE_CHARS)


# ---------------------------------------------------------------------------
# In-memory demo session
# ---------------------------------------------------------------------------


class DemoSession:
    """One call against one building: scripted turns plus anything the dispatcher types in."""

    def __init__(self, building: Building, parser_mode: str = "rules") -> None:
        self.building = building
        self.script = demo_script_for(building.id)
        self.parser_mode = parser_mode
        self.lock = threading.Lock()
        self._history_cache: dict[int, HistoryPoint] = {}
        self.reset()

    # -- transcript --------------------------------------------------------

    def _make_turn(self, raw: dict, scripted: bool) -> TranscriptTurn:
        fix = raw.get("phone_fix")
        if fix is not None and not isinstance(fix, PhoneFix):
            fix = PhoneFix.model_validate(fix)
        speaker = raw.get("speaker", "system")
        clues: list[Clue] = []
        parser = None
        if speaker == "caller":
            clues, parser = extract_clues_with_source(raw["text"], use_llm=self.parser_mode == "llm")
        index = len(self.revealed)
        return TranscriptTurn(
            turn=index, title=raw.get("title", f"Turn {index}"), speaker=speaker, text=raw["text"],
            phone_fix=fix, clues=clues, parser=parser, scripted=scripted,
        )

    # -- mutations ---------------------------------------------------------

    def reset(self) -> None:
        self.revealed: list[TranscriptTurn] = []
        self.script_pos = 0  # index of the latest revealed scripted turn
        self.cleared: list[str] = []
        self.n_teams = 1
        self._history_cache.clear()
        self.revealed.append(self._make_turn(self.script[0], scripted=True))

    def step(self) -> None:
        if self.script_pos < len(self.script) - 1:
            self.script_pos += 1
            self.revealed.append(self._make_turn(self.script[self.script_pos], scripted=True))

    def say(self, text: str) -> None:
        """A caller utterance typed (or pasted) by the dispatcher."""
        self.revealed.append(
            self._make_turn({"title": "Caller (live)", "speaker": "caller", "text": text}, scripted=False)
        )

    def set_parser(self, mode: str) -> None:
        """Switch parser and re-extract every revealed caller turn with it."""
        if mode == self.parser_mode:
            return
        self.parser_mode = mode
        for t in self.revealed:
            if t.speaker == "caller":
                t.clues, t.parser = extract_clues_with_source(t.text, use_llm=mode == "llm")
        self._history_cache.clear()

    def clear_room(self, room_id: str) -> None:
        if room_id not in {r.id for r in self.building.rooms}:
            raise KeyError(room_id)
        if room_id not in self.cleared:
            self.cleared.append(room_id)

    # -- evidence ----------------------------------------------------------

    def _evidence(self, upto: int) -> tuple[Optional[PhoneFix], list[Clue]]:
        fix: Optional[PhoneFix] = None
        clues: list[Clue] = []
        for t in self.revealed[: upto + 1]:
            if t.phone_fix is not None:
                fix = t.phone_fix  # latest fix wins
            clues.extend(t.clues)
        return fix, clues

    def _probs(self, upto: int, cleared: list[str]) -> tuple[dict[str, float], float]:
        fix, clues = self._evidence(upto)
        room_probs, p_nim = posterior_probs(self.building, hour=DEMO_HOUR, fix=fix, clues=clues)
        for room_id in cleared:
            room_probs, p_nim = koopman_update(room_probs, p_nim, room_id)
        return room_probs, p_nim

    def _history_point(self, index: int) -> HistoryPoint:
        """Evidence-only snapshot after a revealed turn (single team, nothing cleared). Cached."""
        if index not in self._history_cache:
            room_probs, p_nim = self._probs(index, cleared=[])
            post = summarize(self.building, room_probs, p_nim)
            plan = plan_search(self.building, room_probs, p_nim, n_teams=1)
            top_id = max(room_probs, key=room_probs.get)
            self._history_cache[index] = HistoryPoint(
                turn=index,
                entropy_bits=post.entropy_bits,
                expected_time_to_find_s=plan.expected_time_to_find_s,
                top_room_id=top_id,
                top_room_prob=room_probs[top_id],
            )
        return self._history_cache[index]

    # -- views -------------------------------------------------------------

    @property
    def _last(self) -> int:
        return len(self.revealed) - 1

    def posterior(self) -> Posterior:
        room_probs, p_nim = self._probs(self._last, self.cleared)
        return summarize(self.building, room_probs, p_nim)

    def plan(self, n_teams: Optional[int] = None) -> SearchPlan:
        room_probs, p_nim = self._probs(self._last, self.cleared)
        return self._plan(room_probs, p_nim, n_teams or self.n_teams)

    def _plan(self, room_probs: dict[str, float], p_nim: float, n_teams: int) -> SearchPlan:
        # After a room is cleared the crew replans from where it is standing, and
        # cleared rooms are not offered again (their residual mass stays in the posterior).
        start = self.cleared[-1] if self.cleared else None
        return plan_search(
            self.building, room_probs, p_nim, n_teams=n_teams, start_node=start, exclude=tuple(self.cleared),
        )

    def state(self) -> DemoState:
        fix, clues = self._evidence(self._last)
        room_probs, p_nim = self._probs(self._last, self.cleared)
        asked = {f"{c.type}:{c.value}" for c in clues} | {c.value for c in clues}
        return DemoState(
            building_id=self.building.id,
            turn=self.script_pos,
            total_turns=len(self.script),
            done=self.script_pos >= len(self.script) - 1,
            transcript=list(self.revealed),
            clues=clues,
            phone_fix=fix,
            cleared_room_ids=list(self.cleared),
            n_teams=self.n_teams,
            posterior=summarize(self.building, room_probs, p_nim),
            plan=self._plan(room_probs, p_nim, self.n_teams),
            approach=recommend_approach(self.building, room_probs),
            next_question=next_question(self.building, room_probs, p_nim, asked=asked),
            history=[self._history_point(i) for i in range(len(self.revealed))],
            parser_mode=self.parser_mode,
            llm=LLMStatus(available=llm_available(), model=LLM_MODEL, usage=llm_usage()),
            buildings=building_summaries(),
        )


def building_summaries() -> list[BuildingSummary]:
    return [
        BuildingSummary(id=b.id, name=b.name, real=bool(load_meta(b.id).get("real")),
                        floors=len(b.floors), rooms=len(b.rooms))
        for b in list_buildings()
    ]


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

_session: Optional[DemoSession] = None
_session_lock = threading.Lock()


def get_session() -> DemoSession:
    global _session
    with _session_lock:
        if _session is None:
            _session = DemoSession(load_building(default_building_id()))
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


@app.get("/vexcel/tile/{layer}/{z}/{x}/{y}")
def vexcel_tile(layer: str, z: int, x: int, y: int) -> Response:
    """One aerial tile (JPEG). Only tiles inside a known building's site window are served."""
    allowed = False
    for b in list_buildings():
        t = _site_tiles(load_meta(b.id))
        if t and t["layer"] == layer and t["zoom"] == z and t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]:
            allowed = True
            break
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
        mode = _session.parser_mode if _session is not None else "rules"
        _session = DemoSession(building, parser_mode=mode)
        s = _session
    with s.lock:
        return s.state()


@app.post("/calls/demo/parser", response_model=DemoState)
def demo_parser(mode: str = Query(..., pattern="^(rules|llm)$")) -> DemoState:
    """Choose the clue parser: ``rules`` (offline) or ``llm`` (Claude Haiku, cached)."""
    if mode == "llm" and not llm_available():
        raise HTTPException(status_code=409, detail="LLM parser needs LLM_PROVIDER=anthropic and LLM_API_KEY in .env")
    s = get_session()
    with s.lock:
        s.set_parser(mode)
        return s.state()


@app.post("/calls/demo/utterance", response_model=DemoState)
def demo_utterance(utterance: Utterance) -> DemoState:
    """Add something the caller just said; it is parsed with the session's parser."""
    text = utterance.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Utterance is empty")
    s = get_session()
    with s.lock:
        s.say(text)
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


@app.post("/rooms/{room_id}/clear", response_model=DemoState)
def clear_room(room_id: str) -> DemoState:
    """Dispatcher marks a room searched-and-empty: Koopman update p_r <- p_r (1 - 0.9), then replan."""
    s = get_session()
    with s.lock:
        try:
            s.clear_room(room_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown room '{room_id}'")
        return s.state()

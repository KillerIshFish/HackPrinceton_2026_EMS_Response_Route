"""The stages in order, timed: lookup -> record -> roster -> scans -> layout -> context -> project -> write.

``generate(query)`` runs at call time. ``query`` is an address, a building name, a
property id, or ``mp:<model id>`` for one whole-building Matterport scan.

Outputs (all under ``data/processed/``, which is gitignored):

    models/<id>.model.json     the BuildingModel (deterministic)
    context/<id>.json          the site-context sidecar (changes over time)
    <id>.building.json         core.schema.Building, picked up by the app's building list
    <id>.meta.json             provenance for the web app

The run report (timings, scan fetch statuses) is returned, never written into the model.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ingest.unified import index
from ingest.unified.context import build_context
from ingest.unified.layout import build_layout
from ingest.unified.lift import from_model3d, from_scan_model
from ingest.unified.project import to_building
from ingest.unified.record import property_facts
from ingest.unified.roster import build_roster
from ingest.unified.schema import SCHEMA_VERSION, BuildingModel
from ingest.unified.stamp import CACHE_DIR, load_unit_model, make_interior

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = PROCESSED / "models"
MAX_SCANS = 6


class _Stages:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self._t = time.perf_counter()

    def mark(self, name: str, detail: str = "") -> None:
        now = time.perf_counter()
        self.items.append({"stage": name, "ms": round((now - self._t) * 1000, 1), "detail": detail})
        self._t = now


def model_path(building_id: str) -> Path:
    return MODELS_DIR / f"{building_id}.model.json"


def load_model(building_id: str) -> Optional[BuildingModel]:
    """A previously generated BuildingModel, rebuilt from its JSON."""
    from ingest.unified.schema import Entrance, Level, Link, Opening, Space, Wall

    path = model_path(building_id)
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))

    def pt(p):
        return (p[0], p[1]) if p is not None else None

    return BuildingModel(
        schema=d["schema"], id=d["id"], name=d["name"], address=d["address"], use_type=d["use_type"],
        archetype=d["archetype"], tier=d["tier"], yaw_deg=d["yaw_deg"], geolocation=pt(d.get("geolocation")),
        levels=tuple(Level(**lv) for lv in d["levels"]),
        spaces=tuple(Space(**{**s, "polygon": tuple(pt(p) for p in s["polygon"]), "centroid": pt(s["centroid"]),
                              "attrs": tuple(tuple(a) for a in s.get("attrs") or ())}) for s in d["spaces"]),
        walls=tuple(Wall(**{**w, "a": pt(w["a"]), "b": pt(w["b"]), "space_ids": tuple(w["space_ids"]),
                            "openings": tuple(Opening(**o) for o in w["openings"])}) for w in d["walls"]),
        links=tuple(Link(**{**ln, "xy": pt(ln.get("xy"))}) for ln in d["links"]),
        entrances=tuple(Entrance(**{**e, "xy": pt(e["xy"]), "outward": pt(e["outward"])}) for e in d["entrances"]),
        stats=d["stats"], notes=tuple(d.get("notes") or ()),
    )


def build_property_model(facts: dict, models: Optional[dict] = None, scheme_override: Optional[str] = None) -> BuildingModel:
    """Layers 1-5 for one property: pure function of the record and the cached scans."""
    roster = build_roster(facts, scheme_override)
    layout = build_layout(facts, roster, interior=make_interior(models) if models else None)
    stamped = layout["stats"]["units_with_stamped_interior"]
    if layout["archetype"] == "floor_stack":
        tier = "stack-only"
    else:
        tier = "template+stamped" if stamped else "template"
    geo = facts.get("latlng")
    return BuildingModel(
        schema=SCHEMA_VERSION, id=f"prop-{facts['property_id']}", name=facts["name"], address=facts["address"],
        use_type=facts["use_type"], archetype=layout["archetype"], tier=tier, yaw_deg=0.0,
        geolocation=(float(geo[0]), float(geo[1])) if geo else None, levels=tuple(layout["levels"]),
        spaces=tuple(sorted(layout["spaces"], key=lambda s: s.id)), walls=tuple(layout["walls"]),
        links=tuple(sorted(layout["links"], key=lambda ln: (ln.a, ln.b, ln.kind))), entrances=tuple(layout["entrances"]),
        stats={**roster["stats"], **layout["stats"], "scheme": roster["scheme"], "stories": roster["stories"],
               "storey_height_m": roster["storey_height_m"], "height_source": roster["height_source"],
               "scans_stamped": sorted(models or {})},
        notes=tuple(layout["notes"]),
    )


def scan_plan(facts: dict, max_scans: int = MAX_SCANS) -> list[str]:
    """Scan ids worth fetching: the ones that cover the most units, most first."""
    counts: Counter = Counter()
    for u in facts.get("units") or []:
        for sid in (u.get("scans") or [])[:1]:
            counts[sid] += 1
    return [sid for sid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_scans]]


def _meta(bm: BuildingModel, building, context: Optional[dict], facts: Optional[dict], scans: dict) -> dict:
    rooms = {}
    by_id = {s.id: s for s in bm.spaces}
    scan_names = {}
    for rm in building.rooms:
        s = by_id.get(rm.id)
        attrs = dict(s.attrs) if s else {}
        entry = {"source": "costar" if s and s.kind in ("unit", "tenant") else
                 "matterport" if s and s.fidelity in ("surveyed", "cluster") else "inferred"}
        if attrs.get("tenant"):
            entry["tenant"] = attrs["tenant"]
        if attrs.get("scan"):
            entry["model_id"] = attrs["scan"]
            entry["sweep_id"] = None
            entry["note"] = "Scan of this unit type, not necessarily of this unit"
            scan_names.setdefault(attrs["scan"], s.raw_class or "Unit type")
        if s and s.fidelity == "surveyed" and ":" in s.source_ref:
            entry["model_id"], entry["scan_room_id"] = s.source_ref.split(":", 1)
            entry["sweep_id"] = None
        rooms[rm.id] = entry
    if bm.archetype in ("whole_scan", "scan_clusters"):
        models = [{"model_id": bm.id[3:], "suite": None, "name": bm.name, "floor": None}]
    else:
        models = [{"model_id": mid, "suite": None, "name": name, "floor": None}
                  for mid, name in sorted(scan_names.items(), key=lambda kv: (kv[1], kv[0]))[:12]]
    sources = Counter(r["source"] for r in rooms.values())
    return {
        "building_id": bm.id, "name": bm.name, "address": bm.address or None, "real": True,
        "kind": "generated" if bm.archetype not in ("whole_scan", "scan_clusters") else "scan_model",
        "center_latlng": (context or {}).get("centre_latlng") or (list(bm.geolocation) if bm.geolocation else None),
        "georeference": {"rotation_deg": (context or {}).get("yaw_deg", bm.yaw_deg)},
        "site": {"source": "scan" if bm.archetype in ("whole_scan", "scan_clusters") else "generated",
                 "imagery": (context or {}).get("imagery") or {},
                 "footprint_latlng": (context or {}).get("footprint_latlng") or [],
                 "outlines": (context or {}).get("outlines") or [],
                 "placement_source": (context or {}).get("placement_source"),
                 "layout_placed": (context or {}).get("layout_placed", True),
                 "address_latlng": (context or {}).get("geolocation"),
                 "buildings": int(bm.stats.get("blocks") or 1),
                 "pending": bool((context or {}).get("pending")),
                 # what is drawn on the site map: roads, car parks, neighbouring buildings, named places
                 "surroundings": (context or {}).get("surroundings")},
        "matterport_models": models, "room_sources": dict(sources), "rooms": rooms,
        "model": {"tier": bm.tier, "archetype": bm.archetype, "stats": bm.stats, "scans": scans,
                  "street": (context or {}).get("street"),
                  "surroundings": {k: ((context or {}).get("surroundings") or {}).get(k)
                                   for k in ("source", "counts", "own_buildings")}
                  if (context or {}).get("surroundings") else None},
        "caveats": list(bm.notes) + _fit_caveat(context),
    }


def _fit_caveat(context: Optional[dict]) -> list[str]:
    """Say so when the generated layout and the real building differ a lot in size."""
    fit = (context or {}).get("fit_scale")
    if not fit:
        return []
    worst = max(max(fit), 1.0 / max(min(fit), 1e-6))
    if worst < 1.3:
        return []
    return [f"The generated layout and the real building differ in size (the layout is drawn at {fit[0]:.2f}x by "
            f"{fit[1]:.2f}x to fit the building's outline); walking distances inside it are approximate."]


def _write(bm: BuildingModel, context: Optional[dict], facts: Optional[dict], scans: dict) -> dict:
    building = to_building(bm, context)
    bm.to_json_file(model_path(bm.id))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / f"{bm.id}.building.json").write_text(building.model_dump_json(), encoding="utf-8")
    (PROCESSED / f"{bm.id}.meta.json").write_text(json.dumps(_meta(bm, building, context, facts, scans)), encoding="utf-8")
    return {"rooms": len(building.rooms), "edges": len(building.edges), "entrances": len(building.entrances)}


def generate_scan(model_id: str, context: bool = True, offline: bool = False, refresh_context: bool = False) -> dict:
    """One whole-building Matterport scan -> BuildingModel (surveyed, or scan clusters when it has no walls)."""
    stages = _Stages()
    model, status = load_unit_model(model_id, offline=offline)
    stages.mark("matterport geometry", status)
    scan = None
    scan_path = CACHE_DIR / f"{model_id}.scan.json"
    try:
        from ingest.mp3d.clusters import SCAN_QUERY, scan_model_from_payload

        if scan_path.is_file():
            payload = json.loads(scan_path.read_text(encoding="utf-8"))
        elif not offline:
            from ingest.mp3d.client import MatterportClient

            payload = MatterportClient().query(SCAN_QUERY, {"id": model_id}).get("model")
            if payload:
                scan_path.parent.mkdir(parents=True, exist_ok=True)
                scan_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        else:
            payload = None
        scan = scan_model_from_payload(payload, model_id) if payload else None
        stages.mark("scan graph", "ok" if scan else "unavailable")
    except Exception as exc:  # noqa: BLE001
        stages.mark("scan graph", f"error: {str(exc)[:100]}")
    if model is not None:
        bm = from_model3d(model, scan, model_id)
    elif scan is not None:
        bm = from_scan_model(scan)
    else:
        raise RuntimeError(f"Matterport model {model_id}: {status}; nothing to build from")
    stages.mark("model", f"{bm.tier}: {len(bm.spaces)} rooms, {len(bm.walls)} walls")
    ctx = build_context(bm, network=context and not offline, refresh=refresh_context,
                        on_late=lambda late: _write(bm, late, None, {model_id: status}))
    stages.mark("site context", f"placed by {ctx.get('placement_source')}"
                                + ("; street lookup still running" if ctx.get("pending") else ""))
    counts = _write(bm, ctx, None, {model_id: status})
    stages.mark("project + write", f"{counts['rooms']} rooms, {counts['edges']} links")
    return _report(bm, stages, {model_id: status}, ctx)


def generate(query: str, stamp: bool = True, context: bool = True, offline: bool = False,
             max_scans: int = MAX_SCANS, refresh_context: bool = False) -> dict:
    """Run the whole pipeline for an address / property id / ``mp:<model id>`` and return the run report."""
    query = str(query).strip()
    if query.lower().startswith("mp:"):
        return generate_scan(query[3:].strip(), context=context, offline=offline, refresh_context=refresh_context)
    stages = _Stages()
    hits = index.lookup(query, limit=5)
    if not hits:
        raise KeyError(f"no property matches {query!r}")
    hit = hits[0]
    stages.mark("address lookup", f"{hit['address']}, {hit['city']} (property {hit['property_id']})")
    facts = property_facts(hit["property_id"])
    stages.mark("property record", f"{facts['use_type']}, {facts['stories']} storeys, "
                                   f"{len(facts['units'])} units, {len(facts['tenants'])} tenants")
    models, statuses = {}, {}
    if stamp:
        for sid in scan_plan(facts, max_scans):
            model, status = load_unit_model(sid, offline=offline)
            statuses[sid] = status
            if model is not None:
                models[sid] = model
        stages.mark("unit-type scans", f"{len(models)} buildable of {len(statuses)} checked" if statuses else "none listed")
    bm = build_property_model(facts, models)
    stages.mark("roster + layout", f"{bm.archetype}: {len(bm.spaces)} spaces, {len(bm.walls)} walls, "
                                   f"{bm.stats['units_with_stamped_interior']} units stamped")
    # if OpenStreetMap answers after the deadline, the building is re-projected with the better placement
    ctx = build_context(bm, facts.get("latlng"), network=context and not offline, refresh=refresh_context,
                        on_late=lambda late: _write(bm, late, facts, statuses))
    stages.mark("site context", f"placed by {ctx.get('placement_source')}"
                                + (f"; street: {ctx['street']['name']}" if ctx.get("street") and ctx["street"].get("name") else "")
                                + ("; street lookup still running" if ctx.get("pending") else ""))
    counts = _write(bm, ctx, facts, statuses)
    stages.mark("project + write", f"{counts['rooms']} rooms, {counts['edges']} links")
    report = _report(bm, stages, statuses, ctx)
    report["alternatives"] = hits[1:]
    return report


def _report(bm: BuildingModel, stages: _Stages, scans: dict, ctx: Optional[dict]) -> dict:
    return {
        "building_id": bm.id, "name": bm.name, "address": bm.address, "tier": bm.tier, "archetype": bm.archetype,
        "levels": len(bm.levels), "spaces": len(bm.spaces), "walls": len(bm.walls), "links": len(bm.links),
        "stats": bm.stats, "notes": list(bm.notes), "scans": scans, "street": (ctx or {}).get("street"),
        "stages": stages.items, "total_ms": round(sum(s["ms"] for s in stages.items), 1),
    }

"""The stages in order, timed: lookup -> record -> scans -> outline -> roster + layout -> context -> project -> write.

``generate(query)`` runs at call time. ``query`` is an address, a building name, a
property id, ``mp:<model id>`` for one whole-building Matterport scan, or ``plan:<id>`` for a
building traced from its published floor plans (``configs/plan_buildings.yaml``).

Outputs (all under ``data/processed/``, which is gitignored):

    models/<id>.model.json     the BuildingModel (deterministic)
    context/<id>.json          the site-context sidecar (changes over time)
    <id>.building.json         core.schema.Building, picked up by the app's building list
    <id>.meta.json             provenance for the web app

The run report (timings, scan fetch statuses) is returned, never written into the model.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ingest.unified import index
from ingest.unified.context import _assemble, build_context, layout_site, load_context, site_sources
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
MAX_SCANS = 6  # tuned: unit-type scans fetched per property (most-covering first); each is a Matterport request
PLAN_MAP_WAIT_S = 60.0  # how long a traced plan waits for OpenStreetMap (roads, neighbours) to answer


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


def floor_area(facts: dict, roster: Optional[dict] = None) -> Optional[float]:
    """The record's area of one floor in m2: the typical floor, else gross area over the storeys,
    else the units' areas over the storeys. None when the record has none of them."""
    if facts.get("typical_floor_m2"):
        return float(facts["typical_floor_m2"])
    stories = int(facts.get("stories") or (roster or {}).get("stories") or 0)
    if not stories:
        return None
    if facts.get("gross_m2"):
        return float(facts["gross_m2"]) / stories
    units = sum(float(u.get("area_m2") or 0.0) for u in facts.get("units") or [])
    return units / stories if units else None


def building_count(facts: dict, roster: Optional[dict] = None) -> int:
    """How many buildings the record describes: the count it states, or the buildings its units name."""
    roster = roster or build_roster(facts)
    return max(len({s["block"] for s in roster["slots"]}) or 1, int(facts.get("num_buildings") or 1))


def build_property_model(facts: dict, models: Optional[dict] = None, scheme_override: Optional[str] = None,
                         site: Optional[dict] = None) -> BuildingModel:
    """Layers 1-5 for one property: pure function of the record, the cached scans and the outline in ``site``.

    ``site`` (``context.layout_site``) is the building's real outline; without it the layout is the assumed bar.
    """
    roster = build_roster(facts, scheme_override)
    layout = build_layout(facts, roster, interior=make_interior(models) if models else None, site=site)
    stamped = layout["stats"]["units_with_stamped_interior"]
    if layout["archetype"] == "floor_stack":
        tier = "stack-only"
    else:
        tier = "template+stamped" if stamped else "template"
    geo = facts.get("latlng")
    return BuildingModel(
        schema=SCHEMA_VERSION, id=f"prop-{facts['property_id']}", name=facts["name"], address=facts["address"],
        use_type=facts["use_type"], archetype=layout["archetype"], tier=tier,
        yaw_deg=float((layout["stats"].get("shape") or {}).get("yaw_deg") or 0.0),
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
        entry = {"source": "floor_plan" if s and s.fidelity == "traced" else
                 "costar" if s and s.kind in ("unit", "tenant") else
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
        if s and s.fidelity == "traced":
            said = [f"capacity {attrs['capacity']}" if attrs.get("capacity") else "",
                    f"scheduled by {attrs['owner']}" if attrs.get("owner") else ""]
            if any(said):
                entry["note"] = ", ".join(t for t in said if t)
        rooms[rm.id] = entry
    tour = bm.stats.get("tour") if bm.archetype == "floor_plan" else None
    if tour:
        # a public tour of part of the building: shown for the rooms it covers, not measured
        for rid in tour.get("covers") or []:
            if rid in rooms:
                rooms[rid].update({"model_id": tour["model_id"], "sweep_id": None})
        models = [{"model_id": tour["model_id"], "suite": None, "name": tour.get("name") or bm.name, "floor": None}]
    elif bm.archetype == "floor_plan":
        models = []
    elif bm.archetype in ("whole_scan", "scan_clusters"):
        models = [{"model_id": bm.id[3:], "suite": None, "name": bm.name, "floor": None}]
    else:
        models = [{"model_id": mid, "suite": None, "name": name, "floor": None}
                  for mid, name in sorted(scan_names.items(), key=lambda kv: (kv[1], kv[0]))[:12]]  # display cap
    sources = Counter(r["source"] for r in rooms.values())
    return {
        "building_id": bm.id, "name": bm.name, "address": bm.address or None, "real": True,
        "kind": "floor_plan" if bm.archetype == "floor_plan" else
                "generated" if bm.archetype not in ("whole_scan", "scan_clusters") else "scan_model",
        "center_latlng": (context or {}).get("centre_latlng") or (list(bm.geolocation) if bm.geolocation else None),
        "georeference": {"rotation_deg": (context or {}).get("yaw_deg", bm.yaw_deg)},
        "site": {"source": "floor_plan" if bm.archetype == "floor_plan" else
                           "scan" if bm.archetype in ("whole_scan", "scan_clusters") else "generated",
                 "imagery": (context or {}).get("imagery") or {},
                 "footprint_latlng": (context or {}).get("footprint_latlng") or [],
                 "outlines": (context or {}).get("outlines") or [],
                 "placement_source": (context or {}).get("placement_source"),
                 "layout_placed": (context or {}).get("layout_placed", True),
                 "address_latlng": (context or {}).get("geolocation"),
                 "buildings": int(bm.stats.get("blocks") or 1),
                 "pending": bool((context or {}).get("pending")),
                 # "outline": every plate follows the real outline; "assumed": a bar sized from the unit list
                 "shape": (context or {}).get("shape") or ({k: v for k, v in (bm.stats.get("shape") or {}).items()
                                                            if k in ("kind", "source", "detail", "reason")} or None),
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
    if worst < 1.3:  # tuned: a stretch or squeeze under 30% on either axis is not worth a caveat
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


def generate_plan(building_id: str, context: bool = True, offline: bool = False, refresh_context: bool = False) -> dict:
    """A building traced from its published floor plans (``configs/plan_buildings.yaml``) -> BuildingModel."""
    from ingest.unified.plan import build_plan_model, plan_config

    stages = _Stages()
    cfg = plan_config(building_id)
    stages.mark("floor plan", f"{sum(len(lv.get('spaces') or []) for lv in cfg.get('levels') or [])} traced spaces "
                              f"on {len(cfg.get('levels') or [])} levels")
    bm = build_plan_model(cfg)
    stages.mark("model", f"{bm.tier}: {len(bm.spaces)} rooms, {len(bm.walls)} walls, {len(bm.links)} links")
    arrived = threading.Event()

    def late(found: dict) -> None:
        _write(bm, found, None, {})
        if not found.get("pending"):
            arrived.set()

    ctx = build_context(bm, network=context and not offline, refresh=refresh_context, on_late=late)
    if ctx.get("pending"):
        # A traced plan is prepared ahead of a call, so there is time to wait for the map: what
        # the windows look at comes from OpenStreetMap, which can take most of a minute.
        arrived.wait(PLAN_MAP_WAIT_S)
        ctx = load_context(bm.id) or ctx
    stages.mark("site context", f"placed by {ctx.get('placement_source')}"
                                + ("; street lookup still running" if ctx.get("pending") else ""))
    counts = _write(bm, ctx, None, {})
    stages.mark("project + write", f"{counts['rooms']} rooms, {counts['edges']} links")
    return _report(bm, stages, {}, ctx)


def generate(query: str, stamp: bool = True, context: bool = True, offline: bool = False,
             max_scans: int = MAX_SCANS, refresh_context: bool = False) -> dict:
    """Run the whole pipeline for an address / property id / ``mp:<model id>`` / ``plan:<id>`` and return the run report."""
    query = str(query).strip()
    if query.lower().startswith("mp:"):
        return generate_scan(query[3:].strip(), context=context, offline=offline, refresh_context=refresh_context)
    if query.lower().startswith("plan:"):
        return generate_plan(query[5:].strip(), context=context, offline=offline, refresh_context=refresh_context)
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
    # The outline comes first: the layout is drawn to it. Vexcel answers are kept on disk and
    # OpenStreetMap is local for the kit's properties, so this is milliseconds on a second run.
    geo = facts.get("latlng")
    roster = build_roster(facts)
    buildings = building_count(facts, roster)
    plate_m2 = floor_area(facts, roster)
    current: dict = {}
    ready = threading.Event()

    def late(state: dict) -> None:
        # a source answered after the deadline: lay the building out again if what it brought
        # (an outline, or the street that says which side is the front) changes the layout
        ready.wait(60.0)  # let the first write finish so a late answer cannot be overwritten by it (60 s cap)
        site = layout_site(state, geo, buildings, plate_m2)
        model = current["bm"] if _same_site(site, current["site"]) else build_property_model(facts, models, site=site)
        current.update(bm=model, site=site)
        _write(model, _assemble(model, geo, state, pending=not state.get("osm_done")), facts, statuses)

    sources = site_sources(f"prop-{facts['property_id']}", geo, single=(len({s["block"] for s in roster["slots"]}) or 1) == 1,
                           buildings=buildings, network=context and not offline, refresh=refresh_context, on_late=late)
    site = layout_site(sources, geo, buildings, plate_m2)
    stages.mark("building outline", (f"{site['source']}: {len(site['polygons_latlng'])} outline"
                                     f"{'' if len(site['polygons_latlng']) == 1 else 's'}") if site["polygons_latlng"]
                else site.get("reason", "none"))
    bm = build_property_model(facts, models, site=site)
    current.update(bm=bm, site=site)
    stages.mark("roster + layout", f"{bm.archetype}: {len(bm.spaces)} spaces, {len(bm.walls)} walls, "
                                   f"{bm.stats['units_with_stamped_interior']} units stamped; shape "
                                   f"{(bm.stats.get('shape') or {}).get('kind')}")
    ctx = build_context(bm, geo, sources=sources)
    stages.mark("site context", f"placed by {ctx.get('placement_source')}"
                                + (f"; street: {ctx['street']['name']}" if ctx.get("street") and ctx["street"].get("name") else "")
                                + ("; street lookup still running" if ctx.get("pending") else ""))
    counts = _write(bm, ctx, facts, statuses)
    ready.set()
    stages.mark("project + write", f"{counts['rooms']} rooms, {counts['edges']} links")
    report = _report(bm, stages, statuses, ctx)
    report["alternatives"] = hits[1:]
    return report


def _same_site(a: dict, b: dict) -> bool:
    keys = ("polygons_latlng", "holes_latlng", "street_bearing_deg", "source")
    return all((a or {}).get(k) == (b or {}).get(k) for k in keys)


def _report(bm: BuildingModel, stages: _Stages, scans: dict, ctx: Optional[dict]) -> dict:
    return {
        "building_id": bm.id, "name": bm.name, "address": bm.address, "tier": bm.tier, "archetype": bm.archetype,
        "levels": len(bm.levels), "spaces": len(bm.spaces), "walls": len(bm.walls), "links": len(bm.links),
        "stats": bm.stats, "notes": list(bm.notes), "scans": scans, "street": (ctx or {}).get("street"),
        "stages": stages.items, "total_ms": round(sum(s["ms"] for s in stages.items), 1),
    }

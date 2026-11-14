"""Assemble a real ``Building`` from CoStar + Matterport + a Vexcel site annotation.

    uv run python -m ingest.real_building bundy-olympic            # build from cached data
    uv run python -m ingest.real_building bundy-olympic --refresh  # re-query Matterport / CoStar

Writes ``data/processed/<id>.building.json`` (validates against the frozen
schema; ``ingest.matterport.list_buildings`` picks it up) and
``data/processed/<id>.meta.json`` (provenance per room, Matterport model and
sweep per scanned room, site annotation for the aerial view).

Where each fact comes from
--------------------------
* **CoStar** (``ingest.costar``): storeys, floor plate area, slab-to-slab height,
  the suite list per floor with rentable area, which suites have a current
  tenant (-> ``occupancy_factor``), elevator count.
* **Matterport** (``ingest.matterport_api``): the rooms inside each scanned
  suite with measured floor area, the scan's room tags (-> ``room_type``), and
  window openings where the scan has room boundaries.
* **Vexcel** (``configs/real_buildings.yaml``, annotated from ortho imagery): the
  roof outline, what each long facade looks at, entrances, staging spots.
* **Inferred** (no dataset has it): the building core on every floor (two
  stairwells, elevators, restrooms, corridors), the parking storeys, and where
  each suite sits along the corridor. The floor plan is therefore a schematic
  with real areas, not a survey. ``meta["rooms"][id]["source"]`` says which of
  ``matterport`` / ``costar`` / ``inferred`` a room is.
"""

from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from core.schema import Building, Room
from ingest import costar, matterport_api
from ingest.georegister import latlng_to_local, local_to_latlng
from ingest.matterport import polygon_centroid, rect_polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "real_buildings.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SF_TO_M2 = 0.09290304
FT_TO_M = 0.3048
DEFAULT_SLAB_TO_SLAB_M = 3.66
STAIR_RUN_M = 8.0
NEAR_RADIUS_M = 10.0
MIN_SCAN_ROOM_M2 = 2.0  # smaller scan "rooms" are shafts / voids
SEGMENTS = ("west", "center", "east")

# Matterport room tag -> room_type key of configs/clue_vocab.yaml. The scans'
# classifier is trained on homes, so office rooms come back with home labels.
TAG_TO_TYPE = {
    "corridor": "corridor",
    "hallway": "corridor",
    "lobby": "reception",
    "bedroom": "office",       # enclosed private office
    "office": "office",
    "closet": "storage",
    "laundry": "break_room",   # kitchenette / copy room with counter and sink
    "kitchen": "break_room",
    "bathroom": "restroom",
    "conference_room": "conference",
    "dining_room": "conference",
}
TYPE_LABEL = {
    "open_office": "open area", "office": "office", "storage": "closet", "break_room": "kitchenette",
    "corridor": "hall", "reception": "reception", "restroom": "restroom", "conference": "conference room",
}


@lru_cache(maxsize=4)
def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def real_building_configs() -> list[dict]:
    return list(_load_config(str(CONFIG_PATH)).get("buildings") or [])


def building_config(building_id: str) -> dict:
    for entry in real_building_configs():
        if entry["id"] == building_id:
            return entry
    raise KeyError(building_id)


# ---------------------------------------------------------------------------
# Matterport scan -> rooms of one suite
# ---------------------------------------------------------------------------


def _scan_room_type(room: dict, is_largest: bool) -> str:
    tags = [str(t).lower() for t in (room.get("tags") or []) if t]
    for tag in tags:
        if tag in TAG_TO_TYPE:
            return TAG_TO_TYPE[tag]
    area = float((room.get("dimensions") or {}).get("areaFloor") or 0.0)
    return "open_office" if is_largest or area >= 40.0 else "office"


def scan_rooms(model: dict) -> list[dict]:
    """Rooms of one scanned suite, largest first.

    ``[{"scan_room_id", "room_type", "area_m2", "has_windows", "windows_source",
        "sweep_id"}]``. ``has_windows`` comes from the scan's wall openings when
    the scan has room boundaries; older scans have none, and then enclosed work
    rooms are assumed to sit on the facade (``windows_source = "inferred"``).
    """
    raw = [
        r for r in (model.get("rooms") or [])
        if float((r.get("dimensions") or {}).get("areaFloor") or 0.0) >= MIN_SCAN_ROOM_M2
    ]
    raw.sort(key=lambda r: -float(r["dimensions"]["areaFloor"]))
    sweeps_by_room: dict[str, list[dict]] = {}
    for loc in model.get("locations") or []:
        rid = (loc.get("room") or {}).get("id")
        if rid and loc.get("position"):
            sweeps_by_room.setdefault(rid, []).append(loc)

    out = []
    for i, r in enumerate(raw):
        rtype = _scan_room_type(r, is_largest=(i == 0))
        edges = (r.get("boundary") or {}).get("edges") or []
        if edges:
            has_windows = any(o.get("type") == "window" for e in edges for o in (e.get("openings") or []))
            windows_source = "scan"
        else:
            has_windows = rtype in ("open_office", "office", "conference")
            windows_source = "inferred"
        sweeps = sweeps_by_room.get(r["id"]) or []
        sweep_id = None
        if sweeps:  # the sweep nearest the middle of the room's sweeps
            mx = sum(s["position"]["x"] for s in sweeps) / len(sweeps)
            my = sum(s["position"]["y"] for s in sweeps) / len(sweeps)
            sweep_id = min(
                sweeps, key=lambda s: (s["position"]["x"] - mx) ** 2 + (s["position"]["y"] - my) ** 2
            )["id"]
        out.append({
            "scan_room_id": r["id"],
            "room_type": rtype,
            "area_m2": round(float(r["dimensions"]["areaFloor"]), 2),
            "has_windows": has_windows,
            "windows_source": windows_source,
            "sweep_id": sweep_id,
        })
    return out


def _suite_from_name(name: str | None) -> Optional[str]:
    m = re.search(r"suite\s*#?\s*(\d+[A-Za-z]?)", name or "", flags=re.IGNORECASE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_real_building(building_id: str, refresh: bool = False) -> tuple[Building, dict]:
    """Assemble ``(Building, meta)`` for one entry of ``configs/real_buildings.yaml``."""
    cfg = building_config(building_id)
    prop = costar.load_property(int(cfg["costar_property_id"]), refresh=refresh)
    site = cfg["site"]
    plate = cfg["plate"]

    # ---- Matterport scans, keyed by suite ---------------------------------
    suite_of_model: dict[str, str] = {str(k): str(v) for k, v in (cfg.get("matterport_suites") or {}).items()}
    scans: dict[str, dict] = {}
    for entry in prop["matterport"]:
        mid = entry["model_id"]
        model = matterport_api.fetch_model(mid, refresh=refresh)
        suite = suite_of_model.get(mid) or _suite_from_name(model.get("name")) or _suite_from_name(entry.get("caption"))
        if suite:
            scans[suite] = model

    # ---- storeys ----------------------------------------------------------
    stories = int(prop.get("stories") or max(f["floor"] for f in prop["floors"]))
    floors = list(range(1, stories + 1))
    slab = float(prop["slab_to_slab_ft"]) * FT_TO_M if prop.get("slab_to_slab_ft") else DEFAULT_SLAB_TO_SLAB_M
    floor_heights = {f: round((f - 1) * slab, 3) for f in floors}
    parking_floors = {int(f) for f in cfg.get("parking_floors") or []}
    floor_sf = {f["floor"]: float(f["sf"] or prop["typical_floor_sf"]) for f in prop["floors"]}
    core_factor = float(prop.get("core_factor_pct") or 15.0) / 100.0

    # ---- plate geometry ---------------------------------------------------
    L = float(plate["length_m"])
    bands: dict[str, tuple[float, float]] = {}
    y = 0.0
    for band in ("south_row", "south_corridor", "core", "north_corridor", "north_row"):
        depth = float(plate["bands_m"][band])
        bands[band] = (y, y + depth)
        y += depth
    D = y
    rotation = float(plate.get("rotation_deg", 0.0))
    center = site["center_latlng"]
    theta = math.radians(rotation)
    # lat/lng of local (0, 0): the plate is centred on the roof-outline centroid
    half_e = (L / 2) * math.cos(theta) - (D / 2) * math.sin(theta)
    half_n = (L / 2) * math.sin(theta) + (D / 2) * math.cos(theta)
    origin = local_to_latlng(-half_e, -half_n, center, 0.0)
    seg_x = {name: (i * L / 3, (i + 1) * L / 3) for i, name in enumerate(SEGMENTS)}

    def segment_for(x: float) -> str:
        return SEGMENTS[min(2, max(0, int(x // (L / 3))))]

    rooms: list[Room] = []
    by_id: dict[str, Room] = {}
    edges: list[dict] = []
    edge_keys: set[tuple[str, str]] = set()
    room_meta: dict[str, dict] = {}

    def add_room(rid: str, name: str, floor: int, rtype: str, x0: float, x1: float, band: str, *,
                 area_m2: Optional[float] = None, windows: bool = False, facing: Optional[str] = None,
                 occupancy: float = 1.0, meta: Optional[dict] = None,
                 y_range: Optional[tuple[float, float]] = None) -> Room:
        y0, y1 = y_range or bands[band]
        poly = rect_polygon(x0, y0, x1, y1)
        cx, cy = polygon_centroid(poly)
        room = Room(
            id=rid, name=name, floor=floor, floor_label=str(floor), polygon=poly,
            centroid=(round(cx, 3), round(cy, 3), floor_heights[floor]),
            area_m2=round(area_m2 if area_m2 is not None else (x1 - x0) * (y1 - y0), 2),
            room_type=rtype, has_windows=windows, window_facing=facing if windows else None,
            near_elevator=False, near_stairs=False, occupancy_factor=float(occupancy),
        )
        if rid in by_id:
            raise ValueError(f"Duplicate room id {rid}")
        rooms.append(room)
        by_id[rid] = room
        room_meta[rid] = meta or {"source": "inferred"}
        return room

    def add_edge(u: str, v: str, kind: str = "door", length_m: Optional[float] = None, floors_delta: int = 0) -> None:
        key = (u, v) if u < v else (v, u)
        if key in edge_keys:
            return
        edge_keys.add(key)
        a, b = by_id[u], by_id[v]
        if length_m is None:
            length_m = math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])
        edges.append({"u": u, "v": v, "kind": kind, "length_m": round(float(length_m), 2), "floors": int(floors_delta)})

    suites_by_floor: dict[int, list[dict]] = {}
    for s in prop["suites"]:
        suites_by_floor.setdefault(int(s["floor"]), []).append(s)
    for suite in scans:  # a scanned suite CoStar has no current record for
        floor = int(suite[:-2]) if suite[:-2].isdigit() else None
        if floor in floors and not any(s["suite"] == suite for s in suites_by_floor.get(floor, [])):
            area_sf = sum(r["area_m2"] for r in scan_rooms(scans[suite])) / SF_TO_M2
            suites_by_floor.setdefault(floor, []).append(
                {"floor": floor, "suite": suite, "sf": area_sf, "status": "available", "tenant": None,
                 "industry": None, "employees": None, "available": True}
            )

    def suite_sort_key(s: dict) -> tuple:
        m = re.match(r"\d+", s["suite"])
        return (int(m.group()) if m else 10**6, s["suite"])

    for floor in floors:
        f = f"f{floor}"
        # ---- core + corridors (inferred) ----------------------------------
        for side, band in (("n", "north_corridor"), ("s", "south_corridor")):
            for seg in SEGMENTS:
                add_room(f"{f}-corridor-{side}-{seg}", f"{'North' if side == 'n' else 'South'} corridor ({seg})",
                         floor, "corridor", *seg_x[seg], band)
            for a, b in zip(SEGMENTS, SEGMENTS[1:]):
                add_edge(f"{f}-corridor-{side}-{a}", f"{f}-corridor-{side}-{b}", kind="corridor")
        mid = L / 2
        add_room(f"{f}-stair-a", "Stair A (west)", floor, "stairwell", 2.0, 7.0, "core")
        add_room(f"{f}-stair-b", "Stair B (east)", floor, "stairwell", L - 7.0, L - 2.0, "core")
        add_room(f"{f}-elev-lobby", "Elevator lobby", floor, "corridor", mid - 5.0, mid + 3.0, "core")
        add_room(f"{f}-elevators", "Elevators", floor, "elevator", mid + 3.0, mid + 9.0, "core",
                 meta={"source": "inferred", "note": f"CoStar lists {prop.get('elevators') or '?'} elevators"})
        add_edge(f"{f}-elevators", f"{f}-elev-lobby")
        for core_id, seg in ((f"{f}-stair-a", "west"), (f"{f}-stair-b", "east"), (f"{f}-elev-lobby", "center")):
            add_edge(core_id, f"{f}-corridor-n-{seg}")
            add_edge(core_id, f"{f}-corridor-s-{seg}")

        if floor in parking_floors:
            for side, band, label in (("n", "north_row", "north bay"), ("s", "south_row", "south bay")):
                add_room(f"{f}-parking-{side}", f"Parking level {floor} ({label})", floor, "parking", 0.0, L, band,
                         area_m2=L * D / 2, meta={"source": "inferred", "note": "no rentable floor in CoStar"})
                for seg in SEGMENTS:
                    add_edge(f"{f}-parking-{side}", f"{f}-corridor-{side}-{seg}")
            continue

        add_room(f"{f}-restroom-m", "Restroom (men)", floor, "restroom", mid - 17.0, mid - 11.0, "core")
        add_room(f"{f}-restroom-w", "Restroom (women)", floor, "restroom", mid - 11.0, mid - 5.0, "core")
        add_edge(f"{f}-restroom-m", f"{f}-corridor-n-center")
        add_edge(f"{f}-restroom-w", f"{f}-corridor-n-center")

        # ---- suites along the two perimeter rows --------------------------
        cells: list[dict] = [
            {"kind": "suite", "suite": s["suite"], "sf": float(s["sf"] or 0.0) or 1500.0, "record": s}
            for s in sorted(suites_by_floor.get(floor, []), key=suite_sort_key)
        ]
        rentable_sf = floor_sf.get(floor, float(prop["typical_floor_sf"])) * (1.0 - core_factor)
        lobby_sf = float(plate.get("lobby_sf") or 0.0) if floor == 1 else 0.0
        unlisted_sf = rentable_sf - lobby_sf - sum(c["sf"] for c in cells)
        if unlisted_sf > 0.1 * rentable_sf:
            cells.append({"kind": "unlisted", "suite": None, "sf": unlisted_sf, "record": None})
        if not cells:
            cells.append({"kind": "unlisted", "suite": None, "sf": rentable_sf, "record": None})

        # Split the suite sequence where the two rows are closest to equal area.
        total = sum(c["sf"] for c in cells) + lobby_sf
        best_k, best_gap, running = 1, float("inf"), lobby_sf
        for k, c in enumerate(cells, start=1):
            running += c["sf"]
            gap = abs(running - total / 2)
            if gap < best_gap and (k < len(cells) or len(cells) == 1):
                best_k, best_gap = k, gap
        north, south = cells[:best_k], cells[best_k:]
        if lobby_sf:
            north.insert(len(north) // 2 if len(north) > 1 else len(north),
                         {"kind": "lobby", "suite": None, "sf": lobby_sf, "record": None})
        south = south[::-1]  # numbering runs west->east on the street side and back on the parking side

        for row, side, band in ((north, "n", "north_row"), (south, "s", "south_row")):
            if not row:
                continue
            facing = site["facing"][band]
            row_sf = sum(c["sf"] for c in row)
            x = 0.0
            for cell in row:
                w = L * cell["sf"] / row_sf
                x0, x1 = x, x + w
                x = x1
                corridor = f"{f}-corridor-{side}-{segment_for((x0 + x1) / 2)}"
                rec = cell["record"]
                if cell["kind"] == "lobby":
                    add_room(f"{f}-lobby", "Main lobby", floor, "lobby", x0, x1, band,
                             area_m2=cell["sf"] * SF_TO_M2, windows=True, facing=facing,
                             meta={"source": "inferred", "note": "size and position assumed"})
                    add_edge(f"{f}-lobby", corridor)
                    add_edge(f"{f}-lobby", f"{f}-elev-lobby")
                    continue
                if cell["kind"] == "unlisted":
                    add_room(f"{f}-unlisted", "Unlisted floor area", floor, "suite", x0, x1, band,
                             area_m2=cell["sf"] * SF_TO_M2, windows=True, facing=facing,
                             occupancy=costar.UNKNOWN_FACTOR,
                             meta={"source": "inferred", "note": "rentable area with no CoStar suite record"})
                    add_edge(f"{f}-unlisted", corridor)
                    continue

                suite = cell["suite"]
                occupancy = costar.suite_occupancy_factor(rec)
                base_meta = {
                    "suite": suite, "status": rec.get("status"), "tenant": rec.get("tenant"),
                    "rentable_sf": round(cell["sf"]),
                }
                model = scans.get(suite)
                srooms = scan_rooms(model) if model else []
                if not srooms:
                    add_room(f"{f}-s{suite}", f"Suite {suite}", floor, "suite", x0, x1, band,
                             area_m2=cell["sf"] * SF_TO_M2, windows=True, facing=facing, occupancy=occupancy,
                             meta={"source": "costar", **base_meta})
                    add_edge(f"{f}-s{suite}", corridor)
                    continue

                scan_area = sum(r["area_m2"] for r in srooms)
                type_count: dict[str, int] = {}
                for r in srooms:
                    type_count[r["room_type"]] = type_count.get(r["room_type"], 0) + 1
                seen: dict[str, int] = {}
                # Schematic split of the suite cell: the largest room takes the full depth
                # at the west end; the others share the rest, windowed rooms on the facade
                # half and windowless ones on the corridor half.
                by0, by1 = bands[band]
                facade = (by0 + (by1 - by0) / 2, by1) if side == "n" else (by0, by0 + (by1 - by0) / 2)
                inner = (by0, by0 + (by1 - by0) / 2) if side == "n" else (by0 + (by1 - by0) / 2, by1)
                main_w = w * srooms[0]["area_m2"] / scan_area
                rest = list(enumerate(srooms))[1:]
                lanes = {
                    "facade": [(i, r) for i, r in rest if r["has_windows"]],
                    "inner": [(i, r) for i, r in rest if not r["has_windows"]],
                }
                slots: dict[int, tuple[float, float, tuple[float, float]]] = {0: (x0, x0 + main_w, (by0, by1))}
                for lane, members in lanes.items():
                    if not members:
                        continue
                    other = lanes["inner" if lane == "facade" else "facade"]
                    y_range = (by0, by1) if not other else (facade if lane == "facade" else inner)
                    lane_area = sum(r["area_m2"] for _, r in members)
                    lx = x0 + main_w
                    for i, r in members:
                        lw = (w - main_w) * r["area_m2"] / lane_area
                        slots[i] = (lx, lx + lw, y_range)
                        lx += lw
                main_id: Optional[str] = None
                for i, r in enumerate(srooms):
                    rx0, rx1, y_range = slots[i]
                    seen[r["room_type"]] = seen.get(r["room_type"], 0) + 1
                    label = TYPE_LABEL.get(r["room_type"], r["room_type"].replace("_", " "))
                    if type_count[r["room_type"]] > 1:
                        label = f"{label} {seen[r['room_type']]}"
                    rid = f"{f}-s{suite}-r{i + 1}"
                    add_room(rid, f"Suite {suite} {label}", floor, r["room_type"], rx0, rx1, band,
                             area_m2=r["area_m2"], windows=r["has_windows"], facing=facing, occupancy=occupancy,
                             y_range=y_range,
                             meta={"source": "matterport", **base_meta, "model_id": model["id"],
                                   "sweep_id": r["sweep_id"], "scan_room_id": r["scan_room_id"],
                                   "windows_source": r["windows_source"]})
                    if main_id is None:
                        main_id = rid
                        add_edge(rid, corridor)
                    else:
                        add_edge(rid, main_id)

    # ---- vertical circulation ---------------------------------------------
    for lower, upper in zip(floors, floors[1:]):
        rise = abs(floor_heights[upper] - floor_heights[lower])
        for slug in ("stair-a", "stair-b"):
            add_edge(f"f{lower}-{slug}", f"f{upper}-{slug}", kind="stairs", length_m=STAIR_RUN_M, floors_delta=1)
        add_edge(f"f{lower}-elevators", f"f{upper}-elevators", kind="elevator", length_m=rise, floors_delta=1)

    # ---- near_elevator / near_stairs --------------------------------------
    for floor in floors:
        on_floor = [r for r in rooms if r.floor == floor]
        elevators = [r for r in on_floor if r.room_type == "elevator"]
        stairs = [r for r in on_floor if r.room_type == "stairwell"]

        def near(room: Room, targets: list[Room]) -> bool:
            return any(
                math.hypot(room.centroid[0] - t.centroid[0], room.centroid[1] - t.centroid[1]) <= NEAR_RADIUS_M
                for t in targets
            )

        for room in on_floor:
            room.near_elevator = room.room_type == "elevator" or near(room, elevators)
            room.near_stairs = room.room_type == "stairwell" or near(room, stairs)

    # ---- entrances and staging --------------------------------------------
    def local_xy(latlng: list[float]) -> list[float]:
        lx, ly = latlng_to_local(latlng[0], latlng[1], origin, rotation)
        return [round(lx, 2), round(ly, 2)]

    opens = {"lobby": "f1-lobby", "south_corridor_west": "f1-corridor-s-west",
             "north_corridor_west": "f1-corridor-n-west"}
    entrances = []
    for e in site.get("entrances") or []:
        room_id = opens.get(e.get("opens_into", ""), e.get("opens_into", ""))
        if room_id not in by_id:
            room_id = "f1-elev-lobby"
        room = by_id[room_id]
        # On the schematic the door sits on the plate edge next to the room it opens into;
        # the lat/lng is the annotated position on the aerial.
        xs = [p[0] for p in room.polygon]
        if e.get("facing") == "street":
            xy = [round(room.centroid[0], 2), round(D, 2)]
        else:
            xy = [min(xs), round(room.centroid[1], 2)] if room.centroid[0] < L / 2 else [max(xs), round(room.centroid[1], 2)]
        entrances.append({"id": e["id"], "name": e["name"], "room_id": room_id, "xy": xy,
                          "latlng": [float(e["latlng"][0]), float(e["latlng"][1])], "facing": e["facing"]})
    entrance_ids = {e["id"] for e in entrances}
    staging = []
    for s in site.get("staging_spots") or []:
        if s["near_entrance_id"] not in entrance_ids:
            raise KeyError(f"Staging spot {s['id']} references unknown entrance {s['near_entrance_id']}")
        staging.append({"id": s["id"], "name": s["name"], "xy": local_xy(s["latlng"]),
                        "latlng": [float(s["latlng"][0]), float(s["latlng"][1])],
                        "kind": s["kind"], "near_entrance_id": s["near_entrance_id"]})

    building = Building(
        id=cfg["id"], name=cfg.get("name") or prop["name"], use_type=cfg.get("use_type", "office"),
        floors=floors, floor_heights_m=floor_heights, rooms=rooms, edges=edges,
        entrances=entrances, staging_spots=staging,
    )

    counts: dict[str, int] = {}
    for m in room_meta.values():
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    meta = {
        "building_id": building.id,
        "name": building.name,
        "address": prop["address"],
        "real": True,
        "center_latlng": [float(center[0]), float(center[1])],
        "georeference": {"origin_latlng": [round(origin[0], 7), round(origin[1], 7)], "rotation_deg": rotation,
                         "plate_m": [L, round(D, 2)]},
        "site": {
            "source": site.get("source"),
            "imagery": site.get("imagery") or {},
            "footprint_latlng": site.get("footprint_latlng") or [],
        },
        "costar": {k: prop.get(k) for k in ("property_id", "stories", "rba_sf", "typical_floor_sf",
                                             "slab_to_slab_ft", "elevators", "parking_spaces")},
        "matterport_models": [
            {"model_id": model["id"], "suite": suite, "name": (model.get("name") or "").strip(),
             "floor": int(suite[:-2]) if suite[:-2].isdigit() else None,
             "created": model.get("created"), "sweeps": len(model.get("locations") or []),
             "has_room_boundaries": any(((r.get("boundary") or {}).get("edges")) for r in model.get("rooms") or [])}
            for suite, model in sorted(scans.items())
        ],
        "room_sources": counts,
        "rooms": room_meta,
        "caveats": [
            "Suite positions along the corridor are not in any dataset; the floor plan is a schematic with real areas.",
            "Stairwells, elevators, restrooms, corridors and the parking storeys are inferred, not scanned.",
            "Entrances, staging spots and facade views are hand-annotated from Vexcel ortho imagery.",
        ],
    }
    return building, meta


def write_processed(building: Building, meta: dict) -> tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    b_path = PROCESSED_DIR / f"{building.id}.building.json"
    m_path = PROCESSED_DIR / f"{building.id}.meta.json"
    b_path.write_text(building.model_dump_json(indent=1), encoding="utf-8")
    m_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return b_path, m_path


def load_meta(building_id: str) -> dict[str, Any]:
    """Sidecar metadata for a building; a minimal dict for buildings without one (synthetic)."""
    path = PROCESSED_DIR / f"{building_id}.meta.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"building_id": building_id, "real": False, "matterport_models": [], "rooms": {},
            "site": {"imagery": {}, "footprint_latlng": []}, "caveats": []}


def main(argv: list[str]) -> None:
    ids = [a for a in argv if not a.startswith("--")] or [c["id"] for c in real_building_configs()]
    for building_id in ids:
        building, meta = build_real_building(building_id, refresh="--refresh" in argv)
        b_path, _ = write_processed(building, meta)
        per_floor = {f: sum(1 for r in building.rooms if r.floor == f) for f in building.floors}
        print(f"{building.id}: {len(building.rooms)} rooms, {len(building.edges)} edges, sources {meta['room_sources']}")
        print(f"  rooms per floor {per_floor}")
        print(f"  wrote {b_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Building registry and the synthetic generator.

``generate_synthetic_office()`` builds "100 Princeton Tech Center" (id
``ptc-100``), a 4-floor office building, from ``configs/demo_buildings.yaml``.
Real buildings are assembled by ``ingest.real_building`` (CoStar + Matterport +
Vexcel) into ``data/processed/*.building.json``; ``list_buildings()`` returns
the synthetic building first, then every processed one.

Public API (see the cross-module interface spec)::

    generate_synthetic_office() -> Building
    list_buildings() -> list[Building]
    load_building(building_id) -> Building        # KeyError if unknown
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from core.building_graph import with_inferred_elevators
from core.schema import Building, Room
from ingest.georegister import local_to_latlng

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "demo_buildings.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

_ND = 3  # decimal places kept in generated geometry


# ---------------------------------------------------------------------------
# Geometry helpers (general polygons, so real floor plans can reuse them)
# ---------------------------------------------------------------------------

def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    """Axis-aligned rectangle as a counter-clockwise polygon."""
    return [
        (round(x0, _ND), round(y0, _ND)),
        (round(x1, _ND), round(y0, _ND)),
        (round(x1, _ND), round(y1, _ND)),
        (round(x0, _ND), round(y1, _ND)),
    ]


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace area (always positive)."""
    s = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """Area centroid of a simple polygon."""
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-12:  # degenerate -> vertex mean
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    return (cx / (3.0 * a), cy / (3.0 * a))


def _planar_dist(a: Room, b: Room) -> float:
    return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _building_cfg(building_id: Optional[str] = None, config_path: Optional[os.PathLike] = None) -> dict:
    cfg = _load_config(str(config_path or CONFIG_PATH))
    entries = cfg.get("buildings") or []
    if not entries:
        raise ValueError(f"No buildings defined in {config_path or CONFIG_PATH}")
    if building_id is None:
        return entries[0]
    for entry in entries:
        if entry["id"] == building_id:
            return entry
    raise KeyError(building_id)


def build_from_config(cfg: dict) -> Building:
    """Turn one ``buildings[]`` entry of demo_buildings.yaml into a Building."""
    width = float(cfg["footprint_m"]["width"])
    floors = [int(f) for f in cfg["floors"]]
    floor_heights = {int(k): float(v) for k, v in cfg["floor_heights_m"].items()}
    floor_labels = {int(k): str(v) for k, v in (cfg.get("floor_labels") or {}).items()}
    bands = {k: (float(v[0]), float(v[1])) for k, v in cfg["bands"].items()}
    facing = cfg.get("facing", {})
    segments = [(float(a), float(b), str(name)) for a, b, name in cfg["corridor_segments"]]
    near_radius = float(cfg.get("near_radius_m", 8.0))
    stair_run = float(cfg.get("stair_run_m", 8.0))
    origin = cfg["origin_latlng"]
    rotation = float(cfg.get("rotation_deg", 0.0))

    rooms: list[Room] = []
    by_id: dict[str, Room] = {}
    edges: list[dict] = []
    edge_keys: set[tuple[str, str]] = set()

    def rid(floor: int, slug: str) -> str:
        return f"f{floor}-{slug}"

    def add_room(
        floor: int,
        slug: str,
        name: str,
        room_type: str,
        x0: float,
        x1: float,
        band: str,
        *,
        windows: bool,
        window_facing: Optional[str],
        occupancy: float = 1.0,
    ) -> Room:
        y0, y1 = bands[band]
        poly = rect_polygon(x0, y0, x1, y1)
        cx, cy = polygon_centroid(poly)
        room = Room(
            id=rid(floor, slug),
            name=name,
            floor=floor,
            floor_label=floor_labels.get(floor, str(floor)),
            polygon=poly,
            centroid=(round(cx, _ND), round(cy, _ND), floor_heights[floor]),
            area_m2=round(polygon_area(poly), 2),
            room_type=room_type,
            has_windows=windows,
            window_facing=window_facing if windows else None,
            near_elevator=False,  # filled in below
            near_stairs=False,
            occupancy_factor=float(occupancy),
        )
        if room.id in by_id:
            raise ValueError(f"Duplicate room id {room.id}")
        rooms.append(room)
        by_id[room.id] = room
        return room

    def add_edge(u: str, v: str, kind: Optional[str] = None, length_m: Optional[float] = None, floors_delta: int = 0) -> None:
        if u not in by_id or v not in by_id:
            raise KeyError(f"Edge endpoint missing: {u} -- {v}")
        key = (u, v) if u < v else (v, u)
        if key in edge_keys:
            return
        edge_keys.add(key)
        a, b = by_id[u], by_id[v]
        if kind is None:
            kind = "corridor" if a.room_type == "corridor" and b.room_type == "corridor" else "door"
        if length_m is None:
            length_m = _planar_dist(a, b)
        edges.append({"u": u, "v": v, "kind": kind, "length_m": round(float(length_m), 2), "floors": int(floors_delta)})

    def segment_for(x: float) -> str:
        for a, b, name in segments:
            if a <= x <= b:
                return name
        return min(segments, key=lambda s: abs((s[0] + s[1]) / 2 - x))[2]

    for floor in floors:
        program = cfg["floor_programs"][floor]

        # Corridors: two east-west runs split into segments.
        for side, band in (("n", "north_corridor"), ("s", "south_corridor")):
            side_name = "North" if side == "n" else "South"
            for a, b, name in segments:
                add_room(
                    floor, f"corridor-{side}-{name}", f"{side_name} Corridor ({name.capitalize()})",
                    "corridor", a, b, band, windows=False, window_facing=None,
                )
            for (_, _, n0), (_, _, n1) in zip(segments, segments[1:]):
                add_edge(rid(floor, f"corridor-{side}-{n0}"), rid(floor, f"corridor-{side}-{n1}"))

        # Core band (interior: never has windows).
        core_specs = list(cfg["core_fixed"]) + list(program.get("core", []))
        for spec in core_specs:
            x0, x1 = float(spec["x"][0]), float(spec["x"][1])
            add_room(
                floor, spec["slug"], spec["name"], spec["type"], x0, x1, "core",
                windows=False, window_facing=None, occupancy=spec.get("occupancy", 1.0),
            )
        for spec in core_specs:
            room = by_id[rid(floor, spec["slug"])]
            if room.room_type == "corridor":
                # Cross-corridor link joins both east-west corridors.
                seg = segment_for(room.centroid[0])
                add_edge(room.id, rid(floor, f"corridor-n-{seg}"))
                add_edge(room.id, rid(floor, f"corridor-s-{seg}"))
                continue
            door = spec.get("door", "north")
            if door in ("north", "south"):
                seg = segment_for(room.centroid[0])
                add_edge(room.id, rid(floor, f"corridor-{door[0]}-{seg}"))
            else:
                add_edge(room.id, rid(floor, door))

        # Perimeter rows (west -> east).
        for band, side in (("north_row", "n"), ("south_row", "s")):
            x = 0.0
            for spec in program[band]:
                w = float(spec["width"])
                windows = bool(spec.get("windows", True))
                room = add_room(
                    floor, spec["slug"], spec["name"], spec["type"], x, x + w, band,
                    windows=windows, window_facing=facing.get(band),
                    occupancy=spec.get("occupancy", 1.0),
                )
                x += w
                add_edge(room.id, rid(floor, f"corridor-{side}-{segment_for(room.centroid[0])}"))
            if abs(x - width) > 1e-6:
                raise ValueError(f"Floor {floor} {band}: widths sum to {x}, footprint is {width}")

        for u, v in program.get("extra_edges", []):
            add_edge(rid(floor, u), rid(floor, v))

    # Vertical circulation between consecutive floors.
    for lower, upper in zip(floors, floors[1:]):
        rise = abs(floor_heights[upper] - floor_heights[lower])
        for room in [r for r in rooms if r.floor == lower and r.room_type in ("stairwell", "elevator")]:
            slug = room.id.split("-", 1)[1]
            twin = rid(upper, slug)
            if twin not in by_id:
                continue
            if room.room_type == "stairwell":
                add_edge(room.id, twin, kind="stairs", length_m=stair_run, floors_delta=1)
            else:
                add_edge(room.id, twin, kind="elevator", length_m=rise, floors_delta=1)

    # near_elevator / near_stairs flags.
    for floor in floors:
        on_floor = [r for r in rooms if r.floor == floor]
        elevators = [r for r in on_floor if r.room_type == "elevator"]
        stairs = [r for r in on_floor if r.room_type == "stairwell"]
        for room in on_floor:
            room.near_elevator = room.room_type == "elevator" or any(
                _planar_dist(room, e) <= near_radius for e in elevators
            )
            room.near_stairs = room.room_type == "stairwell" or any(
                _planar_dist(room, s) <= near_radius for s in stairs
            )

    def latlng(xy: list[float]) -> list[float]:
        lat, lng = local_to_latlng(float(xy[0]), float(xy[1]), origin, rotation)
        return [round(lat, 7), round(lng, 7)]

    entrances = []
    for e in cfg.get("entrances", []):
        if e["room"] not in by_id:
            raise KeyError(f"Entrance {e['id']} opens into unknown room {e['room']}")
        entrances.append({
            "id": e["id"], "name": e["name"], "room_id": e["room"],
            "xy": [float(e["xy"][0]), float(e["xy"][1])], "latlng": latlng(e["xy"]),
            "facing": e["facing"],
        })
    entrance_ids = {e["id"] for e in entrances}
    staging = []
    for s in cfg.get("staging_spots", []):
        if s["near_entrance_id"] not in entrance_ids:
            raise KeyError(f"Staging spot {s['id']} references unknown entrance {s['near_entrance_id']}")
        staging.append({
            "id": s["id"], "name": s["name"],
            "xy": [float(s["xy"][0]), float(s["xy"][1])], "latlng": latlng(s["xy"]),
            "kind": s["kind"], "near_entrance_id": s["near_entrance_id"],
        })

    return Building(
        id=cfg["id"],
        name=cfg["name"],
        use_type=cfg["use_type"],
        floors=floors,
        floor_heights_m=floor_heights,
        rooms=rooms,
        edges=edges,
        entrances=entrances,
        staging_spots=staging,
    )


def generate_synthetic_office(config_path: Optional[os.PathLike] = None) -> Building:
    """Synthetic 4-floor office "100 Princeton Tech Center" (id ``ptc-100``).

    Deterministic: same YAML -> same Building. Returns a fresh object each call.
    """
    return build_from_config(_building_cfg(None, config_path))


# ---------------------------------------------------------------------------
# Building registry
# ---------------------------------------------------------------------------

def _processed_buildings() -> list[Building]:
    """Buildings exported by real ingestion: ``data/processed/*.building.json``.

    Empty in v0. Files that fail schema validation are skipped, not fatal.
    """
    out: list[Building] = []
    if not PROCESSED_DIR.is_dir():
        return out
    for path in sorted(PROCESSED_DIR.glob("*.building.json")):
        try:
            out.append(Building.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a bad export must not take the API down
            continue
    return out


def list_buildings() -> list[Building]:
    """All known buildings. v0: the synthetic office (+ any processed exports).

    Each one carries the elevator links its elevator lobbies imply
    (``core.building_graph.with_inferred_elevators``; none where the data draws the elevator).
    """
    buildings = [generate_synthetic_office()]
    seen = {b.id for b in buildings}
    for b in _processed_buildings():
        if b.id not in seen:
            buildings.append(b)
            seen.add(b.id)
    return [with_inferred_elevators(b) for b in buildings]


def load_building(building_id: str) -> Building:
    """Return one building by id. Raises ``KeyError`` if unknown."""
    for b in list_buildings():
        if b.id == building_id:
            return b
    raise KeyError(building_id)


# ---------------------------------------------------------------------------
# Real Matterport ingestion - STUB (needs dataset + keys; do not implement in v0)
# ---------------------------------------------------------------------------

def matterport_data_dir() -> Path:
    return Path(os.environ.get("MATTERPORT_DATA_DIR", "./data/raw/matterport"))


def load_matterport_scan(path: os.PathLike | str) -> Building:
    """Not used: a single scan is one tenant suite, never a whole building.

    Real buildings are assembled in ``ingest.real_building`` from the CoStar
    suite list, the Matterport scans of individual suites (fetched through the
    Model API by ``ingest.matterport_api``) and a Vexcel site annotation. The
    result is written to ``data/processed/<id>.building.json``, which
    ``list_buildings()`` picks up.
    """
    raise NotImplementedError(
        "Use `uv run python -m ingest.real_building <building id>`; see ingest/real_building.py."
    )


def _summary(building: Building) -> dict[str, Any]:
    per_floor = {f: sum(1 for r in building.rooms if r.floor == f) for f in building.floors}
    return {"id": building.id, "name": building.name, "rooms": len(building.rooms),
            "rooms_per_floor": per_floor, "edges": len(building.edges)}


if __name__ == "__main__":
    print(_summary(generate_synthetic_office()))

"""Assembly: a GEOMETRY payload becomes a :class:`Model3D`.

:func:`model_from_payload` is pure - give it a saved payload and it returns the
same model every time, with no network and no credentials. :func:`build_model`
is the thin wrapper that fetches first.

Determinism is explicit, not incidental: rooms and walls are sorted by their
Matterport ids, floors by ``sequence``, every float is rounded at a fixed
precision, and nothing depends on dict iteration order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ingest.mp3d import geometry as geo
from ingest.mp3d import slabs as slabmod
from ingest.mp3d import walls as wallmod
from ingest.mp3d.client import MatterportClient
from ingest.mp3d.fetch import fetch_geometry
from ingest.mp3d.preflight import TIER_DEGRADED, TIER_FULL

SCHEMA_VERSION = "mp3d/1"
MIN_ROOM_AREA_M2 = 1.5
DROP_CLASSIFICATIONS = frozenset({"Void"})
AREA_DP = 3


@dataclass(frozen=True)
class Opening:
    """A hole in a wall. ``t`` is a fraction along the wall, ``z`` is metres."""

    kind: str
    t0: float
    t1: float
    z0: float
    z1: float


@dataclass(frozen=True)
class Wall:
    id: str
    floor: int
    a: tuple[float, float]
    b: tuple[float, float]
    length_m: float
    thickness_m: float
    thickness_measured: bool
    exterior: bool
    open_boundary: bool
    room_ids: tuple[str, ...]
    openings: tuple[Opening, ...]


@dataclass(frozen=True)
class RoomPlate:
    id: str
    floor: int
    classification: str
    polygon: tuple[tuple[float, float], ...]
    centroid: tuple[float, float]
    area_m2: float
    triangles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Slab:
    floor: int
    label: str
    z0: float
    height_m: float
    elevation_source: str
    elevation_samples: int


@dataclass(frozen=True)
class Model3D:
    """A building as stacked slabs, walls with holes, and floor plates."""

    schema: str
    model_id: str
    name: str
    units: str
    yaw_deg: float
    tier: str
    slabs: tuple[Slab, ...]
    rooms: tuple[RoomPlate, ...]
    walls: tuple[Wall, ...]
    bounds: dict
    stats: dict
    notes: tuple[str, ...] = field(default_factory=tuple)
    # WGS84 position, when the scan carries one. Needed to look the building up
    # in any external source; about 79% of the corpus has it. Static, so it does
    # not threaten determinism - unlike the aerial sidecar keyed off it.
    geolocation: Optional[tuple[float, float]] = None

    # ---- derived helpers ------------------------------------------------
    def slab(self, floor: int) -> Optional[Slab]:
        for s in self.slabs:
            if s.floor == floor:
                return s
        return None

    def panels(self, wall: Wall) -> list[wallmod.Panel]:
        """Solid panels of ``wall`` once its openings are cut out."""
        s = self.slab(wall.floor)
        if s is None:
            return []
        spans = [(o.t0, o.t1, o.z0, o.z1) for o in wall.openings]
        return wallmod.solid_panels(spans, s.z0, s.height_m)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 1) -> str:
        """Canonical JSON. Sorted keys and fixed precision, so byte-stable."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_json_file(self, path: str | Path, indent: int | None = 1) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf8", newline="\n") as handle:
            handle.write(self.to_json(indent=indent))
        return target


def _geolocation(payload: dict) -> Optional[tuple[float, float]]:
    """(lat, lng) when the scan is georeferenced, else None."""
    geo = payload.get("geolocation") or {}
    lat, lng = geo.get("lat"), geo.get("long")
    if lat in (None, "", "null") or lng in (None, "", "null"):
        return None
    try:
        return (float(lat), float(lng))
    except (TypeError, ValueError):
        return None


def _classification(room: dict) -> str:
    entries = room.get("classifications") or []
    if not entries:
        return "Other"
    return str(entries[0].get("label") or "Other")


def model_from_payload(payload: dict, model_id: Optional[str] = None) -> Model3D:
    """Build a :class:`Model3D` from a GEOMETRY payload. Pure."""
    mid = model_id or payload.get("id") or ""
    floors = payload.get("floors") or []
    yaw = geo.yaw_degrees((payload.get("geocoordinates") or {}).get("rotation"))

    elevations = slabmod.floor_elevations(floors)
    datum_z = slabmod.datum(elevations)
    heights = slabmod.floor_heights(elevations)
    slabs = tuple(
        Slab(
            floor=e.floor,
            label=e.label,
            z0=slabmod.shift_to_datum(e.raw_z, datum_z),
            height_m=heights.get(e.floor, slabmod.DEFAULT_FLOOR_HEIGHT_M),
            elevation_source=e.source,
            elevation_samples=e.sample_size,
        )
        for e in sorted(elevations, key=lambda e: e.floor)
    )

    # ---- room plates --------------------------------------------------
    plates: list[RoomPlate] = []
    dropped_void = dropped_small = dropped_ring = 0
    rooms_seen = 0
    for floor in sorted(floors, key=lambda f: f.get("sequence") or 0):
        sequence = int(floor.get("sequence") or 0)
        for room in sorted(floor.get("rooms") or [], key=lambda r: r.get("id") or ""):
            rooms_seen += 1
            edges = (room.get("boundary") or {}).get("edges") or []
            ring = geo.ring_from_edges(edges)
            if ring is None:
                dropped_ring += 1
                continue
            label = _classification(room)
            area = geo.polygon_area(ring)
            if label in DROP_CLASSIFICATIONS:
                dropped_void += 1
                continue
            if area < MIN_ROOM_AREA_M2:
                dropped_small += 1
                continue
            poly = geo.as_ccw(ring)
            plates.append(
                RoomPlate(
                    id=str(room.get("id") or ""),
                    floor=sequence,
                    classification=label,
                    polygon=tuple(geo.round_point(p) for p in poly),
                    centroid=geo.round_point(geo.centroid(poly)),
                    area_m2=round(area, AREA_DP),
                    triangles=tuple(geo.triangulate(poly)),
                )
            )
    plates.sort(key=lambda p: (p.floor, p.id))

    # ---- walls --------------------------------------------------------
    raw_walls = wallmod.dedupe_walls(floors, datum_z)
    wall_objs = tuple(
        Wall(
            id=w["id"],
            floor=w["floor"],
            a=w["a"],
            b=w["b"],
            length_m=w["length_m"],
            thickness_m=w["thickness_m"],
            thickness_measured=w["thickness_measured"],
            exterior=w["exterior"],
            open_boundary=w["open_boundary"],
            room_ids=w["room_ids"],
            openings=tuple(
                Opening(kind=s[0], t0=s[1], t1=s[2], z0=s[3], z1=s[4])
                for s in w["openings"]
            ),
        )
        for w in raw_walls
    )

    # ---- bounds -------------------------------------------------------
    xs: list[float] = []
    ys: list[float] = []
    for plate in plates:
        for x, y in plate.polygon:
            xs.append(x)
            ys.append(y)
    for wall in wall_objs:
        xs.extend([wall.a[0], wall.b[0]])
        ys.extend([wall.a[1], wall.b[1]])
    top = slabs[-1] if slabs else None
    bounds = {
        "x": [round(min(xs), 4), round(max(xs), 4)] if xs else [0.0, 0.0],
        "y": [round(min(ys), 4), round(max(ys), 4)] if ys else [0.0, 0.0],
        "z": [0.0, round(top.z0 + top.height_m, 3) if top else 0.0],
    }

    # ---- stats and provenance ----------------------------------------
    kinds: dict[str, int] = {}
    for kind in wallmod.iter_opening_kinds(raw_walls):
        kinds[kind] = kinds.get(kind, 0) + 1

    # An opening's elevation comes from the scan, the slab elevation is a median
    # of doorway sills, and real buildings have double-height glazing - so some
    # openings straddle or sit outside their own storey band. Panel generation
    # clamps them, but silent clamping would hide genuinely misplaced data.
    band = {s.floor: (s.z0, s.z0 + s.height_m) for s in slabs}
    clipped = outside = 0
    for wall in wall_objs:
        lo, hi = band.get(wall.floor, (0.0, 0.0))
        for opening in wall.openings:
            overlap = min(opening.z1, hi) - max(opening.z0, lo)
            span = opening.z1 - opening.z0
            if overlap <= 0.01:
                outside += 1
            elif overlap < span - 0.01:
                clipped += 1
    fallback_floors = [
        s.floor for s in slabs if s.elevation_source != slabmod.SOURCE_DOORWAY
    ]
    odd_gaps = slabmod.implausible_gaps(elevations)
    tier = (
        TIER_FULL
        if not fallback_floors and not odd_gaps and kinds
        else TIER_DEGRADED
    )

    stats = {
        "floors": len(slabs),
        "rooms_in_payload": rooms_seen,
        "room_plates": len(plates),
        "dropped_no_ring": dropped_ring,
        "dropped_void": dropped_void,
        "dropped_below_min_area": dropped_small,
        "walls": len(wall_objs),
        "walls_exterior": sum(1 for w in wall_objs if w.exterior),
        "walls_open_boundary": sum(1 for w in wall_objs if w.open_boundary),
        "walls_thickness_measured": sum(1 for w in wall_objs if w.thickness_measured),
        "openings": sum(len(w.openings) for w in wall_objs),
        "openings_by_kind": dict(sorted(kinds.items())),
        "floor_area_m2": round(sum(p.area_m2 for p in plates), AREA_DP),
        "openings_clipped_to_storey": clipped,
        "openings_outside_storey": outside,
        "storeys_plausible": not odd_gaps,
        "min_floor_gap_m": (min(s.height_m for s in slabs) if slabs else None),
    }

    notes: list[str] = []
    for slab in slabs:
        if slab.elevation_source == slabmod.SOURCE_DOORWAY:
            notes.append(
                f"floor {slab.floor} slab from {slab.elevation_samples} doorway sills"
            )
        elif slab.elevation_source == slabmod.SOURCE_PANO:
            notes.append(
                f"floor {slab.floor} slab from pano z fallback "
                f"({slab.elevation_samples} scans) - less accurate"
            )
        else:
            notes.append(f"floor {slab.floor} slab elevation assumed (no data)")
    if len(slabs) == 1:
        notes.append("single floor: height is the 3.0 m default, not measured")
    elif slabs:
        notes.append(
            f"top floor height is the mean of the measured spacings "
            f"({slabs[-1].height_m} m), not measured"
        )
    for lower, upper, gap in odd_gaps:
        if gap <= slabmod.MIN_STACK_GAP_M:
            notes.append(
                f"floors {lower} and {upper} share one plane (gap {gap} m): Matterport "
                "segmented a single storey into two floors, so these rooms tile one "
                "storey rather than stacking - the geometry renders as one storey"
            )
        else:
            notes.append(
                f"floors {lower} and {upper} are {gap} m apart, below a plausible "
                "storey: a split-level or half-flight, so one floor index here is not "
                "one storey"
            )
    if outside:
        notes.append(
            f"{outside} opening(s) sit wholly outside their floor's storey band - "
            "mis-elevated in the scan, or stairwell glazing attached to the wrong "
            "floor; they are clamped away rather than cutting a hole"
        )
    if clipped:
        notes.append(
            f"{clipped} opening(s) straddle a slab boundary and are clipped to their "
            "storey (double-height glazing, or a sill a few cm off the median slab)"
        )
    if stats["walls_open_boundary"]:
        notes.append(
            f"{stats['walls_open_boundary']} walls are open boundaries "
            "(Matterport 'invisible' edges) - no wall exists there"
        )
    if not kinds:
        notes.append("no dimensioned openings: walls are solid")

    return Model3D(
        schema=SCHEMA_VERSION,
        model_id=mid,
        name=str(payload.get("name") or ""),
        units="metres",
        yaw_deg=round(yaw, 4),
        tier=tier,
        slabs=slabs,
        rooms=tuple(plates),
        walls=wall_objs,
        bounds=bounds,
        stats=stats,
        notes=tuple(notes),
        geolocation=_geolocation(payload),
    )


def build_model(
    model_id: str, client: Optional[MatterportClient] = None
) -> Model3D:
    """Fetch and assemble. One network call."""
    return model_from_payload(fetch_geometry(model_id, client=client), model_id)

"""Scan-cluster build, for models with no measured wall geometry.

Why this exists. A survey of 70 models found that the structural layer arrives
as one indivisible bundle: outlines, wall thickness, dimensioned openings,
``Edge.type`` and room classifications were present in *exactly* the same 37
models, and the other 33 had **zero boundary edges between them** across 147
rooms. So a reduced build cannot be made from "outlines without thickness" -
there are no outlines to use.

What those 33 models *do* carry, on every room:

* ``panoLocations`` / ``locations`` - scan positions with x, y, z
* ``AnchorLocation.neighbors`` - the scan graph, fully populated
* ``dimensions.areaFloor`` - floor area (imperial)

That is enough for a different and weaker model: rooms as clusters of scan
positions rather than polygons. It gives a centroid, a floor, an area and - from
the anchor graph - real measured connectivity, which is stronger than the full
build's wall-adjacency inference. It gives no walls, no openings and no window
views.

Use it for locating and routing, never for geometry. :class:`ScanModel` is a
separate type from :class:`~ingest.mp3d.model3d.Model3D` precisely so the two
cannot be confused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ingest.mp3d import geometry as geo
from ingest.mp3d import slabs as slabmod
from ingest.mp3d.client import MatterportClient

SCHEMA_VERSION = "mp3d-scan/1"
FT2_TO_M2 = 0.09290304  # exact: 0.3048 m/ft squared
COORD_DP = 3  # 1 mm rounding, for byte-identical output across runs

SCAN_QUERY = """
query Scan($id: ID!) {
  model(id: $id) {
    id
    name
    state
    geocoordinates { rotation { w x y z } }
    propertyClassifications { classification type }
    locations {
      id
      neighbors
      position { x y z }
      room { id }
      floor { sequence }
    }
    floors {
      sequence
      label
      panoLocations { position { z } }
      rooms {
        id
        label
        classifications { label }
        dimensions { areaFloor units }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class ScanRoom:
    """A room known only as a cluster of scan positions."""

    id: str
    floor: int
    classification: str
    centroid: tuple[float, float]
    z_mean: float
    area_m2: Optional[float]
    anchor_count: int
    hull: tuple[tuple[float, float], ...]
    hull_area_m2: float
    # Synthetic stand-in footprint: a square of the reported floor area centred
    # on the cluster. The hull is kept for reference but is a poor footprint -
    # measured median hull/area was 0.12, because a few scan points inside a
    # room enclose almost none of it. Anything needing a polygon should use
    # this and treat it as synthetic.
    footprint: tuple[tuple[float, float], ...] = ()
    footprint_source: str = "synthetic-square"


@dataclass(frozen=True)
class ScanLink:
    """Measured adjacency between two rooms, from the anchor graph."""

    a: str
    b: str
    anchor_edges: int
    gap_m: float
    floors: int


@dataclass(frozen=True)
class ScanModel:
    schema: str
    model_id: str
    name: str
    units: str
    yaw_deg: float
    level: str
    slabs: tuple[slabmod.FloorElevation, ...]
    floor_heights: dict[int, float]
    rooms: tuple[ScanRoom, ...]
    links: tuple[ScanLink, ...]
    stats: dict
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 1) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_json_file(self, path: str | Path, indent: int | None = 1) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf8", newline="\n") as handle:
            handle.write(self.to_json(indent=indent))
        return target


def _classification(room: dict) -> str:
    entries = room.get("classifications") or []
    if entries and entries[0].get("label"):
        return str(entries[0]["label"])
    return str(room.get("label") or "Unknown")


def scan_model_from_payload(
    payload: dict, model_id: Optional[str] = None
) -> ScanModel:
    """Assemble a :class:`ScanModel` from a SCAN payload. Pure."""
    mid = model_id or payload.get("id") or ""
    yaw = geo.yaw_degrees((payload.get("geocoordinates") or {}).get("rotation"))
    floor_list = payload.get("floors") or []

    meta: dict[str, dict] = {}
    for floor in floor_list:
        sequence = int(floor.get("sequence") or 0)
        for room in floor.get("rooms") or []:
            rid = room.get("id")
            if not rid:
                continue
            dims = room.get("dimensions") or {}
            raw_area = dims.get("areaFloor")
            units = (dims.get("units") or "imperial").lower()
            area = None
            if raw_area is not None:
                area = round(
                    float(raw_area) * (FT2_TO_M2 if units == "imperial" else 1.0), 2
                )
            meta[rid] = {
                "floor": sequence,
                "classification": _classification(room),
                "area_m2": area,
            }

    # Scan anchors (x, y, z + scan-graph neighbours), grouped by the room Matterport assigned them to.
    anchors: dict[str, dict] = {}
    by_room: dict[str, list[str]] = {}
    for anchor in payload.get("locations") or []:
        aid = anchor.get("id")
        pos = anchor.get("position") or {}
        if not aid or pos.get("x") is None or pos.get("y") is None:
            continue
        rid = (anchor.get("room") or {}).get("id")
        floor = (anchor.get("floor") or {}).get("sequence")
        anchors[aid] = {
            "x": float(pos["x"]),
            "y": float(pos["y"]),
            "z": float(pos.get("z") or 0.0),
            "room": rid,
            "floor": int(floor) if floor is not None else None,
            "neighbors": tuple(sorted(anchor.get("neighbors") or [])),
        }
        if rid:
            by_room.setdefault(rid, []).append(aid)

    elevations = slabmod.floor_elevations(floor_list)
    datum_z = slabmod.datum(elevations)
    heights = slabmod.floor_heights(elevations)

    rooms: list[ScanRoom] = []
    for rid in sorted(by_room):
        ids = sorted(by_room[rid])
        pts = [(anchors[a]["x"], anchors[a]["y"]) for a in ids]
        zs = [anchors[a]["z"] for a in ids]
        info = meta.get(rid) or {}
        floor = info.get("floor")
        if floor is None:
            floors_seen = [anchors[a]["floor"] for a in ids if anchors[a]["floor"] is not None]
            floor = floors_seen[0] if floors_seen else 0
        hull = geo.convex_hull(pts)
        rooms.append(
            ScanRoom(
                id=rid,
                floor=int(floor),
                classification=str(info.get("classification") or "Unknown"),
                centroid=(
                    round(sum(p[0] for p in pts) / len(pts), COORD_DP),
                    round(sum(p[1] for p in pts) / len(pts), COORD_DP),
                ),
                z_mean=round(sum(zs) / len(zs) - datum_z, COORD_DP),
                area_m2=info.get("area_m2"),
                anchor_count=len(ids),
                hull=tuple((round(x, COORD_DP), round(y, COORD_DP)) for x, y in hull),
                hull_area_m2=round(geo.polygon_area(hull), 2) if len(hull) >= 3 else 0.0,
                footprint=_square_footprint(
                    (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)),
                    info.get("area_m2"),
                ),
            )
        )
    room_by_id = {r.id: r for r in rooms}

    # collapse the anchor graph into room adjacency
    pair_counts: dict[tuple[str, str], int] = {}
    for aid, anchor in anchors.items():
        room_a = anchor["room"]
        if not room_a:
            continue
        for bid in anchor["neighbors"]:
            other = anchors.get(bid)
            if not other or not other["room"] or other["room"] == room_a:
                continue
            key = tuple(sorted((room_a, other["room"])))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    links: list[ScanLink] = []
    for (a, b) in sorted(pair_counts):
        ra, rb = room_by_id.get(a), room_by_id.get(b)
        if ra is None or rb is None:
            continue
        links.append(
            ScanLink(
                a=a,
                b=b,
                # Each anchor edge is seen once from each end, so halve; "or 1" keeps a one-sided neighbour link.
                anchor_edges=pair_counts[(a, b)] // 2 or 1,
                gap_m=round(math.dist(ra.centroid, rb.centroid), COORD_DP),
                floors=abs(ra.floor - rb.floor),
            )
        )

    classified = sum(1 for r in rooms if r.classification != "Unknown")
    with_area = sum(1 for r in rooms if r.area_m2 is not None)
    stats = {
        "floors": len(elevations),
        "rooms": len(rooms),
        "rooms_with_classification": classified,
        "rooms_with_area": with_area,
        "anchors": len(anchors),
        "room_links": len(links),
        "cross_floor_links": sum(1 for l in links if l.floors),
        "anchor_edges": sum(l.anchor_edges for l in links),
        "hull_vs_area_ratio": _hull_ratio(rooms),
    }

    notes = [
        "scan-cluster build: rooms are clusters of scan positions, not measured "
        "outlines - there is no wall geometry in this model",
        f"connectivity is measured from the anchor graph ({stats['anchor_edges']} "
        f"anchor edges collapsed into {len(links)} room links), not inferred",
    ]
    for elevation in sorted(elevations, key=lambda e: e.floor):
        if elevation.source == slabmod.SOURCE_PANO:
            notes.append(
                f"floor {elevation.floor} elevation from scan height minus "
                f"{slabmod.PANO_ABOVE_FLOOR_M} m ({elevation.sample_size} scans)"
            )
        elif elevation.source == slabmod.SOURCE_DOORWAY:
            notes.append(
                f"floor {elevation.floor} elevation from {elevation.sample_size} door sills"
            )
    if not classified:
        notes.append(
            "no room classifications: room type priors have nothing to work from here"
        )
    if with_area < len(rooms):
        notes.append(f"{len(rooms) - with_area} room(s) have no floor area")
    ratio = stats["hull_vs_area_ratio"]
    notes.append(
        "convex hulls of the scan points are NOT usable footprints"
        + (f" (median hull/area {ratio})" if ratio is not None else "")
        + " - use the synthetic square footprint, which carries the reported area"
    )
    if rooms:
        biggest = max(rooms, key=lambda r: r.area_m2 or 0.0)
        # Tuned heuristic: one zone over 35 m2 (several rooms' worth) suggests the floor was not split into rooms.
        if (biggest.area_m2 or 0.0) > 35.0:
            notes.append(
                f"coarse segmentation: the largest zone is {biggest.area_m2} m2 across "
                f"{biggest.anchor_count} scans - without the structural layer a whole "
                "floor can be one 'room', so search granularity is per zone, not per room"
            )

    return ScanModel(
        schema=SCHEMA_VERSION,
        model_id=mid,
        name=str(payload.get("name") or ""),
        units="metres",
        yaw_deg=round(yaw, 4),
        level="scan",
        slabs=tuple(sorted(elevations, key=lambda e: e.floor)),
        floor_heights=heights,
        rooms=tuple(rooms),
        links=tuple(links),
        stats=stats,
        notes=tuple(notes),
    )


def _square_footprint(
    centre: tuple[float, float], area_m2: Optional[float]
) -> tuple[tuple[float, float], ...]:
    """An axis-aligned square of ``area_m2`` centred on ``centre``.

    Synthetic, and labelled as such. It exists because consumers like the phone
    likelihood need *a* polygon with the right area and position; a square of
    the correct area beats a convex hull that captures an eighth of the room.
    """
    if not area_m2 or area_m2 <= 0:
        return ()
    half = math.sqrt(area_m2) / 2.0
    cx, cy = centre
    return tuple(
        (round(cx + dx * half, COORD_DP), round(cy + dy * half, COORD_DP))
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    )


def _hull_ratio(rooms) -> Optional[float]:
    """Median hull area / reported area - how much the hull under-estimates."""
    ratios = [
        r.hull_area_m2 / r.area_m2
        for r in rooms
        if r.area_m2 and r.area_m2 > 0 and r.hull_area_m2 > 0
    ]
    if not ratios:
        return None
    ratios.sort()
    return round(ratios[len(ratios) // 2], 3)


def build_scan_model(
    model_id: str, client: Optional[MatterportClient] = None
) -> ScanModel:
    """Fetch and assemble a scan-cluster model. One network call."""
    client = client or MatterportClient()
    data = client.query(SCAN_QUERY, {"id": model_id})
    model = data.get("model")
    if not model:
        from ingest.mp3d.client import MatterportError

        raise MatterportError(f"no model returned for id {model_id!r}")
    return scan_model_from_payload(model, model_id)

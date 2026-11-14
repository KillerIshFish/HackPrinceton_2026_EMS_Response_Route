"""Wall extraction and decomposition around openings.

Two jobs:

* **Dedupe.** A wall between two rooms appears on both rooms' boundaries under
  the same ``Edge.id``. Collapsing on that id gives one wall per physical wall
  (156 instead of 312 on the reference model) and, as a by-product, tells you
  which walls are exterior (referenced once) and which are interior (twice).
* **Decompose.** A wall carrying openings is cut into the solid panels that
  remain: full-height spans between openings, plus the pieces below a sill and
  above a head. That is what makes a window render as a real hole at its
  surveyed height rather than a painted-on rectangle.

An edge whose ``type`` is ``invisible`` is an open boundary - a room edge where
no wall exists, such as the mouth of an open-plan kitchen. Those are kept in the
model (they still bound the room) but flagged so a renderer can skip them
instead of sealing the space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

MIN_WALL_LEN_M = 0.05
MIN_PANEL_SPAN = 0.004      # fraction of wall length
MIN_PANEL_HEIGHT_M = 0.03
DEFAULT_THICKNESS_M = 0.12
DP = 4

OPENING_KINDS = ("doorway", "window", "opening", "divider")
TRAVERSABLE = frozenset({"doorway", "opening", "divider"})


@dataclass(frozen=True)
class Panel:
    """A solid rectangle of wall: ``t`` along the wall, ``z`` in elevation."""

    t0: float
    t1: float
    z0: float
    z1: float


def wall_length(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def opening_span(
    opening: dict, length_m: float, datum_z: float
) -> Optional[tuple[str, float, float, float, float]]:
    """``(kind, t0, t1, z0, z1)`` for one opening, or ``None`` if under-specified.

    ``relativeCenter`` is a fraction along the wall and ``width`` is metres, so
    the half-width has to be converted into the same fractional space. Measured
    across 70 models, every opening that exists carries all four fields, so a
    ``None`` here means genuinely absent data rather than a common case.
    """
    kind = opening.get("type")
    centre = opening.get("relativeCenter")
    width = opening.get("width")
    lower = opening.get("lowerElevation")
    height = opening.get("height")
    if kind not in OPENING_KINDS:
        return None
    if None in (centre, width, lower, height) or length_m <= 0:
        return None
    half = min(float(width), length_m * 0.98) / 2.0 / length_m
    t0 = max(0.0, float(centre) - half)
    t1 = min(1.0, float(centre) + half)
    if t1 <= t0:
        return None
    z0 = float(lower) - datum_z
    z1 = z0 + float(height)
    return (kind, round(t0, DP), round(t1, DP), round(z0, DP), round(z1, DP))


def solid_panels(
    openings: Sequence[tuple[float, float, float, float]],
    slab_z: float,
    height_m: float,
) -> list[Panel]:
    """Panels of a wall that remain solid once its openings are cut out.

    ``openings`` are ``(t0, t1, z0, z1)`` with z on the model datum; ``slab_z``
    and ``height_m`` bound the wall. Openings are walked in order along the wall
    and clamped so overlapping ones cannot produce negative-width panels.
    """
    top = slab_z + height_m
    panels: list[Panel] = []
    cursor = 0.0
    for t0_raw, t1_raw, z0_raw, z1_raw in sorted(openings):
        t0 = max(cursor, min(1.0, t0_raw))
        t1 = max(t0, min(1.0, t1_raw))
        if t0 > cursor + MIN_PANEL_SPAN:
            panels.append(Panel(round(cursor, DP), round(t0, DP), slab_z, top))
        z0 = max(slab_z, min(top, z0_raw))
        z1 = max(slab_z, min(top, z1_raw))
        if z0 > slab_z + MIN_PANEL_HEIGHT_M:
            panels.append(Panel(round(t0, DP), round(t1, DP), slab_z, round(z0, DP)))
        if z1 < top - MIN_PANEL_HEIGHT_M:
            panels.append(Panel(round(t0, DP), round(t1, DP), round(z1, DP), top))
        cursor = max(cursor, t1)
    if cursor < 1.0 - MIN_PANEL_SPAN:
        panels.append(Panel(round(cursor, DP), 1.0, slab_z, top))
    return panels


def dedupe_walls(floors: Sequence[dict], datum_z: float) -> list[dict]:
    """One record per physical wall, sorted by Matterport edge id.

    Pure. Returns plain dicts so :mod:`ingest.mp3d.model3d` owns the dataclass
    shape; sorting by id is what makes the output deterministic.
    """
    collected: dict[str, dict] = {}
    for floor in sorted(floors, key=lambda f: f.get("sequence") or 0):
        sequence = int(floor.get("sequence") or 0)
        for room in sorted(floor.get("rooms") or [], key=lambda r: r.get("id") or ""):
            room_id = room.get("id") or ""
            for edge in ((room.get("boundary") or {}).get("edges") or []):
                edge_id = edge.get("id")
                vertices = edge.get("vertices") or []
                if not edge_id or len(vertices) != 2:
                    continue
                pa = vertices[0].get("position") or {}
                pb = vertices[1].get("position") or {}
                if None in (pa.get("x"), pa.get("y"), pb.get("x"), pb.get("y")):
                    continue
                a = (float(pa["x"]), float(pa["y"]))
                b = (float(pb["x"]), float(pb["y"]))
                length = wall_length(a, b)
                if length < MIN_WALL_LEN_M:
                    continue

                existing = collected.get(edge_id)
                if existing is not None:
                    existing["room_ids"].append(room_id)
                    existing["exterior"] = False
                    continue

                spans = []
                for opening in edge.get("openings") or []:
                    span = opening_span(opening, length, datum_z)
                    if span is not None:
                        spans.append(span)
                spans.sort(key=lambda s: (s[1], s[3], s[0]))

                collected[edge_id] = {
                    "id": edge_id,
                    "floor": sequence,
                    "a": (round(a[0], DP), round(a[1], DP)),
                    "b": (round(b[0], DP), round(b[1], DP)),
                    "length_m": round(length, DP),
                    "thickness_m": round(
                        float(edge.get("thickness") or DEFAULT_THICKNESS_M), DP
                    ),
                    "thickness_measured": edge.get("thickness") is not None,
                    "open_boundary": edge.get("type") == "invisible",
                    "exterior": True,
                    "room_ids": [room_id],
                    "openings": spans,
                }

    walls = [collected[k] for k in sorted(collected)]
    for wall in walls:
        wall["room_ids"] = tuple(sorted(set(wall["room_ids"])))
    return walls


def iter_opening_kinds(walls: Sequence[dict]) -> Iterator[str]:
    for wall in walls:
        for span in wall["openings"]:
            yield span[0]

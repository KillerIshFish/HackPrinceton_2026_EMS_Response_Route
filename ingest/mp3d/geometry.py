"""Pure planar geometry. No API calls, no globals, no randomness.

Coordinates are Matterport model-space metres, ``(x, y)`` with y pointing the
same way as the model's own y axis. Elevations are handled in
:mod:`ingest.mp3d.slabs`.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

Point = tuple[float, float]

COORD_DP = 4  # decimal places kept in emitted coordinates (determinism)
EPS = 1e-9


def ring_from_edges(edges: Sequence[dict]) -> Optional[list[Point]]:
    """Chain boundary edges into a single ordered closed ring.

    Matterport room boundaries are simple polygons: every vertex is shared by
    exactly two edges. Anything else (a dangling edge, a figure-of-eight, fewer
    than three vertices) returns ``None`` rather than a repaired guess.
    """
    neighbours: dict[str, list[str]] = {}
    position: dict[str, Point] = {}
    for edge in edges:
        vertices = edge.get("vertices") or []
        if len(vertices) != 2:
            return None
        ids = []
        for vertex in vertices:
            pos = vertex.get("position") or {}
            if pos.get("x") is None or pos.get("y") is None:
                return None
            vid = vertex.get("id")
            if vid is None:
                return None
            position[vid] = (float(pos["x"]), float(pos["y"]))
            ids.append(vid)
        neighbours.setdefault(ids[0], []).append(ids[1])
        neighbours.setdefault(ids[1], []).append(ids[0])

    if len(neighbours) < 3:
        return None
    if any(len(v) != 2 for v in neighbours.values()):
        return None

    start = min(neighbours)  # deterministic entry point
    order = [start]
    previous, current = None, start
    while True:
        nxt = [v for v in neighbours[current] if v != previous]
        if not nxt:
            return None
        step = nxt[0]
        if step == start:
            break
        order.append(step)
        previous, current = current, step
        if len(order) > len(neighbours):
            return None
    if len(order) != len(neighbours):
        return None  # more than one disjoint loop
    return [position[v] for v in order]


def signed_area(polygon: Sequence[Point]) -> float:
    """Shoelace signed area; positive when the ring is counter-clockwise."""
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_area(polygon: Sequence[Point]) -> float:
    """Absolute area in square metres.

    This is the authoritative room area for the pipeline. Matterport's own
    ``dimensions.areaFloor`` is imperial and disagrees with the ring by up to
    78% on open-plan rooms, so it is deliberately not used.
    """
    return abs(signed_area(polygon))


def as_ccw(polygon: Sequence[Point]) -> list[Point]:
    """Return the ring wound counter-clockwise."""
    pts = list(polygon)
    return pts if signed_area(pts) >= 0 else pts[::-1]


def centroid(polygon: Sequence[Point]) -> Point:
    """Area centroid, falling back to the vertex mean for degenerate rings."""
    area = signed_area(polygon)
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    if abs(area) < EPS:
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n,
        )
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * area), cy / (6.0 * area))


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting containment test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    neg = d1 < -EPS or d2 < -EPS or d3 < -EPS
    pos = d1 > EPS or d2 > EPS or d3 > EPS
    return not (neg and pos)


def triangulate(polygon: Sequence[Point]) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon.

    Returns index triples into the counter-clockwise winding of ``polygon``.
    Used for floor plates, which are frequently concave (L-shaped rooms), so a
    centroid fan would produce triangles outside the room.
    """
    pts = as_ccw(polygon)
    n = len(pts)
    if n < 3:
        return []
    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard <= n * n:
        guard += 1
        clipped = False
        count = len(remaining)
        for i in range(count):
            ia = remaining[(i - 1) % count]
            ib = remaining[i]
            ic = remaining[(i + 1) % count]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if _cross(a, b, c) <= EPS:
                continue  # reflex or collinear vertex, not an ear
            blocked = False
            for j in remaining:
                if j in (ia, ib, ic):
                    continue
                if _in_triangle(pts[j], a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            triangles.append((ia, ib, ic))
            remaining.pop(i)
            clipped = True
            break
        if not clipped:
            break  # degenerate ring; emit what we have
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def convex_hull(points: Sequence[Point]) -> list[Point]:
    """Counter-clockwise convex hull (Andrew's monotone chain).

    Used by the scan-cluster build, where a room has no measured outline and the
    hull of its scan positions is the best available footprint. It is always an
    under-estimate: a camera never stands against the walls.
    """
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) <= 2:
        return list(pts)

    def half(sequence: Sequence[Point]) -> list[Point]:
        out: list[Point] = []
        for point in sequence:
            while len(out) >= 2 and _cross(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return lower[:-1] + upper[:-1]


def yaw_degrees(rotation: Optional[Mapping[str, float]] = None) -> float:
    """Model-to-world yaw from Matterport's ``geocoordinates.rotation``.

    The quaternion only ever carries a z component (rotation about the vertical
    axis), so the yaw reduces to ``2 * atan2(z, w)``.
    """
    if not rotation:
        return 0.0
    z = float(rotation.get("z") or 0.0)
    w = float(rotation.get("w") or 1.0)
    return math.degrees(2.0 * math.atan2(z, w))


def rotate_to_world(x: float, y: float, yaw_deg: float) -> Point:
    """Rotate a model-space vector into world axes, returning ``(east, north)``."""
    theta = math.radians(yaw_deg)
    return (
        x * math.cos(theta) - y * math.sin(theta),
        x * math.sin(theta) + y * math.cos(theta),
    )


def outward_bearing(
    polygon: Sequence[Point], a: Point, b: Point, yaw_deg: float
) -> float:
    """Compass bearing of a wall segment's outward normal, degrees east of north.

    The normal is chosen to point away from the polygon's centroid, then rotated
    into world axes by the model yaw.
    """
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    cen = centroid(polygon)
    ex, ey = b[0] - a[0], b[1] - a[1]
    nx, ny = -ey, ex
    out = (mid[0] + nx - cen[0]) ** 2 + (mid[1] + ny - cen[1]) ** 2
    inn = (mid[0] - nx - cen[0]) ** 2 + (mid[1] - ny - cen[1]) ** 2
    if out <= inn:
        nx, ny = ey, -ex
    length = math.hypot(nx, ny) or 1.0
    east, north = rotate_to_world(nx / length, ny / length, yaw_deg)
    return math.degrees(math.atan2(east, north)) % 360.0


def circular_mean(bearings: Iterable[float]) -> Optional[float]:
    """Mean of compass bearings, or ``None`` for an empty input."""
    sx = sy = 0.0
    count = 0
    for bearing in bearings:
        sx += math.sin(math.radians(bearing))
        sy += math.cos(math.radians(bearing))
        count += 1
    if not count:
        return None
    return math.degrees(math.atan2(sx, sy)) % 360.0


def angular_gap(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def round_point(point: Point, dp: int = COORD_DP) -> Point:
    return (round(point[0], dp), round(point[1], dp))

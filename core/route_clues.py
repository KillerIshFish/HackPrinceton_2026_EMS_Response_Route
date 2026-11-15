"""Route clues: "I turned right after coming up the stairs".

A route clue names an *anchor* the caller passed (a stair flight they came up or
down, an elevator, a building entrance) and the *turn* they made after it
(left, right, straight). For every room this module walks the shortest
same-floor route from each place that anchor arrives to the room, lists the
significant turns on the way, and scores how likely a caller in that room is to
describe the route that way.

Geometry used (all building-local metres, x east, y north):

* stairs edge (``Building.edges``, optional keys written by ``ingest.scan_building``):
  ``top_xy`` + ``heading`` = where the flight arrives going up and the walker's
  direction of travel there; ``bottom_xy`` + ``bottom_heading`` = the same going
  down. Without them the walker starts at the centroid of the arrival room and
  the first leg of the route sets the heading (right for enclosed stairwells).
* door / corridor edge: optional ``xy`` = the opening. Without it the opening is
  taken as the middle of the wall the two room outlines share.
* entrance: ``xy``; the walker heads inward, perpendicular to the nearest wall.

Scoring. The route is a polyline: start point, each opening passed, the target
room's centroid. Legs shorter than ``MIN_LEG_M`` are merged; a change of
direction of at least ``TURN_MIN_DEG`` is a turn. The clue ties the turn to the
anchor, so a caller nearly always means the first turn after it, occasionally a
later one, occasionally misreports:

    P(first turn) = 0.8, P(some later turn) = 0.1 shared, P(straight) = 0.1
    no turn at all:  P(straight) = 0.8, P(left) = P(right) = 0.1

and ``m = P(reported turn) / 0.9`` capped at 1. With several arrival points on
a floor (two stairs) the scores are averaged, nearer arrivals weighing more.
Rooms on a floor the anchor never arrives at score 0 (coming *up* the stairs
rules out the lowest floor).
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

from core.schema import Building, Room

# Tuned heuristics (no measured basis). Legs under 1 m are door jogs, not walked legs; 35 deg ignores
# the small kinks of traced corridors but still counts a diagonal turn as left / right.
MIN_LEG_M = 1.0
TURN_MIN_DEG = 35.0
ARRIVAL_DISTANCE_SCALE_M = 15.0  # arrival weight decays as exp(-walk / this): 15 m farther weighs ~0.37x
VERTICAL_KINDS = ("stairs", "elevator")

# Report model of the module docstring (judgement, not measured): a caller usually means the first turn
# after the anchor. Lower P_FIRST_TURN softens the clue; P_MAX normalises the best case to m = 1.
P_FIRST_TURN = 0.8
P_LATER_TURNS = 0.1
P_STRAIGHT_DESPITE_TURN = 0.1
P_STRAIGHT_NO_TURN = 0.8
P_MAX = P_FIRST_TURN + P_LATER_TURNS

XY = tuple[float, float]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _unit(dx: float, dy: float) -> Optional[XY]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def _signed_angle_deg(a: XY, b: XY) -> float:
    """Angle from direction ``a`` to direction ``b``; positive = counter-clockwise = a left turn."""
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def _outline_samples(polygon: list, step: float = 0.25) -> list[XY]:
    """Points every ``step`` metres along the outline (0.25 m: finer than a door width, cheap enough)."""
    pts: list[XY] = []
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        k = max(1, int(math.hypot(x2 - x1, y2 - y1) / step))
        pts.extend((x1 + (x2 - x1) * j / k, y1 + (y2 - y1) * j / k) for j in range(k))
    return pts


def _dist_to_segment(p: XY, a: XY, b: XY) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _dist_to_outline(p: XY, polygon: list) -> float:
    n = len(polygon)
    return min(_dist_to_segment(p, polygon[i], polygon[(i + 1) % n]) for i in range(n))


def opening_point(u: Room, v: Room, edge: dict) -> XY:
    """Where a walker passes from room ``u`` to room ``v``."""
    xy = edge.get("xy")
    if xy:
        return float(xy[0]), float(xy[1])
    samples = _outline_samples(u.polygon)
    if samples and len(v.polygon) >= 2:
        scored = [(_dist_to_outline(p, v.polygon), p) for p in samples]
        nearest = min(d for d, _ in scored)
        # Samples within 0.35 m of the closest count as the shared wall (tolerance for traced outlines).
        shared = [p for d, p in scored if d <= nearest + 0.35]
        return sum(p[0] for p in shared) / len(shared), sum(p[1] for p in shared) / len(shared)
    return (u.centroid[0] + v.centroid[0]) / 2, (u.centroid[1] + v.centroid[1]) / 2


def _inward_heading(room: Room, xy: XY) -> Optional[XY]:
    """Direction a walker faces on stepping through a door at ``xy`` into ``room``."""
    poly = room.polygon
    if len(poly) < 3:
        return _unit(room.centroid[0] - xy[0], room.centroid[1] - xy[1])
    n = len(poly)
    i = min(range(n), key=lambda k: _dist_to_segment(xy, poly[k], poly[(k + 1) % n]))
    (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
    normal = _unit(-(y2 - y1), x2 - x1)
    if normal is None:
        return _unit(room.centroid[0] - xy[0], room.centroid[1] - xy[1])
    to_centre = (room.centroid[0] - xy[0], room.centroid[1] - xy[1])
    return normal if normal[0] * to_centre[0] + normal[1] * to_centre[1] >= 0 else (-normal[0], -normal[1])


# ---------------------------------------------------------------------------
# arrivals: where an anchor puts the walker, and which way they are heading
# ---------------------------------------------------------------------------


def arrivals(building: Building, anchor: str) -> list[tuple[str, XY, Optional[XY]]]:
    """``[(room_id, start_xy, heading or None)]`` for ``stairs_up`` / ``stairs_down`` / ``elevator`` / ``entrance``."""
    rooms = {r.id: r for r in building.rooms}
    out: list[tuple[str, XY, Optional[XY]]] = []
    if anchor in ("stairs_up", "stairs_down"):
        up = anchor == "stairs_up"
        for e in building.edges:
            a, b = rooms.get(e.get("u")), rooms.get(e.get("v"))
            if e.get("kind") != "stairs" or a is None or b is None or a.floor == b.floor:
                continue
            lo, hi = (a, b) if a.floor < b.floor else (b, a)
            room = hi if up else lo
            xy = e.get("top_xy" if up else "bottom_xy")
            h = e.get("heading" if up else "bottom_heading")
            if xy:
                out.append((room.id, (float(xy[0]), float(xy[1])), _unit(*h) if h else None))
            else:
                out.append((room.id, (room.centroid[0], room.centroid[1]), None))
    elif anchor == "elevator":
        out = [(r.id, (r.centroid[0], r.centroid[1]), None) for r in building.rooms if r.room_type == "elevator"]
    elif anchor == "entrance":
        for ent in building.entrances:
            room = rooms.get(ent.get("room_id"))
            if room is None or not ent.get("xy"):
                continue
            xy = (float(ent["xy"][0]), float(ent["xy"][1]))
            out.append((room.id, xy, _inward_heading(room, xy)))
    # the same landing reached from several edges (a stair shaft) is one arrival
    seen: set[tuple] = set()
    unique = []
    for rid, xy, h in out:
        key = (rid, round(xy[0], 1), round(xy[1], 1))
        if key not in seen:
            seen.add(key)
            unique.append((rid, xy, h))
    return unique


# ---------------------------------------------------------------------------
# routes and turns
# ---------------------------------------------------------------------------


def _floor_routes(building: Building, source: str) -> dict[str, tuple[float, list[int]]]:
    """Shortest same-floor routes from ``source``: ``{room_id: (length_m, [edge index, ...])}``."""
    rooms = {r.id: r for r in building.rooms}
    floor = rooms[source].floor
    adj: dict[str, list[tuple[str, float, int]]] = {}
    for i, e in enumerate(building.edges):
        u, v = e.get("u"), e.get("v")
        if e.get("kind") in VERTICAL_KINDS or u not in rooms or v not in rooms:
            continue
        if rooms[u].floor != floor or rooms[v].floor != floor:
            continue
        w = float(e.get("length_m", 1.0))  # edge with no length: one unit, so hop count decides
        adj.setdefault(u, []).append((v, w, i))
        adj.setdefault(v, []).append((u, w, i))
    best: dict[str, tuple[float, list[int]]] = {source: (0.0, [])}
    heap: list[tuple[float, str]] = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > best[node][0]:
            continue
        for nxt, w, i in adj.get(node, []):
            nd = d + w
            if nxt not in best or nd < best[nxt][0]:
                best[nxt] = (nd, best[node][1] + [i])
                heapq.heappush(heap, (nd, nxt))
    return best


def turns_along(points: list[XY], heading: Optional[XY]) -> tuple[list[str], float]:
    """Significant turns (``"left"`` / ``"right"``) along a polyline and its walked length.

    ``heading`` is the walker's direction at ``points[0]``; when None the first leg sets it.
    """
    path = [points[0]]
    for p in points[1:-1]:
        if math.dist(p, path[-1]) >= MIN_LEG_M:
            path.append(p)
    # the last step into the room counts only if it is a real leg (not a landing's own centroid)
    if len(points) > 1 and math.dist(points[-1], path[-1]) >= MIN_LEG_M:
        path.append(points[-1])
    turns: list[str] = []
    length = 0.0
    current = heading
    for a, b in zip(path, path[1:]):
        leg = _unit(b[0] - a[0], b[1] - a[1])
        length += math.dist(a, b)
        if leg is None:
            continue
        if current is not None:
            angle = _signed_angle_deg(current, leg)
            if abs(angle) >= TURN_MIN_DEG:
                turns.append("left" if angle > 0 else "right")
        current = leg
    return turns, length


def report_probability(turns: list[str], reported: str) -> float:
    """P(caller describes the route with ``reported``) given the turns actually on it."""
    if not turns:
        return P_STRAIGHT_NO_TURN if reported == "straight" else (1.0 - P_STRAIGHT_NO_TURN) / 2
    if reported == "straight":
        return P_STRAIGHT_DESPITE_TURN
    later = turns[1:]
    p = P_FIRST_TURN if turns[0] == reported else 0.0
    if later:
        p += P_LATER_TURNS * sum(1 for t in later if t == reported) / len(later)
    elif turns[0] == reported:
        p += P_LATER_TURNS
    return p


def route_match(building: Building, anchor: str, turn: str) -> Optional[dict[str, float]]:
    """m(r) in [0, 1] for "after <anchor> I went <turn>", or None when the building has no such anchor."""
    starts = arrivals(building, anchor)
    if not starts:
        return None
    rooms = {r.id: r for r in building.rooms}
    arrival_floors = {rooms[rid].floor for rid, _, _ in starts}
    weighted: dict[str, list[tuple[float, float]]] = {}
    opening_cache: dict[tuple[int, str], XY] = {}

    def opening(i: int, frm: str) -> XY:
        if (i, frm) not in opening_cache:
            e = building.edges[i]
            other = e["v"] if e["u"] == frm else e["u"]
            opening_cache[(i, frm)] = opening_point(rooms[frm], rooms[other], e)
        return opening_cache[(i, frm)]

    for source, start, heading in starts:
        for rid, (_, edge_path) in _floor_routes(building, source).items():
            points = [start]
            node = source
            for i in edge_path:
                points.append(opening(i, node))
                e = building.edges[i]
                node = e["v"] if e["u"] == node else e["u"]
            target = rooms[rid]
            points.append((target.centroid[0], target.centroid[1]))
            turns, length = turns_along(points, heading)
            p = report_probability(turns, turn)
            weighted.setdefault(rid, []).append((math.exp(-length / ARRIVAL_DISTANCE_SCALE_M), min(1.0, p / P_MAX)))

    out: dict[str, float] = {}
    for room in building.rooms:
        scores = weighted.get(room.id)
        if scores:
            total = sum(w for w, _ in scores)
            out[room.id] = sum(w * m for w, m in scores) / total if total > 0 else 0.5  # 0.5: no information
        else:
            # on an arrival floor but cut off from every arrival: no information; other floors: ruled out
            out[room.id] = 0.5 if room.floor in arrival_floors else 0.0
    return out

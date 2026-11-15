"""Building layout map for the reasoning interpreter (parser/agent.py).

The model places a caller from whatever they say about their surroundings: the hallway they
are on, how many doors from the stairs, the end of the hall, a corner, a big or a tiny room,
the elevator they can hear. None of that is a room number or a name, so the model needs the
building's layout spelled out. This module derives it from the room polygons and the walking
graph, once per building:

* **Hallways.** Every corridor segment with its length, the axis it runs along and the rooms
  that open off it *in order from one end to the other*, each with the side of the hall it is
  on, and what sits at either end (a stairwell, another corridor, a room at a dead end).
* **Per room.** Floor area and how it ranks by size on its floor; where it sits on the floor
  plate (corner, outer edge, interior); which hallway(s) it opens onto, on which side and at
  which door position; walking metres to the nearest stairwell, elevator and (on a floor with
  one) entrance; windows; the rooms sharing a wall and across the hall; suite and tenant.

Directions such as "plan W end" are the floor plan's own axes (+x = plan east, +y = plan
north). They are reliable for relations inside one floor (which end of a hall, which side)
but are not necessarily true compass directions; ``windows face ...`` is the real outlook.

Pure functions over ``parser.building_tools.BuildingIndex``; no network.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from core.schema import Room

CIRCULATION = {"corridor", "stairwell", "elevator"}
WALL_GAP_M = 0.6  # two outlines this close share a wall / a door
ACROSS_MAX_M = 16.0
EDGE_BAND_M = 2.5  # a room this close to the floor plate's outline is on the outer edge
TINY_M2 = 6.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _point_segment_m(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def outline_gap_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Smallest distance between two room outlines (0 when they touch)."""
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            b1, b2 = b[j], b[(j + 1) % len(b)]
            best = min(best, _point_segment_m(a1, b1, b2), _point_segment_m(a2, b1, b2),
                       _point_segment_m(b1, a1, a2), _point_segment_m(b2, a1, a2))
            if best <= 1e-9:
                return 0.0
    return best


def _bbox(room: Room) -> tuple[float, float, float, float]:
    if not room.polygon:
        x, y = room.centroid[0], room.centroid[1]
        return x, y, x, y
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# Layout records
# ---------------------------------------------------------------------------


@dataclass
class HallSlot:
    """One room's place along one hallway."""

    hall_id: str
    side: str  # "plan N side", "plan W end", ...
    door: int  # 1-based position from the hall's start end, counted on that side (0 = at an end)
    of: int  # rooms on that side


@dataclass
class RoomLayout:
    id: str
    size_rank: int = 0  # 1 = largest non-circulation room on its floor (0 = circulation)
    size_of: int = 0
    size_word: str = ""
    position: str = ""
    halls: list[HallSlot] = field(default_factory=list)
    stairs_m: Optional[float] = None
    elevator_m: Optional[float] = None
    entrance_m: Optional[float] = None
    next_to: list[str] = field(default_factory=list)
    across: list[str] = field(default_factory=list)


@dataclass
class HallLayout:
    id: str
    name: str
    floor_label: str
    length_m: float
    axis: str  # "plan W-E" or "plan S-N"
    start_end: str  # "plan W end"
    finish_end: str  # "plan E end"
    at_start: list[str] = field(default_factory=list)
    at_finish: list[str] = field(default_factory=list)
    order: list[tuple[str, str]] = field(default_factory=list)  # (room id, "N"/"S"/"E"/"W") from the start end
    connects: list[str] = field(default_factory=list)  # other circulation it touches


@dataclass
class FloorLayout:
    label: str
    floor: int
    width_m: float
    depth_m: float
    rooms: list[str]
    stairs: list[str]
    elevators: list[str]
    entrances: list[str]
    halls: list[HallLayout]


@dataclass
class BuildingLayout:
    floors: list[FloorLayout]
    rooms: dict[str, RoomLayout]
    halls: dict[str, HallLayout]


# ---------------------------------------------------------------------------
# Building the layout
# ---------------------------------------------------------------------------


def _touching(rooms: list[Room]) -> dict[str, set[str]]:
    boxes = {r.id: _bbox(r) for r in rooms}
    order = sorted(rooms, key=lambda r: boxes[r.id][0])
    out: dict[str, set[str]] = defaultdict(set)
    for i, a in enumerate(order):
        ba = boxes[a.id]
        for b in order[i + 1:]:
            bb = boxes[b.id]
            if bb[0] - ba[2] > WALL_GAP_M:
                break  # sorted by min x: nothing further right can touch a
            if ba[1] - bb[3] > WALL_GAP_M or bb[1] - ba[3] > WALL_GAP_M or ba[0] - bb[2] > WALL_GAP_M:
                continue
            if outline_gap_m(list(a.polygon), list(b.polygon)) <= WALL_GAP_M:
                out[a.id].add(b.id)
                out[b.id].add(a.id)
    return out


def _floor_distances(graph: nx.Graph, floor: int, sources: list[str]) -> dict[str, float]:
    if not sources:
        return {}
    nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "room" and d.get("floor") == floor]
    sub = graph.subgraph(nodes)
    present = [s for s in sources if s in sub]
    if not present:
        return {}
    return dict(nx.multi_source_dijkstra_path_length(sub, present, weight="length_m"))


def _straight_m(a: Room, b: Room) -> float:
    return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])


def _size_word(rank: int, of: int, area: float) -> str:
    if area < TINY_M2:
        base = "tiny"
    elif of <= 2:
        base = "large" if rank == 1 else "small"
    else:
        frac = (rank - 1) / max(1, of - 1)
        base = "large" if frac < 1 / 3 else "medium" if frac < 2 / 3 else "small"
    if rank == 1 and of > 1:
        return f"{base}, largest on floor"
    if rank == of and of > 2:
        return f"{base}, smallest on floor"
    return f"{base}, {_ordinal(rank)} largest of {of}"


def _position(box: tuple[float, float, float, float], plate: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = plate
    w, d = x1 - x0, y1 - y0
    band_x = max(EDGE_BAND_M, 0.06 * w)
    band_y = max(EDGE_BAND_M, 0.06 * d)
    west, east = box[0] - x0 <= band_x, x1 - box[2] <= band_x
    south, north = box[1] - y0 <= band_y, y1 - box[3] <= band_y
    ns = "N" if north and not south else "S" if south and not north else ""
    ew = "E" if east and not west else "W" if west and not east else ""
    if ns and ew:
        return f"plan {ns}{ew} corner"
    if (north and south) or (east and west):
        return "spans the floor plate"
    if ns or ew:
        return f"outer edge, plan {ns or ew} side"
    return "interior (no outside wall)"


def _hall_layout(hall: Room, touching: dict[str, set[str]], by_id: dict[str, Room]) -> HallLayout:
    x0, y0, x1, y1 = _bbox(hall)
    along_x = (x1 - x0) >= (y1 - y0)
    lo, hi = (x0, x1) if along_x else (y0, y1)
    mid_cross = (y0 + y1) / 2 if along_x else (x0 + x1) / 2
    start_end, finish_end = ("plan W end", "plan E end") if along_x else ("plan S end", "plan N end")
    h = HallLayout(id=hall.id, name=hall.name, floor_label=hall.floor_label, length_m=round(hi - lo, 1),
                   axis="plan W-E" if along_x else "plan S-N", start_end=start_end, finish_end=finish_end)
    slack = 0.25 * max(1.0, min(x1 - x0, y1 - y0))
    sided: list[tuple[float, str, str]] = []
    for oid in sorted(touching.get(hall.id, ())):
        o = by_id[oid]
        ox, oy = o.centroid[0], o.centroid[1]
        a, c = (ox, oy) if along_x else (oy, ox)
        if o.room_type == "corridor" or o.room_type == "elevator":
            h.connects.append(oid)
        if a < lo + slack and (o.room_type != "corridor" or a < lo):
            h.at_start.append(oid)
            continue
        if a > hi - slack and (o.room_type != "corridor" or a > hi):
            h.at_finish.append(oid)
            continue
        if o.room_type == "corridor":
            continue  # a crossing hallway part-way along: listed under connects
        side = ("N" if c > mid_cross else "S") if along_x else ("E" if c > mid_cross else "W")
        sided.append((a, side, oid))
    sided.sort()
    h.order = [(oid, side) for _, side, oid in sided]
    h.connects = sorted(set(h.connects))
    return h


def build_layout(idx) -> BuildingLayout:
    """The layout of ``idx.building`` (a ``parser.building_tools.BuildingIndex``)."""
    b = idx.building
    graph = idx._graph
    rooms_by_floor: dict[int, list[Room]] = defaultdict(list)
    for r in b.rooms:
        rooms_by_floor[r.floor].append(r)
    entrance_rooms = defaultdict(list)
    for e in b.entrances or []:
        rid = e.get("room_id")
        if rid in idx.rooms:
            entrance_rooms[idx.rooms[rid].floor].append(rid)

    layouts: dict[str, RoomLayout] = {}
    halls: dict[str, HallLayout] = {}
    floors: list[FloorLayout] = []
    for floor in sorted(rooms_by_floor):
        rooms = rooms_by_floor[floor]
        by_id = {r.id: r for r in rooms}
        boxes = {r.id: _bbox(r) for r in rooms}
        plate = (min(bx[0] for bx in boxes.values()), min(bx[1] for bx in boxes.values()),
                 max(bx[2] for bx in boxes.values()), max(bx[3] for bx in boxes.values()))
        touching = _touching(rooms)
        stairs = [r.id for r in rooms if r.room_type == "stairwell"]
        elevators = [r.id for r in rooms if r.room_type == "elevator"]
        ents = sorted(set(entrance_rooms.get(floor, [])))
        d_stairs = _floor_distances(graph, floor, stairs)
        d_elev = _floor_distances(graph, floor, elevators)
        d_ent = _floor_distances(graph, floor, ents)

        ranked = sorted((r for r in rooms if r.room_type not in CIRCULATION), key=lambda r: (-r.area_m2, r.id))
        rank_of = {r.id: i + 1 for i, r in enumerate(ranked)}

        floor_halls = [_hall_layout(r, touching, by_id) for r in rooms if r.room_type == "corridor"]
        for h in floor_halls:
            halls[h.id] = h

        for r in rooms:
            lay = RoomLayout(id=r.id)
            if r.id in rank_of:
                lay.size_rank, lay.size_of = rank_of[r.id], len(ranked)
                lay.size_word = _size_word(lay.size_rank, lay.size_of, r.area_m2)
            lay.position = _position(boxes[r.id], plate)

            def dist(table: dict[str, float], targets: list[str]) -> Optional[float]:
                if r.id in table:
                    return round(table[r.id])
                if targets:  # not connected on the walking graph: straight-line, marked approximate
                    return -round(min(_straight_m(r, by_id[t]) for t in targets))
                return None

            lay.stairs_m = dist(d_stairs, stairs)
            lay.elevator_m = dist(d_elev, elevators)
            lay.entrance_m = dist(d_ent, ents)

            def centre_m(o: str) -> float:
                return round(_straight_m(r, by_id[o]), 3)

            lay.next_to = sorted((o for o in touching.get(r.id, ()) if by_id[o].room_type not in CIRCULATION),
                                 key=lambda o: (centre_m(o), o))[:8 if r.room_type in CIRCULATION else 5]
            if r.room_type not in CIRCULATION:
                seen = set(touching.get(r.id, ())) | {r.id}
                cands: set[str] = set()
                for c in touching.get(r.id, ()):
                    if by_id[c].room_type == "corridor":
                        cands |= {o for o in touching.get(c, ()) if o not in seen
                                  and by_id[o].room_type not in CIRCULATION}
                lay.across = sorted((o for o in cands if centre_m(o) <= ACROSS_MAX_M), key=lambda o: (centre_m(o), o))[:3]
            layouts[r.id] = lay

        for h in floor_halls:
            counts = defaultdict(int)
            for _, side in h.order:
                counts[side] += 1
            seen_side: dict[str, int] = defaultdict(int)
            for oid, side in h.order:
                seen_side[side] += 1
                layouts[oid].halls.append(HallSlot(h.id, f"plan {side} side", seen_side[side], counts[side]))
            for oid in h.at_start:
                if by_id[oid].room_type != "corridor":
                    layouts[oid].halls.append(HallSlot(h.id, h.start_end, 0, 0))
            for oid in h.at_finish:
                if by_id[oid].room_type != "corridor":
                    layouts[oid].halls.append(HallSlot(h.id, h.finish_end, 0, 0))

        floors.append(FloorLayout(
            label=rooms[0].floor_label, floor=floor,
            width_m=round(plate[2] - plate[0]), depth_m=round(plate[3] - plate[1]),
            rooms=[r.id for r in rooms], stairs=stairs, elevators=elevators, entrances=ents,
            halls=floor_halls,
        ))
    return BuildingLayout(floors=floors, rooms=layouts, halls=halls)


# ---------------------------------------------------------------------------
# Text for the cached system block and for tool results
# ---------------------------------------------------------------------------


def _metres(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"~{-v:.0f} m (straight line)" if v < 0 else f"{v:.0f} m"


def hall_slot_text(slot: HallSlot, short: bool = False) -> str:
    if slot.door == 0:
        return f"{slot.hall_id} at its {slot.side}"
    start = "W" if "N side" in slot.side or "S side" in slot.side else "S"
    return f"{slot.hall_id} {slot.side}, door {slot.door} of {slot.of} from its plan {start} end"


def room_layout_text(idx, layout: BuildingLayout, room: Room, *, compact: bool = False) -> str:
    """One room: ``id | name | type, area (size) | position | halls | distances | windows | neighbours | suite``."""
    lay = layout.rooms.get(room.id) or RoomLayout(id=room.id)
    size = f"{room.room_type}, {round(room.area_m2)} m2" + (f" ({lay.size_word})" if lay.size_word else "")
    parts = [room.id, room.name, size, lay.position]
    if lay.halls:
        parts.append("opens onto " + "; ".join(hall_slot_text(s) for s in lay.halls[:3]))
    elif room.room_type not in CIRCULATION:
        parts.append("no hallway touches it (reached through a neighbouring room)")
    dists = []
    if lay.stairs_m is not None and room.room_type != "stairwell":
        dists.append(f"stairs {_metres(lay.stairs_m)}")
    if lay.elevator_m is not None and room.room_type != "elevator":
        dists.append(f"elevator {_metres(lay.elevator_m)}")
    if lay.entrance_m is not None:
        dists.append(f"entrance {_metres(lay.entrance_m)}")
    if dists:
        parts.append("walk to " + ", ".join(dists))
    facts: list[str] = []
    if room.window_facing:
        facts.append(f"windows face {room.window_facing}")
    elif room.has_windows:
        facts.append("has windows")
    else:
        facts.append("no windows")
    if not compact:
        if lay.next_to:
            facts.append("next to: " + ", ".join(lay.next_to))
        if lay.across:
            facts.append("across the hall: " + ", ".join(lay.across))
    suite, tenant = idx.suite(room.id), idx.tenant(room.id)
    if suite:
        facts.append(f"suite {suite}")
    if tenant:
        facts.append(f"tenant {tenant}")
    elif (idx.meta_rooms.get(room.id) or {}).get("status") == "available":
        facts.append("vacant")
    parts.append("; ".join(facts))
    return " | ".join(p for p in parts if p)


def hall_text(h: HallLayout, idx) -> str:
    def label(rid: str) -> str:
        r = idx.rooms.get(rid)
        if r is None:
            return rid
        if r.room_type in ("stairwell", "elevator", "corridor"):
            return f"{rid} [{r.room_type}]"
        return rid

    head = f"- {h.id} \"{h.name}\", {h.length_m:.0f} m, runs {h.axis}."
    body = [f"{h.start_end}: {', '.join(label(x) for x in h.at_start) or 'dead end / nothing listed'}"]
    if h.order:
        body.append("then along the hall: " + ", ".join(f"{label(rid)} ({side})" for rid, side in h.order))
    body.append(f"{h.finish_end}: {', '.join(label(x) for x in h.at_finish) or 'dead end / nothing listed'}")
    mids = [c for c in h.connects if c not in h.at_start and c not in h.at_finish]
    if mids:
        body.append("crossed part-way by " + ", ".join(label(c) for c in mids))
    return head + " " + "; ".join(body)

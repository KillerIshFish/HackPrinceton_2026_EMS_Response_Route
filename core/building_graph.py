"""Building connectivity graph and responder travel times.

Nodes are room ids plus entrance ids; every edge carries ``time_s``:

    time_s = length_m / walk_mps  (+ stair seconds for every floor climbed on stairs)

The rates come from a ``core.schema.TravelProfile`` (``situation.travel.profile_for``). With
no profile they are today's constants: 1.2 m/s walking and 15 s per floor of stairs.

Elevator rides are in the graph whenever the travel profile offers an elevator (medical
incident types, no fire or smoke; ``situation.travel.elevator_allowed_for``), so a ride is
taken when it is the faster way up. With no profile, or a profile that offers no elevator,
elevator edges are left out (ground rule 4). ``planning.approach.recommend_approach`` still
shows the stair and elevator times side by side.

``infer_elevator_edges`` adds elevator links a building's data implies but does not draw: an
"elevator lobby" stacked over the floors with no elevator room serving it.

Gates: ``closed_room_ids`` / ``closed_entrance_ids`` leave those nodes out. Only pass gates
the officer in charge set (ground rule 2).
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

import networkx as nx

from core.schema import Building, Room, TravelProfile

WALK_SPEED_MPS = 1.2
STAIR_PENALTY_S_PER_FLOOR = 15.0
ELEVATOR_S_PER_FLOOR = 10.0  # only used when use_elevators=True and no profile is given
DEFAULT_FLOOR_HEIGHT_M = 4.0


def _edge_length_m(building: Building, edge: dict, centroids: dict[str, tuple[float, float, float]]) -> float:
    """Edge walking distance: ``length_m`` if given, else centroid-to-centroid distance."""
    length = edge.get("length_m")
    if length is not None:
        return float(length)
    (ux, uy, uz), (vx, vy, vz) = centroids[edge["u"]], centroids[edge["v"]]
    return math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2 + (uz - vz) ** 2)


def walk_mps(profile: Optional[TravelProfile] = None) -> float:
    """Walking speed of ``profile`` (today's 1.2 m/s without one)."""
    return profile.walk_mps if profile is not None else WALK_SPEED_MPS


def stair_rate_s(profile: Optional[TravelProfile], floor: Optional[int] = None) -> float:
    """Stair seconds to climb into ``floor``: the high-rise rate above ``high_floor_from``,
    the base rate otherwise (and when ``floor`` is unknown)."""
    if profile is None:
        return STAIR_PENALTY_S_PER_FLOOR
    if floor is not None and profile.high_floor_from is not None and floor > profile.high_floor_from:
        return profile.stair_s_per_floor_high
    return profile.stair_s_per_floor


def stair_climb_s(profile: Optional[TravelProfile], floors_entered: Iterable[int]) -> float:
    """Stair component of a climb that enters each floor in ``floors_entered`` once."""
    return float(sum(stair_rate_s(profile, f) for f in floors_entered))


def elevator_ride_s(profile: Optional[TravelProfile], floors: int) -> Optional[float]:
    """One elevator ride over ``floors`` floors; None when the profile offers no elevator.

    Profile: ``elevator_fixed_s + elevator_s_per_floor * floors``. No profile: today's
    10 s per floor.
    """
    if profile is None:
        return ELEVATOR_S_PER_FLOOR * max(1, abs(floors))
    if not profile.elevator_allowed or profile.elevator_fixed_s is None:
        return None
    return profile.elevator_fixed_s + (profile.elevator_s_per_floor or 0.0) * abs(floors)


def edge_time_s(
    kind: str,
    length_m: float,
    floors: int = 0,
    profile: Optional[TravelProfile] = None,
    from_floor: Optional[int] = None,
) -> float:
    """Traversal time of one edge.

    Walk ``length_m`` at the profile's walking speed; stairs add the stair rate for every
    floor climbed (``from_floor`` is the lower end, used for the high-rise rate; without it
    the base rate applies); an elevator edge is one ride (``elevator_ride_s``, ``inf`` when
    the profile offers no elevator). Without a profile: 1.2 m/s, +15 s per floor on stairs,
    10 s per floor by elevator (today's numbers).
    """
    time_s = length_m / walk_mps(profile)
    n = abs(int(floors))
    if kind == "stairs":
        if from_floor is None:
            time_s += stair_rate_s(profile) * n
        else:
            time_s += stair_climb_s(profile, range(from_floor + 1, from_floor + n + 1))
    elif kind == "elevator":
        ride = elevator_ride_s(profile, n)
        time_s = math.inf if ride is None else ride
    return time_s


def build_graph(
    building: Building,
    use_elevators: Optional[bool] = None,
    profile: Optional[TravelProfile] = None,
    closed_room_ids: Iterable[str] = (),
    closed_entrance_ids: Iterable[str] = (),
) -> nx.Graph:
    """Undirected graph of rooms + entrances with ``time_s`` edge weights (seconds).

    * ``profile``        - movement rates (None = today's constants).
    * ``use_elevators``  - add elevator rides. Default (None): whenever ``profile`` offers an
      elevator, so routes take a ride when it is faster; never without a profile. Always
      ignored when ``profile`` offers no elevator (ground rule 4). With a profile, every pair
      of floors one shaft serves becomes a single ride, so the fixed per-ride time is paid once.
    * ``closed_room_ids`` / ``closed_entrance_ids`` - officer gates; those nodes (and an
      entrance opening into a closed room) are left out.
    """
    closed_rooms = set(closed_room_ids)
    closed_ents = set(closed_entrance_ids)
    if use_elevators is None:
        use_elevators = profile is not None and profile.elevator_allowed
    elevators_on = use_elevators and (profile is None or profile.elevator_allowed)

    G = nx.Graph()
    centroids = {r.id: r.centroid for r in building.rooms if r.id not in closed_rooms}
    for room in building.rooms:
        if room.id in closed_rooms:
            continue
        G.add_node(
            room.id,
            kind="room",
            floor=room.floor,
            floor_label=room.floor_label,
            room_type=room.room_type,
            xy=(room.centroid[0], room.centroid[1]),
        )

    def add(u: str, v: str, kind: str, length_m: float, floors: int, time_s: float) -> None:
        # Keep the fastest connection if two rooms are linked more than once.
        if G.has_edge(u, v) and G[u][v]["time_s"] <= time_s:
            return
        G.add_edge(u, v, kind=kind, length_m=length_m, floors=floors, time_s=time_s)

    shaft = nx.Graph()  # elevator edges, grouped into shafts for whole rides
    for edge in building.edges:
        u, v = edge["u"], edge["v"]
        if u not in centroids or v not in centroids:
            continue
        kind = edge.get("kind", "door")
        if kind == "elevator" and not elevators_on:
            continue
        floors = int(edge.get("floors", 0) or 0)
        if kind in ("stairs", "elevator") and floors == 0:
            floors = abs(G.nodes[u]["floor"] - G.nodes[v]["floor"])
        length_m = _edge_length_m(building, edge, centroids)
        if kind == "elevator" and profile is not None:
            shaft.add_edge(u, v, length_m=length_m)
            continue
        lower = min(G.nodes[u]["floor"], G.nodes[v]["floor"])
        from_floor = lower if profile is not None else None
        add(u, v, kind, length_m, floors, edge_time_s(kind, length_m, floors, profile, from_floor))

    # One ride between every pair of stops in a shaft (the fixed time is paid once per ride).
    for comp in nx.connected_components(shaft):
        stops = sorted(comp)
        for i, u in enumerate(stops):
            for v in stops[i + 1:]:
                floors = abs(G.nodes[u]["floor"] - G.nodes[v]["floor"])
                if floors == 0:
                    continue
                length_m = shaft[u][v]["length_m"] if shaft.has_edge(u, v) else 0.0
                add(u, v, "elevator", length_m, floors, edge_time_s("elevator", length_m, floors, profile))

    speed = walk_mps(profile)
    for ent in building.entrances:
        room_id = ent.get("room_id")
        if room_id not in centroids or ent.get("id") in closed_ents:
            continue
        xy = ent.get("xy")
        if ent.get("length_m") is not None:
            length_m = float(ent["length_m"])
        elif xy is not None:
            cx, cy, _ = centroids[room_id]
            length_m = math.hypot(float(xy[0]) - cx, float(xy[1]) - cy)
        else:
            length_m = 3.0
        G.add_node(
            ent["id"],
            kind="entrance",
            floor=G.nodes[room_id]["floor"],
            floor_label=G.nodes[room_id]["floor_label"],
            room_type="entrance",
            xy=tuple(xy) if xy is not None else G.nodes[room_id]["xy"],
        )
        G.add_edge(
            ent["id"], room_id, kind="entrance", length_m=length_m, floors=0,
            time_s=length_m / speed,
        )
    return G


def travel_times_from(G: nx.Graph, source: str) -> dict[str, float]:
    """Shortest travel time (s) from ``source`` to every reachable node (Dijkstra on time_s)."""
    return dict(nx.single_source_dijkstra_path_length(G, source, weight="time_s"))


def all_pairs_travel_times(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Shortest travel time (s) between every pair of connected nodes."""
    return {u: dict(d) for u, d in nx.all_pairs_dijkstra_path_length(G, weight="time_s")}


def shortest_path(G: nx.Graph, source: str, target: str) -> list[str]:
    """Node sequence of the fastest route from ``source`` to ``target``."""
    return nx.dijkstra_path(G, source, target, weight="time_s")


# ---------------------------------------------------------------------------
# Elevators implied by elevator lobbies
# ---------------------------------------------------------------------------

_ELEVATOR_LOBBY = re.compile(
    r"\b(?:elevators?|lifts?)\s+(?:lobby|lobbies|hall|landing|vestibule|foyer)\b", re.IGNORECASE
)
LOBBY_STACK_RADIUS_M = 8.0  # two lobbies on adjacent floors this close in plan are one shaft's stops
SERVED_RADIUS_M = 10.0  # a drawn elevator stop this close to a lobby already serves it


def _plan_dist(a: Room, b: Room) -> float:
    return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])


def _inside(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Even-odd point-in-polygon test."""
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _stacked(a: Room, b: Room) -> bool:
    """Two rooms on different floors sit over one another: centroids close, or one inside the other."""
    return (
        _plan_dist(a, b) <= LOBBY_STACK_RADIUS_M
        or (len(a.polygon) >= 3 and _inside(b.centroid[0], b.centroid[1], a.polygon))
        or (len(b.polygon) >= 3 and _inside(a.centroid[0], a.centroid[1], b.polygon))
    )


def infer_elevator_edges(building: Building) -> list[dict]:
    """Elevator links implied by elevator lobbies that the data draws no elevator for.

    A room named "elevator lobby" (or lift lobby / elevator hall / landing / vestibule), and
    not itself typed ``elevator``, means a car stops there. Such lobbies stacked on adjacent
    floors (``_stacked``) are stops of one shaft. The shaft also stops at a main lobby
    (``room_type == "lobby"``) directly above or below an elevator lobby, because a ground floor
    usually names its lobby without the word elevator. Each pair of adjacent stops gets one
    ``elevator`` edge between the two lobbies, marked ``"inferred": True``. No edge is added
    where a drawn shaft already stops beside both lobbies (a node of an elevator edge, or an
    elevator room, within ``SERVED_RADIUS_M`` of each, in the same shaft).
    """
    rooms = {r.id: r for r in building.rooms}
    lobby_ids = {
        r.id for r in building.rooms if r.room_type != "elevator" and _ELEVATOR_LOBBY.search(r.name or "")
    }
    if not lobby_ids:
        return []
    stops = [r for r in building.rooms if r.id in lobby_ids or r.room_type == "lobby"]
    levels = sorted(set(building.floors) | {r.floor for r in building.rooms})
    next_up = dict(zip(levels, levels[1:]))

    # Drawn shafts: elevator rooms, joined by elevator edges.
    drawn = nx.Graph()
    drawn.add_nodes_from(r.id for r in building.rooms if r.room_type == "elevator")
    for e in building.edges:
        if e.get("kind") == "elevator" and e["u"] in rooms and e["v"] in rooms:
            drawn.add_edge(e["u"], e["v"])
    shaft_of = {n: i for i, comp in enumerate(nx.connected_components(drawn)) for n in comp}

    def shafts_beside(room: Room) -> set[int]:
        return {
            s for n, s in shaft_of.items()
            if rooms[n].floor == room.floor and _plan_dist(rooms[n], room) <= SERVED_RADIUS_M
        }

    heights = building.floor_heights_m
    out: list[dict] = []
    for low in sorted(stops, key=lambda r: (r.floor, r.id)):
        up = next_up.get(low.floor)
        if up is None:
            continue
        # one end at least must be an elevator lobby (two main lobbies say nothing about an elevator)
        above = [
            r for r in stops
            if r.floor == up and (low.id in lobby_ids or r.id in lobby_ids) and _stacked(low, r)
        ]
        if not above:
            continue
        high = min(above, key=lambda r: (_plan_dist(low, r), r.id))
        if shafts_beside(low) & shafts_beside(high):
            continue
        if low.floor in heights and high.floor in heights:
            rise = abs(heights[high.floor] - heights[low.floor])
        else:
            rise = abs(high.centroid[2] - low.centroid[2]) or DEFAULT_FLOOR_HEIGHT_M
        out.append({
            "u": low.id, "v": high.id, "kind": "elevator",
            "length_m": round(rise, 2), "floors": abs(high.floor - low.floor),
            "inferred": True,
        })
    return out


def with_inferred_elevators(building: Building) -> Building:
    """``building`` with ``infer_elevator_edges`` appended to its edges (the same object when none)."""
    extra = infer_elevator_edges(building)
    if not extra:
        return building
    return building.model_copy(update={"edges": [*building.edges, *extra]})

"""Building connectivity graph and responder travel times.

Nodes are room ids plus entrance ids; every edge carries ``time_s``:

    time_s = length_m / WALK_SPEED_MPS  (+ STAIR_PENALTY_S_PER_FLOOR * floors on stairs)

Elevator edges are left out by default (responders take the stairs).
"""

from __future__ import annotations

import math

import networkx as nx

from core.schema import Building

WALK_SPEED_MPS = 1.2
STAIR_PENALTY_S_PER_FLOOR = 15.0
ELEVATOR_S_PER_FLOOR = 10.0  # only used when use_elevators=True
DEFAULT_FLOOR_HEIGHT_M = 4.0


def _edge_length_m(building: Building, edge: dict, centroids: dict[str, tuple[float, float, float]]) -> float:
    """Edge walking distance: ``length_m`` if given, else centroid-to-centroid distance."""
    length = edge.get("length_m")
    if length is not None:
        return float(length)
    (ux, uy, uz), (vx, vy, vz) = centroids[edge["u"]], centroids[edge["v"]]
    return math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2 + (uz - vz) ** 2)


def edge_time_s(kind: str, length_m: float, floors: int = 0) -> float:
    """Traversal time: walk at 1.2 m/s, +15 s per floor on stairs, 10 s per floor by elevator."""
    time_s = length_m / WALK_SPEED_MPS
    if kind == "stairs":
        time_s += STAIR_PENALTY_S_PER_FLOOR * abs(floors)
    elif kind == "elevator":
        time_s = ELEVATOR_S_PER_FLOOR * max(1, abs(floors))
    return time_s


def build_graph(building: Building, use_elevators: bool = False) -> nx.Graph:
    """Undirected graph of rooms + entrances with ``time_s`` edge weights (seconds)."""
    G = nx.Graph()
    centroids = {r.id: r.centroid for r in building.rooms}
    for room in building.rooms:
        G.add_node(
            room.id,
            kind="room",
            floor=room.floor,
            floor_label=room.floor_label,
            room_type=room.room_type,
            xy=(room.centroid[0], room.centroid[1]),
        )

    for edge in building.edges:
        u, v = edge["u"], edge["v"]
        if u not in centroids or v not in centroids:
            continue
        kind = edge.get("kind", "door")
        if kind == "elevator" and not use_elevators:
            continue
        floors = int(edge.get("floors", 0) or 0)
        if kind in ("stairs", "elevator") and floors == 0:
            floors = abs(G.nodes[u]["floor"] - G.nodes[v]["floor"])
        length_m = _edge_length_m(building, edge, centroids)
        time_s = edge_time_s(kind, length_m, floors)
        # Keep the fastest connection if two rooms are linked more than once.
        if G.has_edge(u, v) and G[u][v]["time_s"] <= time_s:
            continue
        G.add_edge(u, v, kind=kind, length_m=length_m, floors=floors, time_s=time_s)

    for ent in building.entrances:
        room_id = ent.get("room_id")
        if room_id not in centroids:
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
            time_s=length_m / WALK_SPEED_MPS,
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

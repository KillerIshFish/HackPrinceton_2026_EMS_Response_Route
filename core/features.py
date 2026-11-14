"""Per-room feature dictionaries used by the prior, clue likelihoods and the UI."""

from __future__ import annotations

import math

from core.schema import Building, Room

PHONE_HEIGHT_ABOVE_FLOOR_M = 1.2  # a held / pocketed phone sits ~1.2 m above the slab
CIRCULATION_TYPES = frozenset({"corridor", "stairwell", "elevator", "lobby"})


def expected_phone_z_m(building: Building, room: Room) -> float:
    """Expected barometric height of a phone in ``room``: floor elevation + 1.2 m."""
    return building.floor_heights_m[room.floor] + PHONE_HEIGHT_ABOVE_FLOOR_M


def adjacency(building: Building) -> dict[str, set[str]]:
    """Room id -> set of directly connected room ids (from ``Building.edges``)."""
    adj: dict[str, set[str]] = {r.id: set() for r in building.rooms}
    for edge in building.edges:
        u, v = edge.get("u"), edge.get("v")
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def room_features(building: Building) -> dict[str, dict]:
    """Feature dict per room id.

    Contract fields (room_type, floor, floor_label, area_m2, has_windows,
    window_facing, near_elevator, near_stairs, occupancy_factor) plus derived
    ones: expected_phone_z_m, is_circulation, is_top_floor, is_ground_floor,
    n_neighbors, neighbor_types.
    """
    adj = adjacency(building)
    by_id = {r.id: r for r in building.rooms}
    top_floor = max(building.floors) if building.floors else None
    ground_floor = (
        min(building.floors, key=lambda f: (abs(building.floor_heights_m.get(f, 0.0)), f))
        if building.floors
        else None
    )
    feats: dict[str, dict] = {}
    for room in building.rooms:
        neighbors = sorted(adj[room.id])
        feats[room.id] = {
            "room_type": room.room_type,
            "floor": room.floor,
            "floor_label": room.floor_label,
            "area_m2": room.area_m2,
            "has_windows": room.has_windows,
            "window_facing": room.window_facing,
            "near_elevator": room.near_elevator,
            "near_stairs": room.near_stairs,
            "occupancy_factor": room.occupancy_factor,
            "expected_phone_z_m": expected_phone_z_m(building, room),
            "is_circulation": room.room_type in CIRCULATION_TYPES,
            "is_top_floor": room.floor == top_floor,
            "is_ground_floor": room.floor == ground_floor,
            "n_neighbors": len(neighbors),
            "neighbor_types": sorted({by_id[n].room_type for n in neighbors}),
        }
    return feats


def rooms_near_type(building: Building, room_type: str, radius_m: float = 8.0) -> set[str]:
    """Ids of rooms whose centroid is within ``radius_m`` (same floor) of a ``room_type`` room.

    Utility for ingest code that needs to derive ``near_elevator`` / ``near_stairs``.
    """
    anchors = [r for r in building.rooms if r.room_type == room_type]
    near: set[str] = set()
    for room in building.rooms:
        for a in anchors:
            if a.floor != room.floor:
                continue
            if math.hypot(room.centroid[0] - a.centroid[0], room.centroid[1] - a.centroid[1]) <= radius_m:
                near.add(room.id)
                break
    return near

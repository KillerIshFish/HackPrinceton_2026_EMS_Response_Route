"""Approach recommendation: which entrance, where to stage, which stairwell.

The entrance is the one minimising the posterior-expected travel time to the
caller, ``sum_r p_r * travel(entrance -> r) / sum_r p_r``. The staging spot is the
one tied to that entrance, and the stairwell is the first one on the shortest
route from that entrance to the single most probable room.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

import networkx as nx

from core.building_graph import build_graph, travel_times_from
from core.schema import Building

UNREACHABLE_S = 3600.0


def _latlng(d: Mapping) -> tuple[float, float]:
    ll = d.get("latlng") or (0.0, 0.0)
    return (float(ll[0]), float(ll[1]))


def staging_for_entrance(building: Building, entrance_id: str) -> Optional[dict]:
    """Staging spot linked to ``entrance_id`` (fallback: nearest by xy, then first)."""
    spots = building.staging_spots
    if not spots:
        return None
    for s in spots:
        if s.get("near_entrance_id") == entrance_id:
            return s
    ent = next((e for e in building.entrances if e.get("id") == entrance_id), None)
    if ent is not None and ent.get("xy") is not None:
        ex, ey = ent["xy"][0], ent["xy"][1]
        with_xy = [s for s in spots if s.get("xy") is not None]
        if with_xy:
            return min(with_xy, key=lambda s: math.hypot(s["xy"][0] - ex, s["xy"][1] - ey))
    return spots[0]


def expected_travel_by_entrance(
    building: Building, room_probs: Mapping[str, float], G: nx.Graph
) -> dict[str, float]:
    """entrance id -> posterior-expected travel seconds to the caller's room."""
    total_p = sum(p for p in room_probs.values() if p > 0.0)
    out: dict[str, float] = {}
    for ent in building.entrances:
        eid = ent["id"]
        if eid not in G:
            out[eid] = UNREACHABLE_S
            continue
        times = travel_times_from(G, eid)
        if total_p <= 0.0:
            out[eid] = 0.0
            continue
        acc = sum(p * times.get(rid, UNREACHABLE_S) for rid, p in room_probs.items() if p > 0.0)
        out[eid] = acc / total_p
    return out


def recommend_approach(
    building: Building, room_probs: Mapping[str, float], G: nx.Graph | None = None
) -> dict:
    """Recommend entrance, staging spot and stairwell for the current posterior.

    Returns the dict documented in the interface spec. ``stairwell_id`` /
    ``stairwell_name`` are ``None`` when the target room is on the entrance floor
    (or no stairwell lies on the route).
    """
    if G is None:
        G = build_graph(building)
    rooms = {r.id: r for r in building.rooms}

    known = {rid: p for rid, p in room_probs.items() if rid in rooms}
    target_id = max(known, key=lambda rid: (known[rid], rid)) if known else None
    target = rooms.get(target_id) if target_id else None

    exp_travel = expected_travel_by_entrance(building, known, G)

    result = {
        "entrance_id": "",
        "entrance_name": "",
        "entrance_latlng": (0.0, 0.0),
        "staging_id": "",
        "staging_name": "",
        "staging_latlng": (0.0, 0.0),
        "stairwell_id": None,
        "stairwell_name": None,
        "target_room_id": target_id or "",
        "target_floor_label": target.floor_label if target else "",
        "entrance_expected_travel_s": {k: round(v, 1) for k, v in exp_travel.items()},
    }
    if not building.entrances:
        return result

    ent_by_id = {e["id"]: e for e in building.entrances}
    best_id = min(exp_travel, key=lambda eid: (exp_travel[eid], eid))
    ent = ent_by_id[best_id]
    result["entrance_id"] = best_id
    result["entrance_name"] = ent.get("name", best_id)
    result["entrance_latlng"] = _latlng(ent)

    spot = staging_for_entrance(building, best_id)
    if spot is not None:
        result["staging_id"] = spot.get("id", "")
        result["staging_name"] = spot.get("name", spot.get("id", ""))
        result["staging_latlng"] = _latlng(spot)

    # Stairwell: first stairwell room on the fastest route entrance -> top room.
    if target is not None and best_id in G and target_id in G:
        ent_room = rooms.get(ent.get("room_id", ""))
        same_floor = ent_room is not None and ent_room.floor == target.floor
        if not same_floor:
            try:
                path = nx.shortest_path(G, best_id, target_id, weight="time_s")
            except nx.NetworkXNoPath:
                path = []
            for node in path:
                room = rooms.get(node)
                if room is not None and room.room_type == "stairwell":
                    result["stairwell_id"] = room.id
                    result["stairwell_name"] = room.name
                    break
            else:
                # Open stairs (a flight inside a room, no stairwell room): name the room it starts in.
                for u, v in zip(path, path[1:]):
                    if G[u][v].get("kind") == "stairs" and u in rooms and v in rooms:
                        lower = min((rooms[u], rooms[v]), key=lambda r: r.floor)
                        result["stairwell_id"] = lower.id
                        result["stairwell_name"] = f"Stairs in {lower.name}"
                        break
    return result

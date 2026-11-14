"""Approach recommendation: which entrance, where to stage, which stairwell.

The entrance is the one minimising the posterior-expected travel time to the
caller, ``sum_r p_r * travel(entrance -> r) / sum_r p_r``. The staging spot is the
one tied to that entrance, and the stairwell is the first one on the shortest
route from that entrance to the single most probable room.

Step B1 adds, without changing that choice:

* ``stairs_s_to_target_floor`` / ``elevator_s_to_target_floor`` - the vertical leg from the
  entrance floor to the target floor by stairs and by elevator, side by side. The elevator
  value is None when the travel profile offers no elevator (fire, smoke, non-medical), when
  no elevator shaft serves both floors, or when the target is on the entrance floor.
  Nothing here picks one for the crew.
* ``entrance_access_s`` - walk from each entrance's staging spot to the entrance at
  clear-air walking speed. No seconds are added for locks, codes, desks or gates (no
  per-barrier time is published); ``entrance_warnings`` says what is recorded or missing.
* ``meet_person_credit_s`` / ``predicted_curb_to_target_s`` - curb (staging spot) to the
  most likely room through the recommended entrance, less 24 s once when someone will meet
  the crew (Silverman 2007).
* ``warnings`` - whole-approach warnings, e.g. a prediction faster than measured
  curb-to-patient intervals.

Officer gates (``gates`` / the situation's closed lists) are never recommended or routed
through.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import networkx as nx

from core.building_graph import build_graph, elevator_ride_s, stair_climb_s, travel_times_from
from core.schema import Building, Gates, Situation, TravelProfile
from situation.config import section, setting_value
from situation.travel import profile_for, travel_settings

UNREACHABLE_S = 3600.0

NO_KEY_HOLDER = "No key holder recorded for this entrance"
CLOSED_BY_OFFICER = "Closed by the officer in charge"
FASTER_THAN_MEASURED = (
    "Predicted time is faster than measured curb-to-patient intervals for this kind of building"
)
_OPEN_ACCESS = ("open", "public", "unlocked")


def _latlng(d: Mapping) -> tuple[float, float]:
    ll = d.get("latlng") or (0.0, 0.0)
    return (float(ll[0]), float(ll[1]))


def _travel_setting(*path: str, default: float) -> float:
    node = section("travel")
    for p in path:
        node = node.get(p) if isinstance(node, dict) else None
    return float(setting_value(node, default))


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


def _distance_m(a: Mapping, b: Mapping) -> Optional[float]:
    """Ground distance between two spots: local xy if both have it, else lat/lng, else None."""
    if a.get("xy") is not None and b.get("xy") is not None:
        return math.hypot(float(a["xy"][0]) - float(b["xy"][0]), float(a["xy"][1]) - float(b["xy"][1]))
    if a.get("latlng") and b.get("latlng"):
        (la, lo), (lb, lob) = _latlng(a), _latlng(b)
        k = 111_320.0
        dx = (lob - lo) * k * math.cos(math.radians((la + lb) / 2.0))
        dy = (lb - la) * k
        return math.hypot(dx, dy)
    return None


def entrance_warnings(ent: Mapping) -> list[str]:
    """What is recorded (or missing) about getting through this entrance. Never adds time.

    Reads optional entrance fields: ``access`` (e.g. "open", "locked", "code", "staffed desk",
    "gate"), ``barriers`` (list of strings), ``key_holder``, ``access_code``. An entrance with
    no record is not assumed open.
    """
    access = str(ent.get("access") or "").strip().lower()
    barriers = [str(b) for b in (ent.get("barriers") or []) if str(b).strip()]
    if access and access not in _OPEN_ACCESS and access not in (b.lower() for b in barriers):
        barriers.insert(0, access)
    out = [f"Barrier recorded: {b}. No time added (no per-barrier time is published)" for b in barriers]
    if access not in _OPEN_ACCESS and not ent.get("key_holder"):
        out.append(NO_KEY_HOLDER)
    if any("code" in b.lower() for b in barriers) and not ent.get("access_code"):
        out.append("Door code needed; no code recorded")
    return out


def floors_entered(building: Building, from_floor: int, to_floor: int) -> list[int]:
    """Floors entered going from ``from_floor`` to ``to_floor`` (by number, either direction)."""
    lo, hi = sorted((int(from_floor), int(to_floor)))
    levels = sorted(set(building.floors) | {lo, hi})
    return [f for f in levels if lo < f <= hi]


def elevator_serves(building: Building, floor_a: int, floor_b: int, closed_room_ids: Iterable[str] = ()) -> bool:
    """True when one elevator shaft (connected elevator edges, open rooms) stops on both floors."""
    closed = set(closed_room_ids)
    floor_of = {r.id: r.floor for r in building.rooms if r.id not in closed}
    shafts = nx.Graph()
    for e in building.edges:
        if e.get("kind") == "elevator" and e["u"] in floor_of and e["v"] in floor_of:
            shafts.add_edge(e["u"], e["v"])
    for comp in nx.connected_components(shafts):
        floors = {floor_of[n] for n in comp}
        if floor_a in floors and floor_b in floors:
            return True
    return False


def expected_travel_by_entrance(
    building: Building,
    room_probs: Mapping[str, float],
    G: nx.Graph,
    closed_entrance_ids: Iterable[str] = (),
) -> dict[str, float]:
    """entrance id -> posterior-expected travel seconds to the caller's room (closed entrances left out)."""
    closed = set(closed_entrance_ids)
    total_p = sum(p for p in room_probs.values() if p > 0.0)
    out: dict[str, float] = {}
    for ent in building.entrances:
        eid = ent["id"]
        if eid in closed:
            continue
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
    building: Building,
    room_probs: Mapping[str, float],
    G: nx.Graph | None = None,
    *,
    profile: TravelProfile | None = None,
    situation: Situation | None = None,
    gates: Gates | None = None,
) -> dict:
    """Recommend entrance, staging spot and stairwell for the current posterior.

    * ``profile``   - travel rates; default ``profile_for(situation)`` when a situation is
      given, else today's constants.
    * ``situation`` - supplies ``meet_person`` (24 s credit), ``load`` and closed lists.
    * ``gates``     - officer-closed rooms/entrances (merged with the situation's lists).
    * ``G``         - a prebuilt graph; it must already reflect ``profile`` and the gates.
      Closed entrances and rooms are still kept out of the choice when it does not.

    Returns the dict documented in the interface spec plus the B1 keys described in the
    module docstring. ``stairwell_id`` / ``stairwell_name`` are ``None`` when the target room
    is on the entrance floor (or no stairwell lies on the route).
    """
    if profile is None and situation is not None:
        profile = profile_for(situation)
    prof = profile if profile is not None else TravelProfile()
    closed_rooms = set(gates.closed_room_ids if gates else ()) | set(situation.closed_room_ids if situation else ())
    closed_ents = set(gates.closed_entrance_ids if gates else ()) | set(
        situation.closed_entrance_ids if situation else ()
    )
    if G is None:
        G = build_graph(building, profile=profile, closed_room_ids=closed_rooms, closed_entrance_ids=closed_ents)
    rooms = {r.id: r for r in building.rooms}

    known = {rid: p for rid, p in room_probs.items() if rid in rooms and rid not in closed_rooms}
    target_id = max(known, key=lambda rid: (known[rid], rid)) if known else None
    target = rooms.get(target_id) if target_id else None

    exp_travel = expected_travel_by_entrance(building, known, G, closed_ents)

    # Access and barrier warnings for every entrance (outdoors: clear-air walking speed).
    outdoor_mps = travel_settings()["walk_mps"]
    access_s: dict[str, float] = {}
    ent_warnings: dict[str, list[str]] = {}
    for ent in building.entrances:
        eid = ent["id"]
        if eid in closed_ents:
            ent_warnings[eid] = [CLOSED_BY_OFFICER]
            continue
        spot = staging_for_entrance(building, eid)
        dist = _distance_m(spot, ent) if spot is not None else None
        if dist is not None:
            access_s[eid] = round(dist / outdoor_mps, 1)
        ent_warnings[eid] = entrance_warnings(ent)

    credit = 0.0
    if situation is not None and situation.meet_person:
        credit = _travel_setting("meet_person_credit_s", default=24.0)

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
        "stairs_s_to_target_floor": None,
        "elevator_s_to_target_floor": None,
        "entrance_access_s": access_s,
        "entrance_warnings": ent_warnings,
        "meet_person_credit_s": credit,
        "predicted_curb_to_target_s": None,
        "warnings": [],
    }
    if not exp_travel:
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

    ent_room = rooms.get(ent.get("room_id", ""))

    # Vertical leg, stairs and elevator side by side.
    if target is not None and ent_room is not None:
        crossed = floors_entered(building, ent_room.floor, target.floor)
        result["stairs_s_to_target_floor"] = round(stair_climb_s(prof, crossed), 1)
        if crossed and elevator_serves(building, ent_room.floor, target.floor, closed_rooms):
            ride = elevator_ride_s(prof, len(crossed))
            result["elevator_s_to_target_floor"] = None if ride is None else round(ride, 1)
        light_to = _travel_setting("light_rate_measured_to_floor", default=10)
        if situation is not None and situation.load == "light" and crossed and max(crossed) > light_to:
            result["warnings"].append(
                f"Stair time above floor {light_to:g} uses the light-load rate, timed only up to about floor {light_to:g}"
            )

    # Curb to the most likely room, through the recommended entrance.
    if target is not None and best_id in G and target_id in G:
        inside = travel_times_from(G, best_id).get(target_id)
        if inside is not None and math.isfinite(inside):
            predicted = max(0.0, access_s.get(best_id, 0.0) + inside - credit)
            result["predicted_curb_to_target_s"] = round(predicted, 1)
            min_floors = _travel_setting("sanity_check", "min_floors_exclusive", default=2)
            floor_min = _travel_setting("sanity_check", "floor_min", default=2.0)
            if len(building.floors) > min_floors and predicted < floor_min * 60.0:
                result["warnings"].append(FASTER_THAN_MEASURED)

    # Stairwell: first stairwell room on the fastest route entrance -> top room.
    if target is not None and best_id in G and target_id in G:
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

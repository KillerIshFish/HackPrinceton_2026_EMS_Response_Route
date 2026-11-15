"""Step B1: travel profiles, elevators, smoke, officer gates and entrance access."""

from __future__ import annotations

import pytest

from core.building_graph import (
    build_graph,
    edge_time_s,
    elevator_ride_s,
    infer_elevator_edges,
    shortest_path,
    stair_climb_s,
    travel_times_from,
    with_inferred_elevators,
)
from core.schema import Building, Gates, Room, Situation, TravelProfile
from ingest.matterport import load_building
from planning.approach import (
    CLOSED_BY_OFFICER,
    FASTER_THAN_MEASURED,
    NO_KEY_HOLDER,
    entrance_warnings,
    recommend_approach,
)
from situation.config import LABELS, section
from situation.travel import profile_for

TARGET = "f4-office-403"  # floor 4; both entrances are on floor 1


@pytest.fixture(scope="module")
def building():
    return load_building("ptc-100")


@pytest.fixture(scope="module")
def probs(building):
    p = {r.id: 1.0 / len(building.rooms) for r in building.rooms}
    p[TARGET] = 0.5
    return p


# ---------------------------------------------------------------------------
# Defaults reproduce today's numbers
# ---------------------------------------------------------------------------

def test_edge_time_defaults_are_todays_constants() -> None:
    assert edge_time_s("door", 12.0) == pytest.approx(10.0)
    assert edge_time_s("stairs", 8.0, 1) == pytest.approx(8.0 / 1.2 + 15.0)
    assert edge_time_s("stairs", 8.0, 3) == pytest.approx(8.0 / 1.2 + 45.0)
    assert edge_time_s("elevator", 4.0, 2) == pytest.approx(20.0)


def test_default_profile_reproduces_current_travel_times(building) -> None:
    legacy = build_graph(building)
    # Every edge equals today's formula: length / 1.2 m/s (+ 15 s per floor on stairs).
    for u, v, d in legacy.edges(data=True):
        expected = d["length_m"] / 1.2 + (15.0 * d["floors"] if d["kind"] == "stairs" else 0.0)
        assert d["time_s"] == pytest.approx(expected), (u, v)
    assert not any(d["kind"] == "elevator" for _, _, d in legacy.edges(data=True))

    for profile in (TravelProfile(), profile_for(Situation())):
        G = build_graph(building, profile=profile)
        walk = {(u, v) for u, v, d in G.edges(data=True) if d["kind"] != "elevator"}
        assert walk == set(legacy.edges)
        for u, v, d in legacy.edges(data=True):
            assert G[u][v]["time_s"] == pytest.approx(d["time_s"])
        # Without rides, today's times exactly; with them, never slower (a ride only wins where
        # the walk to the stairs is long: the far side of floor 4).
        stairs_only = build_graph(building, use_elevators=False, profile=profile)
        assert travel_times_from(stairs_only, "ent-main") == pytest.approx(travel_times_from(legacy, "ent-main"))
        old = travel_times_from(legacy, "ent-main")
        assert all(t <= old[n] + 1e-9 for n, t in travel_times_from(G, "ent-main").items())


def test_default_situation_profile(building) -> None:
    p = profile_for(Situation())
    assert p.walk_mps == pytest.approx(1.2) and p.stair_s_per_floor == pytest.approx(15.0)
    assert p.search_advance_mps is None and p.notes == []
    assert p.stair_s_per_floor_high == pytest.approx(90.0) and p.high_floor_from == 18
    a_old = recommend_approach(building, {TARGET: 1.0})
    a_new = recommend_approach(building, {TARGET: 1.0}, profile=p)
    keys = ("entrance_id", "staging_id", "stairwell_id", "stairs_s_to_target_floor")
    assert {k: a_old[k] for k in keys} == {k: a_new[k] for k in keys}
    # The default (unknown, medical) call may ride the elevator: expected travel only gets shorter.
    old_t, new_t = a_old["entrance_expected_travel_s"], a_new["entrance_expected_travel_s"]
    assert old_t.keys() == new_t.keys() and all(new_t[e] <= old_t[e] for e in old_t)
    assert a_old["stairs_s_to_target_floor"] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Load, height, smoke
# ---------------------------------------------------------------------------

def test_heavy_load_doubles_the_stair_component(building, probs) -> None:
    light = recommend_approach(building, probs, situation=Situation(load="light"))
    heavy = recommend_approach(building, probs, situation=Situation(load="heavy"))
    assert light["stairs_s_to_target_floor"] == pytest.approx(3 * 15.0)
    assert heavy["stairs_s_to_target_floor"] == pytest.approx(3 * 29.0)
    assert heavy["stairs_s_to_target_floor"] / light["stairs_s_to_target_floor"] == pytest.approx(2.0, abs=0.1)

    # Same on the graph: the stair part of a stair edge, walking part unchanged.
    lp, hp = profile_for(Situation(load="light")), profile_for(Situation(load="heavy"))
    Gl, Gh = build_graph(building, profile=lp), build_graph(building, profile=hp)
    stairs = [(u, v, d) for u, v, d in Gl.edges(data=True) if d["kind"] == "stairs"]
    assert stairs
    for u, v, d in stairs:
        walk = d["length_m"] / 1.2
        assert (Gh[u][v]["time_s"] - walk) == pytest.approx(29.0 / 15.0 * (d["time_s"] - walk))


def test_high_floor_rate_applies_above_floor_18() -> None:
    p = profile_for(Situation())
    assert edge_time_s("stairs", 0.0, 1, p, from_floor=10) == pytest.approx(15.0)
    assert edge_time_s("stairs", 0.0, 2, p, from_floor=17) == pytest.approx(15.0 + 90.0)
    heavy = profile_for(Situation(load="heavy"))
    assert edge_time_s("stairs", 0.0, 2, heavy, from_floor=17) == pytest.approx(29.0 + 90.0)


def test_smoke_rates_only_when_smoke_reported() -> None:
    clear = profile_for(Situation(incident_type="fire_smoke"))
    assert clear.walk_mps == pytest.approx(1.2) and clear.search_advance_mps is None
    smoke = profile_for(Situation(incident_type="fire_smoke", smoke_reported=True))
    assert smoke.walk_mps == pytest.approx(0.5)
    assert smoke.search_advance_mps == pytest.approx(0.040)
    assert any("smoke rates" in n for n in smoke.notes)
    assert edge_time_s("door", 10.0, 0, smoke) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Elevator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("situation", [
    Situation(incident_type="fire_smoke"),
    Situation(incident_type="cardiac_arrest", smoke_reported=True),
    Situation(incident_type="active_threat"),
])
def test_no_elevator_for_fire_smoke_or_non_medical(building, probs, situation) -> None:
    p = profile_for(situation)
    assert not p.elevator_allowed and p.elevator_fixed_s is None and p.elevator_s_per_floor is None
    a = recommend_approach(building, probs, situation=situation)
    assert a["elevator_s_to_target_floor"] is None
    assert a["stairs_s_to_target_floor"] is not None
    G = build_graph(building, use_elevators=True, profile=p)
    assert not any(d["kind"] == "elevator" for _, _, d in G.edges(data=True))


def test_medical_shows_stairs_and_elevator_side_by_side(building, probs) -> None:
    sit = Situation(incident_type="cardiac_arrest")
    p = profile_for(sit)
    assert p.elevator_allowed and p.elevator_fixed_s == pytest.approx(75.0) and p.elevator_s_per_floor == pytest.approx(2.0)
    a = recommend_approach(building, probs, situation=sit)
    assert a["stairs_s_to_target_floor"] == pytest.approx(45.0)
    assert a["elevator_s_to_target_floor"] == pytest.approx(75.0 + 2.0 * 3)
    # The planner graph carries the rides whenever the profile offers them; a ride pays the fixed time once.
    G = build_graph(building, profile=p)
    assert G["f1-elevator"]["f4-elevator"]["time_s"] == pytest.approx(81.0)
    # They can still be left out on request.
    assert not any(d["kind"] == "elevator" for _, _, d in build_graph(building, use_elevators=False, profile=p).edges(data=True))


@pytest.mark.parametrize("load, first_floor_count", [("light", 6), ("heavy", 3)])
def test_elevator_beats_the_stairs_from_6_floors_light_3_heavy(load, first_floor_count) -> None:
    p = profile_for(Situation(incident_type="cardiac_arrest", load=load))
    for n in range(1, 12):
        ride = elevator_ride_s(p, n)
        climb = stair_climb_s(p, range(2, n + 2))
        assert (ride < climb) == (n >= first_floor_count), (load, n, ride, climb)


def test_route_takes_the_elevator_when_it_is_faster(building) -> None:
    quick = TravelProfile(elevator_allowed=True, elevator_fixed_s=10.0, elevator_s_per_floor=2.0)
    path = shortest_path(build_graph(building, profile=quick), "ent-main", TARGET)
    assert "f1-elevator" in path and "f4-elevator" in path
    assert not any(n.startswith(("f2-", "f3-")) for n in path), "one ride, no stops on the floors between"
    # Same building, the stairs are faster for a 120 s ride: the route climbs.
    slow = quick.model_copy(update={"elevator_fixed_s": 120.0})
    assert "f4-elevator" not in shortest_path(build_graph(building, profile=slow), "ent-main", TARGET)
    # No elevator offered (fire): never a ride, however fast it would be.
    no = quick.model_copy(update={"elevator_allowed": False})
    assert "f4-elevator" not in shortest_path(build_graph(building, profile=no), "ent-main", TARGET)
    # The stair option keeps its stairwell when the ride is faster.
    a = recommend_approach(building, {TARGET: 1.0}, profile=quick)
    assert a["stairwell_id"] is not None and a["elevator_s_to_target_floor"] == pytest.approx(16.0)


def _stack(floors: int, lobby_name: str = "Elevator lobby", elevator: bool = False) -> Building:
    """A tower: per floor a corridor and a lobby stacked at the same spot, stairs at the far end."""
    def room(rid: str, floor: int, name: str, rtype: str, x0: float, x1: float) -> Room:
        return Room(id=rid, name=name, floor=floor, floor_label=str(floor),
                    polygon=[(x0, 0.0), (x1, 0.0), (x1, 4.0), (x0, 4.0)],
                    centroid=((x0 + x1) / 2, 2.0, 4.0 * (floor - 1)), area_m2=4.0 * (x1 - x0),
                    room_type=rtype, has_windows=False, near_elevator=False, near_stairs=False,
                    occupancy_factor=1.0)
    rooms, edges = [], []
    for f in range(1, floors + 1):
        rooms += [room(f"f{f}-hall", f, "Corridor", "corridor", 0, 20),
                  room(f"f{f}-lob", f, "Lobby" if f == 1 else lobby_name, "lobby" if f == 1 else "corridor", 20, 24),
                  room(f"f{f}-stair", f, "Stair", "stairwell", -3, 0)]
        edges += [{"u": f"f{f}-hall", "v": f"f{f}-lob", "kind": "door", "length_m": 12.0, "floors": 0},
                  {"u": f"f{f}-hall", "v": f"f{f}-stair", "kind": "door", "length_m": 11.5, "floors": 0}]
        if elevator:
            rooms.append(room(f"f{f}-el", f, "Elevator", "elevator", 24, 26))
            edges.append({"u": f"f{f}-lob", "v": f"f{f}-el", "kind": "door", "length_m": 2.0, "floors": 0})
            if f > 1:
                edges.append({"u": f"f{f - 1}-el", "v": f"f{f}-el", "kind": "elevator", "length_m": 4.0, "floors": 1})
        if f > 1:
            edges.append({"u": f"f{f - 1}-stair", "v": f"f{f}-stair", "kind": "stairs", "length_m": 8.0, "floors": 1})
    return Building(id="tower", name="Tower", use_type="office", floors=list(range(1, floors + 1)),
                    floor_heights_m={f: 4.0 * (f - 1) for f in range(1, floors + 1)}, rooms=rooms, edges=edges,
                    entrances=[{"id": "ent", "name": "Front", "room_id": "f1-lob", "xy": [22.0, -1.0]}],
                    staging_spots=[])


def test_elevator_inferred_from_stacked_lobbies() -> None:
    b = _stack(8)
    inferred = infer_elevator_edges(b)
    # the ground floor's main lobby is the first stop; one link per pair of adjacent floors
    assert [(e["u"], e["v"]) for e in inferred] == [(f"f{f}-lob", f"f{f + 1}-lob") for f in range(1, 8)]
    assert all(e["kind"] == "elevator" and e["inferred"] and e["floors"] == 1 for e in inferred)

    medical = profile_for(Situation(incident_type="cardiac_arrest"))
    path = shortest_path(build_graph(with_inferred_elevators(b), profile=medical), "ent", "f8-hall")
    assert path[:3] == ["ent", "f1-lob", "f8-lob"], "seven floors up: the inferred elevator beats the stairs"
    fire = profile_for(Situation(incident_type="fire_smoke"))
    assert "f8-stair" in shortest_path(build_graph(with_inferred_elevators(b), profile=fire), "ent", "f8-hall")


def test_no_elevator_inferred_where_one_is_drawn_or_named_otherwise() -> None:
    assert infer_elevator_edges(_stack(5, elevator=True)) == []
    assert infer_elevator_edges(_stack(5, lobby_name="Corridor")) == []
    assert with_inferred_elevators(_stack(5, elevator=True)).edges == _stack(5, elevator=True).edges


def test_same_floor_target_has_no_vertical_leg(building) -> None:
    a = recommend_approach(building, {"f1-lobby": 1.0}, situation=Situation(incident_type="stroke"))
    assert a["stairs_s_to_target_floor"] == 0.0 and a["elevator_s_to_target_floor"] is None


# ---------------------------------------------------------------------------
# Officer gates
# ---------------------------------------------------------------------------

def test_closed_nodes_left_out_of_graph(building) -> None:
    G = build_graph(building, closed_room_ids=["f2-stair-west"], closed_entrance_ids=["ent-rear"])
    assert "f2-stair-west" not in G and "ent-rear" not in G and "ent-main" in G
    assert len(G) == len(build_graph(building)) - 2


def test_closed_entrance_is_never_recommended(building, probs) -> None:
    base = recommend_approach(building, probs)
    best = base["entrance_id"]
    other = next(e["id"] for e in building.entrances if e["id"] != best)

    for kwargs in ({"gates": Gates(closed_entrance_ids=[best])},
                   {"situation": Situation(closed_entrance_ids=[best])}):
        a = recommend_approach(building, probs, **kwargs)
        assert a["entrance_id"] == other
        assert best not in a["entrance_expected_travel_s"]
        assert a["entrance_warnings"][best] == [CLOSED_BY_OFFICER]

    # Even with a prebuilt graph that still holds the closed entrance.
    a = recommend_approach(building, probs, build_graph(building), gates=Gates(closed_entrance_ids=[best]))
    assert a["entrance_id"] == other

    everything = Gates(closed_entrance_ids=[e["id"] for e in building.entrances])
    a = recommend_approach(building, probs, gates=everything)
    assert a["entrance_id"] == "" and a["staging_id"] == ""


def test_closed_room_is_not_the_target(building, probs) -> None:
    a = recommend_approach(building, probs, gates=Gates(closed_room_ids=[TARGET]))
    assert a["target_room_id"] != TARGET


# ---------------------------------------------------------------------------
# Access, meet-person credit, sanity check
# ---------------------------------------------------------------------------

def test_access_time_and_barrier_warnings(building, probs) -> None:
    a = recommend_approach(building, probs)
    # Street staging is 12 m from the main entrance, parking staging 14 m from the rear one.
    assert a["entrance_access_s"] == {"ent-main": pytest.approx(10.0), "ent-rear": pytest.approx(11.7)}
    # Nothing recorded about either entrance: not assumed open.
    assert all(NO_KEY_HOLDER in w for w in a["entrance_warnings"].values())

    assert entrance_warnings({"access": "open"}) == []
    assert entrance_warnings({"access": "open", "key_holder": "Concierge"}) == []
    w = entrance_warnings({"access": "locked", "barriers": ["door code"]})
    assert NO_KEY_HOLDER in w and "Door code needed; no code recorded" in w
    assert sum("Barrier recorded" in x for x in w) == 2
    assert NO_KEY_HOLDER not in entrance_warnings({"access": "locked", "key_holder": "Night security"})


def test_meet_person_credit_applies_once(building, probs) -> None:
    plain = recommend_approach(building, probs, situation=Situation())
    met = recommend_approach(building, probs, situation=Situation(meet_person="Building security at the main door"))
    assert plain["meet_person_credit_s"] == 0.0 and met["meet_person_credit_s"] == pytest.approx(24.0)
    assert plain["predicted_curb_to_target_s"] - met["predicted_curb_to_target_s"] == pytest.approx(24.0, abs=0.11)
    # Not spread over entrances or edges.
    assert met["entrance_access_s"] == plain["entrance_access_s"]
    assert met["entrance_expected_travel_s"] == plain["entrance_expected_travel_s"]
    assert met["stairs_s_to_target_floor"] == plain["stairs_s_to_target_floor"]


def test_faster_than_measured_warning(building, probs) -> None:
    a = recommend_approach(building, probs)
    assert a["predicted_curb_to_target_s"] < 120.0
    assert a["warnings"].count(FASTER_THAN_MEASURED) == 1
    # A slow crew is not flagged (stairs only: with the elevator a heavy crew rides up and is quick).
    heavy = recommend_approach(building, probs, situation=Situation(load="heavy", incident_type="fire_smoke"))
    assert heavy["predicted_curb_to_target_s"] >= 120.0 and FASTER_THAN_MEASURED not in heavy["warnings"]
    # Two floors or fewer: no check.
    low = building.model_copy(update={"floors": [1, 2]})
    assert FASTER_THAN_MEASURED not in recommend_approach(low, probs)["warnings"]


def test_travel_settings_are_labelled() -> None:
    def walk(node, path=""):
        if isinstance(node, dict) and "value" in node:
            assert node.get("label") in LABELS, path
            assert node.get("source"), path
            return 1
        return sum(walk(v, f"{path}.{k}") for k, v in node.items()) if isinstance(node, dict) else 0

    travel = section("travel")
    assert walk(travel, "travel") >= 10
    assert travel["stair_s_per_floor_high"]["label"] == "judgement"
    assert travel["elevator"]["fixed_s"]["label"] == "judgement"

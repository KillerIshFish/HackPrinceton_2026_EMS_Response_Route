"""Step B1: travel profiles, elevators, smoke, officer gates and entrance access."""

from __future__ import annotations

import pytest

from core.building_graph import build_graph, edge_time_s, travel_times_from
from core.schema import Gates, Situation, TravelProfile
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
        assert set(G.edges) == set(legacy.edges)
        for u, v, d in legacy.edges(data=True):
            assert G[u][v]["time_s"] == pytest.approx(d["time_s"])
        assert travel_times_from(G, "ent-main") == pytest.approx(travel_times_from(legacy, "ent-main"))


def test_default_situation_profile(building) -> None:
    p = profile_for(Situation())
    assert p.walk_mps == pytest.approx(1.2) and p.stair_s_per_floor == pytest.approx(15.0)
    assert p.search_advance_mps is None and p.notes == []
    assert p.stair_s_per_floor_high == pytest.approx(90.0) and p.high_floor_from == 18
    a_old = recommend_approach(building, {TARGET: 1.0})
    a_new = recommend_approach(building, {TARGET: 1.0}, profile=p)
    keys = ("entrance_id", "staging_id", "stairwell_id", "entrance_expected_travel_s", "stairs_s_to_target_floor")
    assert {k: a_old[k] for k in keys} == {k: a_new[k] for k in keys}
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
    # The planner graph still never routes by elevator unless asked.
    assert not any(d["kind"] == "elevator" for _, _, d in build_graph(building, profile=p).edges(data=True))
    # When asked, a ride pays the fixed time once.
    G = build_graph(building, use_elevators=True, profile=p)
    assert G["f1-elevator"]["f4-elevator"]["time_s"] == pytest.approx(81.0)


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
    # A slow crew is not flagged.
    heavy = recommend_approach(building, probs, situation=Situation(load="heavy"))
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

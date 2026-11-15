"""Tests for buildings traced from a published floor plan (ingest/unified/plan.py).

No network: the model is a pure function of configs/plan_buildings.yaml, and the site
context is built with ``network=False`` into a temporary folder.

Run: uv run pytest tests/test_plan_building.py
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
from shapely.geometry import Polygon

from core.building_graph import build_graph
from core.clue_likelihood import clue_match
from core.evidence import basement_floors, name_labels, unit_room_ids
from core.prior import compute_prior
from core.schema import Building, Clue
from ingest.georegister import latlng_to_local
from ingest.unified import context as ctx_mod
from ingest.unified.plan import build_plan_model, plan_config
from ingest.unified.project import placement, to_building, to_latlng, viewer_payload

JRR = "jrr-princeton"


@pytest.fixture(scope="module")
def model():
    return build_plan_model(plan_config(JRR))


@pytest.fixture(scope="module")
def building(model) -> Building:
    return to_building(model)


def test_unknown_plan_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        plan_config("no-such-building")


def test_model_is_deterministic_and_carries_its_own_placement(model) -> None:
    again = build_plan_model(plan_config(JRR))
    assert model.to_json() == again.to_json()
    assert model.archetype == "floor_plan" and model.tier == "traced-plan"
    assert {s.fidelity for s in model.spaces} == {"traced"}
    assert [lv.label for lv in model.levels] == ["B", "A", "1", "2", "3"]
    # 20 Washington Road, Princeton; the plan's "up" is about 31 degrees west of north
    assert model.geolocation is not None
    assert abs(model.geolocation[0] - 40.3492) < 0.001 and abs(model.geolocation[1] + 74.6552) < 0.001
    assert 25.0 < model.yaw_deg < 37.0
    assert model.stats["georeference_rms_m"] < 1.0


def test_control_points_land_on_the_map_corners(model) -> None:
    place = placement(model, None)
    cfg = plan_config(JRR)
    scale, (ox, oy) = cfg["metres_per_px"], cfg["origin_px"]
    for cp in cfg["georeference"]["control_points"]:
        x, y = (cp["px"][0] - ox) * scale, -(cp["px"][1] - oy) * scale
        lat, lng = to_latlng(place, x, y)
        east, north = latlng_to_local(lat, lng, cp["latlng"])
        assert math.hypot(east, north) < 1.5, cp


def test_every_room_is_reachable_and_named_rooms_exist(model, building) -> None:
    assert len(building.rooms) == len(model.spaces) > 150
    assert nx.is_connected(build_graph(building))
    assert model.stats["links_inferred"] == 0  # every link was read off the plan
    names = {r.id: r.name for r in building.rooms}
    for rid in ("a17", "a71", "101", "198", "117", "171", "200", "217", "271", "301", "399", "b60b"):
        assert rid in names, rid
    by_id = {r.id: r for r in building.rooms}
    assert by_id["a17"].room_type == "classroom" and by_id["a17"].floor_label == "A"
    assert by_id["171"].room_type == "lobby" and by_id["171"].floor_label == "1"
    assert by_id["1-piirs"].room_type == "suite"  # a department's area stands for its offices
    assert by_id["399"].floor_label == "3" and by_id["399"].has_windows
    assert not by_id["a17"].has_windows  # the lecture bowl is in the middle of the block
    assert by_id["a-elev-1"].near_elevator and by_id["1-stair-nw"].near_stairs


def test_rooms_drawn_inside_a_department_are_cut_out_of_it(model) -> None:
    by_id = {s.id: Polygon(s.polygon) for s in model.spaces}
    for inner, outer in (("144", "1-piirs-w"), ("a66", "a-oip"), ("254", "2-ir"), ("117", "1-forum")):
        assert by_id[inner].intersection(by_id[outer]).area < 0.5, (inner, outer)
    # nothing on a level overlaps anything else by more than tracing noise
    for lv in model.levels:
        polys = [(s.id, Polygon(s.polygon)) for s in model.spaces if s.level == lv.index]
        for i, (a, pa) in enumerate(polys):
            for b, pb in polys[i + 1:]:
                assert pa.intersection(pb).area < 1.0, (a, b)


def test_stairs_and_elevators_join_the_levels(model) -> None:
    level = {s.id: s.level for s in model.spaces}
    vertical = [ln for ln in model.links if ln.kind in ("stairs", "elevator")]
    assert all(ln.floors == abs(level[ln.a] - level[ln.b]) >= 1 for ln in vertical)
    pairs = {tuple(sorted((level[ln.a], level[ln.b]))) for ln in vertical}
    assert {(1, 2), (2, 3), (3, 4), (4, 5)} <= pairs
    assert any(ln.kind == "elevator" for ln in vertical)


def test_entrances_are_on_their_own_levels(model, building) -> None:
    rooms = {r.id: r for r in building.rooms}
    by_entrance = {e["id"]: rooms[e["room_id"]] for e in building.entrances}
    assert by_entrance["ent-washington"].floor_label == "1"
    assert by_entrance["ent-scudder"].floor_label == "1"
    assert by_entrance["ent-north"].floor_label == "A"
    payload = viewer_payload(model)
    drawn = {e["id"]: e["l"] for e in payload["entrances"]}
    assert drawn == {"ent-washington": 3, "ent-scudder": 3, "ent-north": 2}
    assert payload["counts"]["fidelity"] == {"traced": len(model.spaces)}
    assert len(payload["walls"]) > 500


def test_callers_words_find_the_rooms(building) -> None:
    assert unit_room_ids(building, "A17", {}) == ["a17"]
    assert unit_room_ids(building, "399", {}) == ["399"]
    assert basement_floors(building) == [1]  # the B level
    m = clue_match(building, Clue(type="room_type", value="classroom", polarity=True, confidence=1.0, span="lecture hall"))
    hits = {rid for rid, v in m.items() if v == 1.0}
    assert {"a17", "a71", "101", "a02"} <= hits and "102" not in hits
    m = clue_match(building, Clue(type="floor", value="2", polarity=True, confidence=1.0, span="second floor"))
    assert m["200"] == 1.0 and m["101"] < 1.0 and m["b60a"] == 0.0
    room_probs, p_not_in_model = compute_prior(building, hour=11)
    assert abs(sum(room_probs.values()) + p_not_in_model - 1.0) < 1e-9
    # a label printed on a door belongs to one room only
    seen: dict[str, str] = {}
    for r in building.rooms:
        for label in name_labels(r.name):
            assert label not in seen, (label, seen[label], r.id)
            seen[label] = r.id


def test_site_context_keeps_the_plans_own_placement(model, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", tmp_path)
    ctx = ctx_mod.build_context(model, network=False)
    assert ctx["yaw_deg"] == model.yaw_deg and ctx["anchor_latlng"] is None and ctx["fit_scale"] is None
    assert "floor plan" in ctx["placement_source"]
    b = to_building(model, ctx)
    ent = next(e for e in b.entrances if e["id"] == "ent-washington")
    # the Washington Road door is on the west side of the building, about 40 m from its middle
    east, north = latlng_to_local(ent["latlng"][0], ent["latlng"][1], model.geolocation)
    assert 30.0 < math.hypot(east, north) < 50.0 and east < 0


def test_scripted_call_ends_in_the_lecture_hall(building) -> None:
    from api.session import CallSession
    from sim.generate_calls import JRR_TRUE_ROOM_ID, demo_script_for

    script = demo_script_for(building.id, building)
    assert "20 Washington Road" in script[0]["text"]
    session = CallSession(building)
    tops = []
    for _ in script[1:]:
        session.step()
        probs = session.state().posterior.room_probs
        tops.append(max(probs, key=probs.get))
    state = session.state()
    assert state.done and tops[-1] == JRR_TRUE_ROOM_ID
    assert state.posterior.room_probs[JRR_TRUE_ROOM_ID] > 0.9
    # "a lecture hall, no windows" leaves the two lecture bowls ahead of every other room
    after_room_type = CallSession(building)
    after_room_type.step()
    after_room_type.step()
    probs = after_room_type.state().posterior.room_probs
    assert set(sorted(probs, key=probs.get, reverse=True)[:2]) == {"a17", "a71"}
    # the crew is sent in by the entrance on the caller's level
    assert state.plan.entrance_id == "ent-north" and state.plan.teams[0][0].room_id == JRR_TRUE_ROOM_ID

"""Tests for the pieces added after v0: free-text utterances, parser modes, building
switching, the tile proxy guard, and (when it has been built) the real building.

Nothing here touches the network: the parser stays in ``rules`` mode, and the
real-building tests read ``data/processed/`` and are skipped when it is absent.

Run: uv run pytest
"""

from __future__ import annotations

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from api.server import app
from core.building_graph import build_graph
from core.clue_likelihood import clue_match
from core.schema import Clue
from ingest import vexcel
from ingest.matterport import list_buildings, load_building
from ingest.real_building import load_meta, scan_rooms
from parser.extract import _validate_llm_clues, load_vocab

REAL_ID = "bundy-olympic"
HAS_REAL = any(b.id == REAL_ID for b in list_buildings())
needs_real = pytest.mark.skipif(not HAS_REAL, reason="run `uv run python -m ingest.real_building` first")


@pytest.fixture()
def client() -> TestClient:
    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    c.post("/calls/demo/building", params={"id": "ptc-100"})
    return c


def _total(post: dict) -> float:
    return sum(post["room_probs"].values()) + post["p_not_in_model"]


# ---------------------------------------------------------------------------
# API additions (synthetic building, offline)
# ---------------------------------------------------------------------------


def test_utterance_adds_a_parsed_caller_turn(client: TestClient) -> None:
    before = client.get("/calls/demo/state").json()
    s = client.post("/calls/demo/utterance", json={"text": "I'm locked in a bathroom on the fourth floor"}).json()
    assert len(s["transcript"]) == len(before["transcript"]) + 1
    last = s["transcript"][-1]
    assert last["speaker"] == "caller" and last["scripted"] is False and last["parser"] == "rules"
    assert {(c["type"], c["value"]) for c in last["clues"]} == {("room_type", "restroom"), ("floor", "4")}
    assert s["turn"] == before["turn"], "typing an utterance must not advance the script"
    assert len(s["history"]) == len(s["transcript"])
    assert s["posterior"]["entropy_bits"] < before["posterior"]["entropy_bits"]
    assert _total(s["posterior"]) == pytest.approx(1.0, abs=1e-9)
    assert client.post("/calls/demo/utterance", json={"text": "   "}).status_code == 422


def test_parser_mode_and_building_switch(client: TestClient) -> None:
    s = client.get("/calls/demo/state").json()
    assert s["parser_mode"] == "rules"
    assert s["buildings"][0]["id"] == "ptc-100" and s["buildings"][0]["real"] is False
    assert client.post("/calls/demo/parser", params={"mode": "nope"}).status_code == 422
    assert client.post("/calls/demo/building", params={"id": "nope"}).status_code == 404
    meta = client.get("/buildings/ptc-100/meta").json()
    assert meta["real"] is False and meta["matterport_models"] == []


def test_tile_proxy_refuses_tiles_outside_a_site_window(client: TestClient) -> None:
    assert client.get("/vexcel/tile/urban/20/1/1").status_code == 404


def test_config_exposes_no_secret(client: TestClient) -> None:
    cfg = client.get("/config").json()
    assert set(cfg) == {"matterport_sdk_key", "llm_available", "llm_model", "vexcel_available"}


# ---------------------------------------------------------------------------
# Parser / likelihood units
# ---------------------------------------------------------------------------


def test_llm_reply_validation_drops_invented_spans_and_values() -> None:
    text = "I'm in a janitor's cupboard, nothing to look out of"
    raw = [
        {"type": "room_type", "value": "storage", "polarity": True, "confidence": 0.85, "span": "Janitor's Cupboard"},
        {"type": "negative", "value": "landmarks:window", "polarity": True, "confidence": 0.8, "span": "nothing to look out of"},
        {"type": "room_type", "value": "dungeon", "polarity": True, "confidence": 0.9, "span": "cupboard"},
        {"type": "view", "value": "street", "polarity": True, "confidence": 0.9, "span": "the street outside"},
        {"type": "floor", "value": "8", "polarity": True, "confidence": 0.2, "span": "cupboard"},
    ]
    clues = _validate_llm_clues(text, raw, load_vocab())
    assert [(c.type, c.value) for c in clues] == [("room_type", "storage"), ("negative", "landmarks:window")]
    assert clues[0].span == "janitor's cupboard", "span must be the caller's own text"
    assert clues[1].polarity is False


def test_view_of_a_parking_structure_is_a_view_not_a_room() -> None:
    from parser.extract import extract_clues_rules

    seen = {(c.type, c.value) for c in extract_clues_rules("I can see the parking structure out the window")}
    assert ("view", "parking_lot") in seen and ("room_type", "parking") not in seen
    inside = {(c.type, c.value) for c in extract_clues_rules("I'm in the parking garage")}
    assert ("room_type", "parking") in inside


def test_web_mercator_round_trip() -> None:
    lat, lng = 34.0322954, -118.4522354
    x, y = vexcel.latlng_to_tile(lat, lng, 20)
    back = vexcel.tile_to_latlng(x, y, 20)
    assert back == pytest.approx((lat, lng), abs=1e-9)
    t = vexcel.site_tiles((lat, lng), 95, 20)
    assert t["x_min"] <= x <= t["x_max"] + 1 and t["y_min"] <= y <= t["y_max"] + 1
    assert 0.11 < t["m_per_px"] < 0.14


def test_scan_rooms_maps_tags_and_reads_windows() -> None:
    model = {
        "rooms": [
            {"id": "a", "tags": ["other", "unfurnished"], "dimensions": {"areaFloor": 80.0},
             "boundary": {"edges": [{"openings": [{"type": "window"}, {"type": "doorway"}]}]}},
            {"id": "b", "tags": ["closet"], "dimensions": {"areaFloor": 6.0},
             "boundary": {"edges": [{"openings": [{"type": "doorway"}]}]}},
            {"id": "void", "tags": [], "dimensions": {"areaFloor": 0.0}, "boundary": {"edges": []}},
            {"id": "c", "tags": ["bedroom"], "dimensions": {"areaFloor": 12.0}, "boundary": {"edges": []}},
        ],
        "locations": [{"id": "s1", "room": {"id": "b"}, "position": {"x": 0, "y": 0, "z": 0}}],
    }
    rooms = scan_rooms(model)
    assert [(r["room_type"], r["has_windows"], r["windows_source"]) for r in rooms] == [
        ("open_office", True, "scan"), ("office", True, "inferred"), ("storage", False, "scan"),
    ]
    assert rooms[2]["sweep_id"] == "s1"


# ---------------------------------------------------------------------------
# The real building (skipped until it has been built)
# ---------------------------------------------------------------------------


def test_room_ring_orders_wall_edges_and_drops_stubs() -> None:
    from ingest.scan_building import room_ring, scan_room_type

    def edge(i: int, a: tuple, b: tuple) -> dict:
        return {"id": f"e{i}", "vertices": [{"id": f"{a}", "position": {"x": a[0], "y": a[1]}},
                                           {"id": f"{b}", "position": {"x": b[0], "y": b[1]}}]}

    square = [(0, 0), (4, 0), (4, 3), (0, 3)]
    edges = [edge(i, square[i], square[(i + 1) % 4]) for i in (2, 0, 3, 1)]  # out of order on purpose
    edges.append(edge(9, (4, 3), (6, 5)))  # a dangling wall stub
    ring = room_ring({"boundary": {"edges": edges}})
    assert sorted(ring) == sorted((float(x), float(y)) for x, y in square)
    assert room_ring({"boundary": {"edges": edges[:2]}}) == []

    assert scan_room_type({"label": "Conference Room", "classifications": [{"label": "Other"}]}, 30) == "conference"
    assert scan_room_type({"classifications": [{"label": "Office"}]}, 12) == "office"
    assert scan_room_type({"classifications": [{"label": "Office"}]}, 200) == "open_office"
    assert scan_room_type({"classifications": [{"label": "Bathroom"}]}, 3) == "restroom"
    assert scan_room_type({"classifications": [{"label": "Other"}]}, 10) == "room"


def test_georegistration_recovers_a_known_placement() -> None:
    from ingest.georegister import latlng_to_local, local_to_latlng, register_matterport_model

    origin, rotation = (34.0410269, -118.3571297), 32.87
    corners = [(-19.3, 13.4), (11.2, 13.4), (11.2, -10.1), (-19.3, -10.1)]
    points = [{"model_xy": list(c), "latlng": list(local_to_latlng(c[0], c[1], origin, rotation))} for c in corners]
    fit = register_matterport_model(points)
    assert fit["rotation_deg"] == pytest.approx(rotation, abs=0.05)
    assert fit["rms_m"] < 0.05
    assert latlng_to_local(*fit["origin_latlng"], origin, rotation) == pytest.approx((0.0, 0.0), abs=0.05)
    with pytest.raises(ValueError):
        register_matterport_model(points[:1])


SCAN_ID = "washington-5285"
needs_scan = pytest.mark.skipif(
    not any(b.id == SCAN_ID for b in list_buildings()), reason="run `uv run python -m ingest.real_building` first"
)


@needs_scan
def test_scan_building_is_real_geometry_end_to_end() -> None:
    b = load_building(SCAN_ID)
    meta = load_meta(SCAN_ID)
    assert meta["kind"] == "full_scan"
    assert {m["source"] for m in meta["rooms"].values()} == {"matterport"}, "nothing is schematic or inferred"
    assert nx.is_connected(build_graph(b))
    assert b.floors == [1, 2, 3] and 2.5 < b.floor_heights_m[2] < 3.5
    types = {r.room_type for r in b.rooms}
    assert {"office", "restroom", "break_room", "conference", "lobby", "warehouse"} <= types
    assert any(e["kind"] == "stairs" for e in b.edges)
    assert len(meta["inferred_links"]) <= 2
    assert b.entrances and all(e["facing"] == "street" for e in b.entrances)
    assert {r.window_facing for r in b.rooms if r.window_facing} >= {"street", "residential"}
    # every floor has a plan image whose extent contains that floor's rooms
    for floor in b.floors:
        plan = meta["floorplans"][str(floor)]
        x0, y0 = plan["origin"]
        x1, y1 = x0 + plan["width"] / plan["resolution"], y0 + plan["height"] / plan["resolution"]
        for r in b.rooms:
            if r.floor == floor:
                assert all(x0 - 0.5 <= x <= x1 + 0.5 and y0 - 0.5 <= y <= y1 + 0.5 for x, y in r.polygon), r.name


@needs_scan
def test_scan_building_demo_finds_the_callers_office_first() -> None:
    from sim.generate_calls import WASHINGTON_TRUE_ROOM_ID

    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    s = c.post("/calls/demo/building", params={"id": SCAN_ID}).json()
    rooms = {r["id"]: r for r in c.get(f"/buildings/{SCAN_ID}").json()["rooms"]}
    for _ in range(2):
        s = c.post("/calls/demo/step").json()
    before = s["posterior"]["room_probs"]
    s = c.post("/calls/demo/step").json()  # "I came up the stairs and turned right ..."
    assert [(k["type"], k["value"]) for k in s["clues"]][-1] == ("route", "stairs_up_right")
    after = s["posterior"]["room_probs"]
    # the rear offices are on the right of their stair, the front ones on the left of theirs
    rear = [r for r in rooms.values() if r["floor"] == 2 and r["room_type"] == "office" and r["centroid"][1] > 5]
    front = [r for r in rooms.values() if r["floor"] == 2 and r["room_type"] == "office" and r["centroid"][1] < 0]
    assert rear and front
    assert all(after[r["id"]] > before[r["id"]] for r in rear)
    assert all(after[r["id"]] < before[r["id"]] for r in front)
    s = c.post("/calls/demo/step").json()
    probs = s["posterior"]["room_probs"]
    top4 = sorted(probs, key=probs.get, reverse=True)[:4]
    assert WASHINGTON_TRUE_ROOM_ID in top4
    assert all(rooms[r]["floor"] == 2 and rooms[r]["window_facing"] == "residential" for r in top4)
    assert sum(probs[r] for r in top4) > 0.6
    plan = s["plan"]
    assert plan["teams"][0][0]["room_id"] == WASHINGTON_TRUE_ROOM_ID
    assert plan["expected_time_to_find_s"] < min(plan["baseline_nearest_s"], plan["baseline_highest_prob_s"])
    assert s["approach"]["stairwell_name"], "the route upstairs is named even though the stairs are open"

    png = c.get(f"/buildings/{SCAN_ID}/floorplan/2")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    assert c.get(f"/buildings/{SCAN_ID}/floorplan/9").status_code == 404
    assert c.get("/buildings/ptc-100/floorplan/1").status_code == 404
    c.post("/calls/demo/building", params={"id": "ptc-100"})


# ---------------------------------------------------------------------------
# Route clues ("I turned right after coming up the stairs")
# ---------------------------------------------------------------------------


def _corridor_building():
    """Two floors. Floor 2: a stair landing opens north onto an east-west corridor with a room at each end."""
    from core.schema import Building, Room

    def room(rid, floor, x0, y0, x1, y1, rtype):
        return Room(id=rid, name=rid, floor=floor, floor_label=str(floor),
                    polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    centroid=((x0 + x1) / 2, (y0 + y1) / 2, 0.0 if floor == 1 else 3.0),
                    area_m2=(x1 - x0) * (y1 - y0), room_type=rtype, has_windows=False,
                    near_elevator=False, near_stairs=False, occupancy_factor=1.0)

    rooms = [
        room("lobby", 1, -2, -6, 2, 0, "lobby"),
        room("landing", 2, -2, -6, 2, 0, "stairwell"),
        room("hall", 2, -20, 0, 20, 2, "corridor"),
        room("west", 2, -20, 2, -14, 8, "office"),
        room("east", 2, 14, 2, 20, 8, "office"),
    ]
    edges = [
        {"u": "lobby", "v": "landing", "kind": "stairs", "length_m": 6.0, "floors": 1,
         "top_xy": [0.0, -2.0], "heading": [0.0, 1.0]},  # the walker arrives heading north
        {"u": "landing", "v": "hall", "kind": "door", "length_m": 4.0, "floors": 0, "xy": [0.0, 0.0]},
        {"u": "hall", "v": "west", "kind": "door", "length_m": 18.0, "floors": 0},  # no xy: shared wall midpoint
        {"u": "hall", "v": "east", "kind": "door", "length_m": 18.0, "floors": 0},
    ]
    return Building(id="t", name="t", use_type="office", floors=[1, 2], floor_heights_m={1: 0.0, 2: 3.0},
                    rooms=rooms, edges=edges, entrances=[], staging_spots=[])


def test_route_clue_scores_rooms_by_the_turn_after_the_stairs() -> None:
    from core.route_clues import report_probability, turns_along

    assert turns_along([(0, 0), (0, 5), (5, 5)], (0.0, 1.0))[0] == ["right"]
    assert turns_along([(0, 0), (0, 5), (-5, 5), (-5, 10)], None)[0] == ["left", "right"]
    assert turns_along([(0, 0), (0, 5), (0.5, 10)], (0.0, 1.0))[0] == []
    assert report_probability(["right"], "right") > report_probability(["left", "right"], "right") > report_probability(["left"], "right")

    b = _corridor_building()
    right = clue_match(b, Clue(type="route", value="stairs_up_right", polarity=True, confidence=0.8, span="x"))
    left = clue_match(b, Clue(type="route", value="stairs_up_left", polarity=True, confidence=0.8, span="x"))
    # heading north out of the landing, east is on the right and west on the left
    assert right["east"] > 0.85 and right["west"] < 0.2
    assert left["west"] > 0.85 and left["east"] < 0.2
    assert right["lobby"] == 0.0, "coming up the stairs rules out the floor the stairs start on"
    # no elevator in this building: an elevator route clue says nothing
    lift = clue_match(b, Clue(type="route", value="elevator_left", polarity=True, confidence=0.8, span="x"))
    assert set(lift.values()) == {1.0}


def test_rule_parser_reads_route_statements() -> None:
    from parser.extract import extract_clues_rules

    def parsed(text):
        return [(c.type, c.value) for c in extract_clues_rules(text)]

    assert parsed("I turned right after coming up the staircase getting to the room") == [("route", "stairs_up_right")]
    assert parsed("got off the elevator, took a left") == [("route", "elevator_left")]
    assert parsed("I went down the stairs and kept going straight") == [("route", "stairs_down_straight")]
    # an unanchored left/right is not a route, and being in the stairwell is still a room clue
    assert parsed("the door is on the left side of the room") == []
    assert parsed("I'm in the stairwell") == [("room_type", "stairwell")]
    # the model's reply for a route clue passes validation
    text = "I turned right after coming up the staircase"
    kept = _validate_llm_clues(text, [{"type": "route", "value": "stairs_up_right", "polarity": True,
                                       "confidence": 0.8, "span": "turned right after coming up the staircase"},
                                      {"type": "route", "value": "stairs_sideways", "polarity": True,
                                       "confidence": 0.8, "span": "turned right"}], load_vocab())
    assert [(c.type, c.value) for c in kept] == [("route", "stairs_up_right")]


@needs_real
def test_real_building_is_valid_and_connected() -> None:
    b = load_building(REAL_ID)
    meta = load_meta(REAL_ID)
    assert b.floors == list(range(1, 9))
    assert b.floor_heights_m[8] == pytest.approx(7 * 12 * 0.3048, abs=0.01)
    assert nx.is_connected(build_graph(b))
    ids = {r.id for r in b.rooms}
    assert set(meta["rooms"]) == ids
    assert {m["source"] for m in meta["rooms"].values()} == {"matterport", "costar", "inferred"}
    for e in b.entrances:
        assert e["room_id"] in ids
    scanned = [m for m in meta["rooms"].values() if m["source"] == "matterport"]
    assert scanned and all(m["model_id"] for m in scanned)
    assert {m["suite"] for m in meta["matterport_models"]} >= {"420", "860"}


@needs_real
def test_unscanned_suite_partially_matches_an_interior_room_clue() -> None:
    b = load_building(REAL_ID)
    m = clue_match(b, Clue(type="room_type", value="storage", polarity=True, confidence=1.0, span="closet"))
    by_type = {r.room_type: m[r.id] for r in b.rooms}
    assert by_type["storage"] == 1.0
    assert 0.0 < by_type["suite"] < 1.0
    assert by_type["restroom"] == 0.0


@needs_real
def test_real_building_demo_narrows_to_the_parking_side_of_the_top_floor() -> None:
    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    states = [c.post("/calls/demo/building", params={"id": REAL_ID}).json()]
    for _ in range(3):
        states.append(c.post("/calls/demo/step").json())
    s = states[-1]
    assert s["done"] and s["building_id"] == REAL_ID
    b = {r["id"]: r for r in c.get(f"/buildings/{REAL_ID}").json()["rooms"]}

    pf = states[1]["posterior"]["p_floor"]
    assert pf["8"] > pf["7"] > pf["6"], pf
    assert pf["8"] + pf["7"] > 0.85

    post = s["posterior"]
    assert _total(post) == pytest.approx(1.0, abs=1e-9)
    top = sorted(post["room_probs"], key=post["room_probs"].get, reverse=True)[:3]
    names = [b[r]["name"] for r in top]
    assert all(b[r]["window_facing"] == "parking_lot" for r in top), names
    assert all(b[r]["floor"] == 8 for r in top[:2]), names  # floor 7 stays plausible: +/-3 m is two storeys
    assert post["room_probs"]["f8-s860-r2"] > post["room_probs"]["f8-s860-r3"], "the windowed office beats the windowless one"

    ents = [h["entropy_bits"] for h in s["history"]]
    assert ents == sorted(ents, reverse=True), ents
    plan = s["plan"]
    assert plan["expected_time_to_find_s"] <= plan["baseline_nearest_s"] + 1e-6
    assert plan["expected_time_to_find_s"] <= plan["baseline_highest_prob_s"] + 1e-6

    meta = c.get(f"/buildings/{REAL_ID}/meta").json()
    t = meta["site"]["tiles"]
    assert t["layer"] == "urban" and t["x_max"] >= t["x_min"]
    assert len(meta["site"]["footprint_latlng"]) >= 4
    c.post("/calls/demo/building", params={"id": "ptc-100"})

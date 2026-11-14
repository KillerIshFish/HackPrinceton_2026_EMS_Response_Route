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

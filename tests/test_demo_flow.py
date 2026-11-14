"""End-to-end acceptance test for the 4-turn scripted demo, through the HTTP API.

Run: uv run pytest
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import app

PL_RESTROOM = "f4-restroom-north"
CORE_RESTROOM = "f4-restroom-core"


@pytest.fixture()
def client() -> TestClient:
    """A fresh call on the synthetic building with the offline rule parser."""
    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    c.post("/calls/demo/building", params={"id": "ptc-100"})
    return c


def _floor_of(building: dict, room_id: str) -> str:
    return next(r["floor_label"] for r in building["rooms"] if r["id"] == room_id)


def _total(post: dict) -> float:
    return sum(post["room_probs"].values()) + post["p_not_in_model"]


def test_buildings_endpoint(client: TestClient) -> None:
    buildings = client.get("/buildings").json()
    b = buildings[0]  # the synthetic building is always first; real ones follow when built
    assert b["id"] == "ptc-100"
    assert b["name"] == "100 Princeton Tech Center"
    assert b["floors"] == [1, 2, 3, 4]
    f4_restrooms = sorted(r["id"] for r in b["rooms"] if r["floor"] == 4 and r["room_type"] == "restroom")
    assert f4_restrooms == [CORE_RESTROOM, PL_RESTROOM]


def test_turn0_prior_spans_all_floors(client: TestClient) -> None:
    s = client.get("/calls/demo/state").json()
    assert s["turn"] == 0 and s["total_turns"] == 4 and not s["done"]
    post = s["posterior"]
    assert _total(post) == pytest.approx(1.0, abs=1e-9)
    assert post["p_not_in_model"] == pytest.approx(0.05, abs=1e-9)
    assert set(post["p_floor"]) == {"1", "2", "3", "4"}
    assert all(p > 0.10 for p in post["p_floor"].values()), post["p_floor"]


def test_turn1_z_fix_narrows_to_floors_3_and_4(client: TestClient) -> None:
    s = client.post("/calls/demo/step").json()
    assert s["turn"] == 1
    assert s["phone_fix"]["z_m"] == pytest.approx(12.6)
    pf = s["posterior"]["p_floor"]
    in_model = 1.0 - s["posterior"]["p_not_in_model"]
    assert (pf["3"] + pf["4"]) / in_model >= 0.90, pf
    assert pf["4"] > pf["3"] > pf["2"] > pf["1"]


def test_turn2_bathroom_gives_two_peaks_on_floor_4(client: TestClient) -> None:
    building = client.get("/buildings/ptc-100").json()
    client.post("/calls/demo/step")
    s = client.post("/calls/demo/step").json()
    assert [c["type"] + "=" + c["value"] for c in s["clues"]] == ["room_type=restroom"]
    post = s["posterior"]
    assert _total(post) == pytest.approx(1.0, abs=1e-9)
    assert post["peak_count"] == 2
    top2 = sorted(post["room_probs"], key=post["room_probs"].get, reverse=True)[:2]
    assert sorted(top2) == [CORE_RESTROOM, PL_RESTROOM]
    assert all(_floor_of(building, r) == "4" for r in top2)


def test_turn3_parking_lot_concentrates_and_beats_baselines(client: TestClient) -> None:
    states = [client.get("/calls/demo/state").json()]
    for _ in range(3):
        states.append(client.post("/calls/demo/step").json())
    s = states[-1]
    assert s["done"] and s["turn"] == 3
    post = s["posterior"]
    probs = post["room_probs"]
    top = max(probs, key=probs.get)
    assert top == PL_RESTROOM
    runner_up = sorted(probs.values(), reverse=True)[1]
    assert probs[top] >= 0.4, probs[top]
    assert probs[top] > 5 * runner_up, (probs[top], runner_up)
    assert probs[top] > 5 * probs[CORE_RESTROOM]
    assert post["peak_count"] == 1

    plan = s["plan"]
    assert plan["teams"][0][0]["room_id"] == PL_RESTROOM
    assert plan["expected_time_to_find_s"] <= plan["baseline_nearest_s"] + 1e-6
    assert plan["expected_time_to_find_s"] <= plan["baseline_highest_prob_s"] + 1e-6
    assert plan["expected_time_to_find_s"] < plan["baseline_nearest_s"]

    # Evidence makes the picture sharper and the search faster, turn over turn.
    ents = [h["entropy_bits"] for h in s["history"]]
    ets = [h["expected_time_to_find_s"] for h in s["history"]]
    assert ents == sorted(ents, reverse=True), ents
    assert ets[-1] < ets[0], ets

    # Stepping past the end is a no-op.
    again = client.post("/calls/demo/step").json()
    assert again["turn"] == 3


def test_mark_cleared_koopman_replan(client: TestClient) -> None:
    for _ in range(3):
        client.post("/calls/demo/step")
    before = client.get("/calls/demo/state").json()
    p_before = before["posterior"]["room_probs"][PL_RESTROOM]

    after = client.post(f"/rooms/{PL_RESTROOM}/clear").json()
    post = after["posterior"]
    assert after["cleared_room_ids"] == [PL_RESTROOM]
    assert _total(post) == pytest.approx(1.0, abs=1e-9)

    # Koopman: p_r <- p_r (1 - 0.9), then renormalise.
    expected = 0.1 * p_before / (1.0 - 0.9 * p_before)
    assert post["room_probs"][PL_RESTROOM] == pytest.approx(expected, rel=1e-6)
    # The searched room keeps its residual mass: the planner may send the crew back for a
    # second look (it is standing there), but only as a re-search, never as a first look.
    first = after["plan"]["teams"][0][0]
    assert first["room_id"] != PL_RESTROOM or first["kind"] == "re_search"
    assert all(st["kind"] == "re_search" for st in after["plan"]["teams"][0] if st["room_id"] == PL_RESTROOM)
    assert after["plan"]["p_out"] == pytest.approx(post["p_not_in_model"], abs=1e-6)

    # Clearing the same room twice is idempotent; unknown rooms 404.
    again = client.post(f"/rooms/{PL_RESTROOM}/clear").json()
    assert again["posterior"]["room_probs"][PL_RESTROOM] == pytest.approx(expected, rel=1e-6)
    assert client.post("/rooms/nope/clear").status_code == 404

    reset = client.post("/calls/demo/reset").json()
    assert reset["turn"] == 0 and reset["cleared_room_ids"] == []


def test_two_teams_cover_each_room_once(client: TestClient) -> None:
    for _ in range(3):
        client.post("/calls/demo/step")
    s = client.post("/calls/demo/teams", params={"n": 2}).json()
    teams = s["plan"]["teams"]
    assert len(teams) == 2
    ids = [stop["room_id"] for team in teams for stop in team]
    assert len(ids) == len(set(ids))
    one = client.get("/plan", params={"teams": 1}).json()
    assert s["plan"]["expected_time_to_find_s"] <= one["expected_time_to_find_s"] + 1e-6

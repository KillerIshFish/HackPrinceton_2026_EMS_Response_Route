"""Step I1: the three scripted scenarios driven through the HTTP API, offline (rules parser).

No test here touches the network: the parser mode is "rules" and the model provider is unset.
Call events go to a temporary call log (tests/conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.session as session_mod
from api.server import app
from api.session import CallSession, DemoState
from core.schema import Fact, TranscriptTurn
from ingest.matterport import load_building
from parser.rule_facts import read_facts, room_named
from sim.generate_calls import (
    CARDIAC_TRUE_ROOM_ID,
    FIRE_TRUE_ROOM_ID,
    THREAT_TRUE_ROOM_ID,
    scenario_definitions,
)

SCENARIO_IDS = ("cardiac_arrest_office", "fire_smoke_office", "active_threat_office")
FORBIDDEN = ("safe", "empty", "unsurvivable")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    yield c
    c.post("/calls/demo/building", params={"id": "ptc-100"})


def _ok(resp) -> dict:
    assert resp.status_code == 200, resp.text
    body = resp.json()
    DemoState.model_validate(body)
    return body


def _run(c: TestClient, scenario_id: str) -> dict:
    """Start a scenario and reveal every scripted turn."""
    s = _ok(c.post("/calls/demo/scenario", params={"id": scenario_id}))
    assert s["scenario_id"] == scenario_id and s["turn"] == 0
    while not s["done"]:
        s = _ok(c.post("/calls/demo/step"))
    return s


def _stops(s: dict) -> list[dict]:
    return [st for team in s["plan"]["teams"] for st in team]


def _confirm(c: TestClient, incident_type: str) -> dict:
    return _ok(c.post("/calls/demo/situation", json={"incident_type": incident_type}))


# ---------------------------------------------------------------------------
# Scenario list and scripts
# ---------------------------------------------------------------------------


def test_scenarios_are_listed_with_true_rooms(client: TestClient) -> None:
    s = _ok(client.get("/calls/demo/state"))
    by_id = {sc["id"]: sc for sc in s["scenarios"]}
    building = load_building("ptc-100")
    rooms = {r.id for r in building.rooms}
    for sid in SCENARIO_IDS:
        sc = by_id[sid]
        assert sc["building_id"] == "ptc-100" and sc["true_room_id"] in rooms
        assert 6 <= sc["turns"] <= 8 and sc["description"]
    assert {by_id[i]["incident_type"] for i in SCENARIO_IDS} == {"cardiac_arrest", "fire_smoke", "active_threat"}
    # the older scripted calls are still offered
    assert {"demo", "elevator_stuck", "hiding_no_windows", "lobby_collapse"} <= set(by_id)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_runs_offline_and_starts_unset(client: TestClient, scenario_id: str) -> None:
    first = _ok(client.post("/calls/demo/scenario", params={"id": scenario_id}))
    assert first["situation"]["incident_type"] == "unknown" and not first["situation"]["incident_confirmed"]
    s = _run(client, scenario_id)
    callers = [t for t in s["transcript"] if t["speaker"] == "caller"]
    assert callers and all(t["parser"] == "rules" for t in callers)
    assert {t["speaker"] for t in s["transcript"]} == {"system", "caller", "dispatcher"}
    assert len(s["transcript"]) == s["total_turns"]
    for f in s["facts"]:
        assert f["verified"] is False and f["source"] == "rules"
    for b in s["policy"]["banners"]:  # the console's own wording (reported speech is quoted)
        low = b["text"].lower()
        if not low.startswith("caller reported:"):
            assert not any(w in low.split() for w in FORBIDDEN), b["text"]


# ---------------------------------------------------------------------------
# Active threat
# ---------------------------------------------------------------------------


def test_active_threat_has_no_route_and_puts_facts_on_the_board(client: TestClient,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    s = _run(client, "active_threat_office")
    # the interpreter only suggests; an unconfirmed active-threat suggestion already withholds the route
    assert s["situation"]["suggested_type"] == "active_threat"
    assert s["policy"]["route_enabled"] is False
    assert all(team == [] for team in s["plan"]["teams"])
    assert s["approach"]["entrance_id"] == "" and s["approach"]["staging_id"] == ""
    assert s["questions"] == [] and s["next_question"] is None

    # two callers, distinguishable in the transcript
    titles = {t["title"] for t in s["transcript"] if t["speaker"] == "caller"}
    assert {"Caller 1", "Caller 2"} <= titles

    # route planning is never computed while the route is off (not computed-then-hidden)
    def boom(*a, **k):
        raise AssertionError("route computed while route_enabled is false")

    with monkeypatch.context() as m:
        m.setattr(session_mod, "plan_search", boom)
        m.setattr(session_mod, "recommend_approach", boom)
        s = _confirm(client, "active_threat")
    assert s["policy"]["show_incident_board"] is True and s["policy"]["route_enabled"] is False
    assert all(team == [] for team in s["plan"]["teams"]) and s["plan"]["entrance_id"] == ""
    ap = s["approach"]
    assert ap["entrance_id"] == "" and ap["staging_id"] == "" and ap["stairwell_id"] is None
    assert ap["stairs_s_to_target_floor"] is None and ap["elevator_s_to_target_floor"] is None
    assert ap["entrance_warnings"] == {} and ap["warnings"] == []
    assert s["next_question"] is None and s["policy"]["stake"] is None
    assert s["policy"]["protocol_prompts"] == []
    assert all(h["expected_time_to_find_s"] == 0.0 for h in s["history"])

    facts = s["facts"]
    assert {"threat", "people", "patient"} <= {f["category"] for f in facts}
    by_turn = {t["turn"]: t["title"] for t in s["transcript"]}
    assert any(by_turn[f["turn"]] == "Caller 1" and f["category"] == "people" for f in facts)
    assert any(by_turn[f["turn"]] == "Caller 2" and f["category"] == "patient" for f in facts)


def test_pending_active_threat_is_not_replaced_by_a_milder_suggestion(client: TestClient) -> None:
    _run(client, "active_threat_office")
    s = _ok(client.post("/calls/demo/utterance", json={"text": "There's smoke now too, I can smell it"}))
    assert s["situation"]["suggested_type"] == "active_threat" and s["policy"]["route_enabled"] is False
    s = _ok(client.post("/calls/demo/situation", json={"dismiss_suggestion": True}))
    assert s["policy"]["route_enabled"] is True and s["plan"]["teams"][0]


# ---------------------------------------------------------------------------
# Fire and smoke
# ---------------------------------------------------------------------------


def test_fire_offers_no_elevator_and_uses_smoke_rates_once_confirmed(client: TestClient) -> None:
    s = _run(client, "fire_smoke_office")
    assert s["situation"]["suggested_type"] == "fire_smoke"
    # a smoke fact does not flip smoke_reported by itself...
    assert s["situation"]["smoke_reported"] is False
    assert s["policy"]["travel"]["search_advance_mps"] is None
    # ...but a caller hazard report already withdraws the elevator option (ground rule 4)
    assert s["policy"]["elevator_allowed"] is False and s["approach"]["elevator_s_to_target_floor"] is None
    assert any(f["category"] == "hazard" for f in s["facts"])
    clear_air_look = next(st["search_s"] for st in _stops(s) if st["room_id"] == FIRE_TRUE_ROOM_ID)

    s = _confirm(client, "fire_smoke")  # the dispatcher confirms the suggestion
    assert s["situation"]["incident_type"] == "fire_smoke" and s["situation"]["smoke_reported"] is True
    pol = s["policy"]
    assert pol["elevator_allowed"] is False and pol["travel"]["elevator_allowed"] is False
    assert pol["travel"]["search_advance_mps"] and pol["travel"]["notes"]
    assert s["approach"]["elevator_s_to_target_floor"] is None
    assert s["approach"]["stairs_s_to_target_floor"] is not None
    assert any("Smoke reported" in n for n in s["plan"]["notes"])
    smoke_look = next(st["search_s"] for st in _stops(s) if st["room_id"] == FIRE_TRUE_ROOM_ID)
    assert smoke_look > clear_air_look


def test_fire_hazard_classes_follow_the_facts(client: TestClient) -> None:
    s = _run(client, "fire_smoke_office")
    assert all(st["hazard_class"] is None for st in _stops(s))  # not fire_smoke yet: no hazard class
    order = [st["room_id"] for st in _stops(s)]
    s = _confirm(client, "fire_smoke")
    assert s["policy"]["show_hazard_class"] is True
    stops = {st["room_id"]: st for st in _stops(s)}
    assert stops[FIRE_TRUE_ROOM_ID]["hazard_class"] == "closed_door"  # "I'm in my office, 302. I shut the door"
    # The caller named her room, so the plan covers 99 % of the belief in a few stops and
    # office 301 need not be one of them; its class comes from the session's own reading.
    from api import server

    classes = server.get_session().hazard_classes()
    assert classes["f3-office-301"] == "fire_room"  # "I think the fire's in 301"
    assert classes[FIRE_TRUE_ROOM_ID] == "closed_door"
    if "f3-office-301" in stops:
        assert stops["f3-office-301"]["hazard_class"] == "fire_room"
    for rid, st in stops.items():
        floor = rid.split("-")[0]
        if floor == "f3" and rid not in (FIRE_TRUE_ROOM_ID, "f3-office-301"):
            assert st["hazard_class"] == "open_path"
        elif floor != "f3":
            assert st["hazard_class"] is None
    assert order  # the clear-air plan existed before confirmation
    # no fact naming a room -> no hazard class at all
    assert CallSession(load_building("ptc-100")).hazard_classes([]) == {}


def test_hazard_class_does_not_change_the_stop_order(monkeypatch: pytest.MonkeyPatch) -> None:
    building = load_building("ptc-100")
    s = session_mod.start_scenario("fire_smoke_office", building)
    while s.script_pos < len(s.script) - 1:
        s.step()
    from api.session import SituationPatch

    s.patch_situation(SituationPatch(incident_type="fire_smoke"))
    with_classes = s.state().plan
    monkeypatch.setattr(CallSession, "hazard_classes", lambda self, facts=None: {})
    without = s.state().plan
    assert [st.room_id for t in with_classes.teams for st in t] == [st.room_id for t in without.teams for st in t]


# ---------------------------------------------------------------------------
# Cardiac arrest
# ---------------------------------------------------------------------------


def test_cardiac_has_a_quantified_stake_and_both_vertical_options(client: TestClient) -> None:
    s = _run(client, "cardiac_arrest_office")
    assert s["situation"]["suggested_type"] == "cardiac_arrest"
    assert s["situation"]["meet_person"] and "meet you" in s["situation"]["meet_person"]
    assert any(f["premise_key"] == "meet_person" for f in s["facts"])
    s = _confirm(client, "cardiac_arrest")
    stake = s["policy"]["stake"]
    assert stake and stake["quantified"] is True and stake["per_min_low"] > 0 and stake["per_min_high"] > 0
    ap = s["approach"]
    assert ap["entrance_id"] and ap["staging_id"]
    assert ap["stairs_s_to_target_floor"] is not None and ap["elevator_s_to_target_floor"] is not None
    assert ap["meet_person_credit_s"] > 0
    assert s["plan"]["objective"] == "survival"
    assert s["plan"]["teams"][0][0]["room_id"] == CARDIAC_TRUE_ROOM_ID
    # the slang description is not readable by the vocabulary: no location evidence from that turn
    slang = next(t for t in s["transcript"] if "fishbowl" in t["text"])
    assert slang["evidence_ids"] == []


def test_rejecting_evidence_changes_the_posterior(client: TestClient) -> None:
    s = _run(client, "cardiac_arrest_office")
    conference = next(e for e in s["evidence"] if e["clue"] and e["clue"]["value"] == "conference")
    before = s["posterior"]["room_probs"][CARDIAC_TRUE_ROOM_ID]
    r = _ok(client.post(f"/calls/demo/evidence/{conference['id']}/reject"))
    assert r["posterior"]["room_probs"][CARDIAC_TRUE_ROOM_ID] < before
    assert next(e for e in r["evidence"] if e["id"] == conference["id"])["status"] == "rejected"
    back = _ok(client.post(f"/calls/demo/evidence/{conference['id']}/restore"))
    assert back["posterior"]["room_probs"][CARDIAC_TRUE_ROOM_ID] == pytest.approx(before, abs=1e-12)


# ---------------------------------------------------------------------------
# Searched rooms, second looks and undo
# ---------------------------------------------------------------------------


def test_undo_restores_the_previous_state(client: TestClient) -> None:
    s = _run(client, "fire_smoke_office")
    before_post, before_plan = s["posterior"], [st["room_id"] for st in _stops(s)]
    a = _ok(client.post(f"/rooms/{FIRE_TRUE_ROOM_ID}/searched", params={"mode": "rapid"}))
    assert a["posterior"]["room_probs"][FIRE_TRUE_ROOM_ID] < before_post["room_probs"][FIRE_TRUE_ROOM_ID]
    g = _ok(client.post("/calls/demo/gates", json={"closed_room_ids": ["f3-office-303"]}))
    assert "f3-office-303" not in {st["room_id"] for st in _stops(g)}
    _ok(client.post("/calls/demo/undo"))
    u = _ok(client.post("/calls/demo/undo"))
    assert u["searched"] == [] and u["situation"]["closed_room_ids"] == [] and not u["can_undo"]
    assert u["posterior"]["room_probs"] == pytest.approx(before_post["room_probs"], abs=1e-12)
    assert u["posterior"]["p_not_in_model"] == pytest.approx(before_post["p_not_in_model"], abs=1e-12)
    assert [st["room_id"] for st in _stops(u)] == before_plan


def test_searched_room_can_return_as_a_second_look(client: TestClient) -> None:
    s = _run(client, "fire_smoke_office")
    p_out_before = s["plan"]["p_out"]
    s = _ok(client.post(f"/rooms/{FIRE_TRUE_ROOM_ID}/searched", params={"mode": "rapid"}))
    again = [st for st in _stops(s) if st["room_id"] == FIRE_TRUE_ROOM_ID]
    assert again and all(st["kind"] == "re_search" for st in again)
    assert s["plan"]["p_out"] > p_out_before  # the outside-the-model figure rises as rooms are searched
    assert s["plan"]["p_out"] == pytest.approx(s["posterior"]["p_not_in_model"], abs=1e-6)
    # a closed room never returns: only the officer's gate takes it out
    s = _ok(client.post("/calls/demo/gates", json={"closed_room_ids": [FIRE_TRUE_ROOM_ID]}))
    assert FIRE_TRUE_ROOM_ID not in {st["room_id"] for st in _stops(s)}


def test_plan_chances_are_given_the_searches_so_far(client: TestClient) -> None:
    """I2: after a look the plan's p_find / cumulative_success use the same scale as the
    displayed posterior (renormalised after the looks), not the pre-search belief."""
    _run(client, "cardiac_arrest_office")
    _confirm(client, "cardiac_arrest")
    s = _ok(client.post(f"/rooms/{CARDIAC_TRUE_ROOM_ID}/searched", params={"mode": "rapid"}))
    q = 0.9  # rapid detection probability (configs/situations.yaml, unverified)
    for st in _stops(s):
        if st["kind"] == "first":
            assert st["p_find"] == pytest.approx(s["posterior"]["room_probs"][st["room_id"]] * q, rel=1e-4)
    again = next(st for st in _stops(s) if st["room_id"] == CARDIAC_TRUE_ROOM_ID)
    assert again["p_find"] == pytest.approx(s["posterior"]["room_probs"][CARDIAC_TRUE_ROOM_ID] * q, rel=1e-4)
    assert s["plan"]["cumulative_success"] == pytest.approx(sum(st["p_find"] for st in _stops(s)), abs=1e-5)
    assert s["plan"]["cumulative_success"] <= 1.0 - s["plan"]["p_out"] + 1e-9


def test_found_ends_the_plan_and_records_the_time(client: TestClient) -> None:
    _run(client, "cardiac_arrest_office")
    _confirm(client, "cardiac_arrest")
    s = _ok(client.post(f"/rooms/{CARDIAC_TRUE_ROOM_ID}/found"))
    assert s["found_room_id"] == CARDIAC_TRUE_ROOM_ID and s["found_at"]
    assert _stops(s) == [] and s["plan"]["cumulative_success"] is None
    u = _ok(client.post("/calls/demo/undo"))
    assert u["found_room_id"] is None and u["found_at"] is None and _stops(u)


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------


def test_call_events_go_to_the_call_log(client: TestClient, _temp_call_log: Path) -> None:
    _run(client, "cardiac_arrest_office")
    _confirm(client, "cardiac_arrest")
    _ok(client.post(f"/rooms/{CARDIAC_TRUE_ROOM_ID}/searched", params={"mode": "thorough"}))
    _ok(client.post("/calls/demo/timestamp", params={"name": "patient_contact"}))
    _ok(client.post("/calls/demo/gates", json={"closed_entrance_ids": ["ent-rear"]}))
    _ok(client.post(f"/rooms/{CARDIAC_TRUE_ROOM_ID}/found"))
    _ok(client.post("/calls/demo/undo"))
    lines = [json.loads(x) for x in _temp_call_log.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in lines] == ["searched", "timestamp", "gate", "found", "undo"]
    searched = lines[0]
    assert searched["room_id"] == CARDIAC_TRUE_ROOM_ID and searched["mode"] == "thorough"
    assert searched["room_area_m2"] == pytest.approx(63.0, abs=1.0)
    assert searched["incident_type"] == "cardiac_arrest" and searched["scenario_id"] == "cardiac_arrest_office"
    assert searched["t_s"] >= 0.0 and len({r["call_id"] for r in lines}) == 1
    assert lines[1]["name"] == "patient_contact"
    assert lines[2]["closed_entrance_ids"] == ["ent-rear"]
    assert lines[3]["room_id"] == CARDIAC_TRUE_ROOM_ID
    assert lines[4]["undone"] == "found"
    assert all(r["t_s"] <= lines[-1]["t_s"] for r in lines)


def test_call_log_path_argument_wins(tmp_path: Path) -> None:
    path = tmp_path / "elsewhere.jsonl"
    s = CallSession(load_building("ptc-100"), call_log_path=path)
    s.search_room("f1-lobby")
    assert json.loads(path.read_text(encoding="utf-8"))["room_id"] == "f1-lobby"


# ---------------------------------------------------------------------------
# Asked questions
# ---------------------------------------------------------------------------


def test_unit_named_by_the_caller_retires_the_open_unit_question() -> None:
    s = CallSession(load_building("ptc-100"))
    s.say("We're in suite 304")
    _, evidence = s._evidence(s._last)
    assert "unit:304" in s._asked_keys(evidence)


def test_unanswered_floor_question_keeps_the_open_floor_question(client: TestClient) -> None:
    client.post("/calls/demo/building", params={"id": "ptc-100"})
    s = _ok(client.get("/calls/demo/state"))
    assert "open:floor" in {q["id"] for q in s["questions"]}
    s = _ok(client.post("/calls/demo/ask", json={"question": "Are you on the fourth floor?", "id": "floor:4",
                                                 "clue_type": "floor", "value": "4"}))
    s = _ok(client.post("/calls/demo/utterance", json={
        "text": "I'm not sure", "speaker": "caller", "answers": {"clue_type": "floor", "value": "4", "answer": "unsure"}}))
    ids = {q["id"] for q in s["questions"]}
    assert "floor:4" not in ids
    assert (s["next_question"] or {}).get("value") != "4" or (s["next_question"] or {}).get("clue_type") != "floor"
    assert "open:floor" in ids


# ---------------------------------------------------------------------------
# Rule facts
# ---------------------------------------------------------------------------


def _turn(text: str) -> list[TranscriptTurn]:
    return [TranscriptTurn(turn=0, title="Caller", speaker="caller", text=text)]


def test_rule_facts_read_reports_and_skip_denials() -> None:
    b = load_building("ptc-100")
    facts, sug, _ = read_facts(b, _turn("There's no smoke, I don't smell anything burning."), 0)
    assert sug is None and not [f for f in facts if f.category == "hazard"]
    facts, sug, span = read_facts(b, _turn("He's not breathing. Dave will meet you at the front."), 0)
    assert sug == "cardiac_arrest" and "not breathing" in span
    assert {(f.category, f.premise_key) for f in facts} == {("patient", None), ("people", "meet_person")}
    assert all(isinstance(f, Fact) and f.verified is False for f in facts)
    facts, sug, _ = read_facts(b, _turn("I heard shots. There's smoke too."), 0)
    assert sug == "active_threat"  # a threat report wins over everything
    facts, _, _ = read_facts(b, _turn("I'm in 302. I shut the door."), 0)
    assert [(f.category, f.room_id) for f in facts] == [("access", "f3-office-302")]
    assert room_named(b, "the fire's in the Records Room") == "f4-storage-records"
    assert room_named(b, "fix 9.4 m above ground") is None
    nothing, sug, _ = read_facts(b, [TranscriptTurn(turn=0, title="Dispatcher", speaker="dispatcher",
                                                    text="Is there smoke?")], 0)
    assert nothing == [] and sug is None


def test_true_rooms_are_scenario_constants() -> None:
    by_id = {d["id"]: d["true_room_id"] for d in scenario_definitions()}
    assert by_id == {"cardiac_arrest_office": CARDIAC_TRUE_ROOM_ID, "fire_smoke_office": FIRE_TRUE_ROOM_ID,
                     "active_threat_office": THREAT_TRUE_ROOM_ID}


def test_grounded_agent_evidence_is_capped_in_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session fuses with core.fusion.evidence_confidence: grounded agent items count at most 0.85."""
    from core.fusion import posterior_probs
    from core.schema import Evidence, Interpretation, RoomSelector

    building = load_building("ptc-100")
    item = Evidence(id="t1-e0", kind="grounded", selector=RoomSelector(room_ids=[CARDIAC_TRUE_ROOM_ID]),
                    confidence=0.99, turn=1, source="agent", span="the fishbowl")
    monkeypatch.setattr(session_mod, "interpret_turn",
                        lambda *a, **k: Interpretation(evidence=[item], source="agent"))
    s = CallSession(building, script=[{"title": "c", "speaker": "system", "text": "connected"}])
    s.say("We're in the fishbowl")
    got = s.state().posterior
    capped, p0 = posterior_probs(building, hour=14, clues=[item])
    assert got.room_probs[CARDIAC_TRUE_ROOM_ID] == pytest.approx(capped[CARDIAC_TRUE_ROOM_ID], abs=1e-12)
    uncapped, _ = posterior_probs(building, hour=14, clues=[item.model_copy(update={"source": "rules"})])
    assert got.room_probs[CARDIAC_TRUE_ROOM_ID] < uncapped[CARDIAC_TRUE_ROOM_ID]

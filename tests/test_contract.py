"""Step 0 contract (docs/BUILD_PLAN.md section 4): new endpoints, unchanged numbers, valid fixtures.

No test here touches the network: the agent parser mode is only checked for being accepted.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from api.session import DemoState
from core.evidence import EvidenceMatch, evidence_match
from core.fusion import posterior_probs
from core.schema import (
    INCIDENT_TYPES,
    Evidence,
    Gates,
    Interpretation,
    PhoneFix,
    Policy,
    RoomSelector,
    Situation,
    TranscriptTurn,
    TravelProfile,
)
from ingest.matterport import load_building
from parser.interpret import interpret_turn
from planning.search_order import plan_search, residual_probs
from situation.config import load_situations
from situation.policy import policy_for
from situation.survival import stake_for, survival_for
from situation.travel import profile_for

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "web" / "src" / "fixtures"
PL_RESTROOM = "f4-restroom-north"


@pytest.fixture()
def client() -> TestClient:
    """A fresh call on the synthetic building with the offline rule parser."""
    c = TestClient(app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    c.post("/calls/demo/building", params={"id": "ptc-100"})
    return c


def _ok(resp) -> dict:
    assert resp.status_code == 200, resp.text
    body = resp.json()
    DemoState.model_validate(body)
    return body


def _to_turn(c: TestClient, n: int) -> dict:
    s = c.get("/calls/demo/state").json()
    for _ in range(n):
        s = c.post("/calls/demo/step").json()
    return s


# ---------------------------------------------------------------------------
# Numbers are unchanged
# ---------------------------------------------------------------------------

# Recorded from the code before Step 0 (commit 5122233) through the same API calls.
GOLDEN = {
    0: (0.05, [("f2-open-office-n1", 0.037002038026235334)], 5.812304107928523),
    1: (0.05, [("f4-open-office-s", 0.09426398314779966), ("f4-office-401", 0.07097116032794946)], 4.886933311870008),
    2: (0.04446980691113667, [(PL_RESTROOM, 0.20277847596424176), ("f4-restroom-core", 0.20230666276861606)],
        4.1665549439398735),
    3: (0.04444350701161753, [(PL_RESTROOM, 0.4892595325778796), ("f4-boardroom", 0.0661123235740738)],
        3.0630824458540555),
}


def test_scripted_demo_posterior_matches_pre_step0_values(client: TestClient) -> None:
    s = client.get("/calls/demo/state").json()
    for turn in range(4):
        if turn:
            s = client.post("/calls/demo/step").json()
        p_nim, tops, entropy = GOLDEN[turn]
        post = s["posterior"]
        assert post["p_not_in_model"] == pytest.approx(p_nim, abs=1e-12)
        for rid, p in tops:
            assert post["room_probs"][rid] == pytest.approx(p, abs=1e-12)
        assert post["entropy_bits"] == pytest.approx(entropy, abs=1e-9)
        assert s["prior_hour"] == 14


def test_session_matches_the_pipeline(client: TestClient) -> None:
    """Posterior = posterior_probs(prior, fix, evidence) + one Koopman update per look (residual_probs);
    plan = plan_search on the evidence-only belief with every look in ``searched`` (step I1 wiring)."""
    building = load_building("ptc-100")
    s = _to_turn(client, 3)
    s = client.post(f"/rooms/{PL_RESTROOM}/searched", params={"mode": "rapid"}).json()
    s = client.post(f"/rooms/{PL_RESTROOM}/searched", params={"mode": "thorough"}).json()
    evidence = [Evidence.model_validate(e) for e in s["evidence"]]
    fix = PhoneFix.model_validate(s["phone_fix"]) if s["phone_fix"] else None
    belief, p0 = posterior_probs(building, hour=14, fix=fix, clues=evidence)
    looks = [(PL_RESTROOM, "rapid"), (PL_RESTROOM, "thorough")]
    probs, p_nim = residual_probs(belief, p0, looks)
    assert s["posterior"]["p_not_in_model"] == pytest.approx(p_nim, abs=1e-12)
    for rid, p in probs.items():
        assert s["posterior"]["room_probs"][rid] == pytest.approx(p, abs=1e-12)
    pol = Policy.model_validate(s["policy"])
    plan = plan_search(building, belief, p0, n_teams=1, start_node=PL_RESTROOM, profile=pol.travel,
                       searched=looks, gates=Gates())
    got = s["plan"]
    assert [st["room_id"] for st in got["teams"][0]] == [st.room_id for st in plan.teams[0]]
    assert got["expected_time_to_find_s"] == pytest.approx(plan.expected_time_to_find_s, abs=1e-9)
    assert got["entrance_id"] == plan.entrance_id
    # the planner's outside-the-model figure is the displayed one
    assert got["p_out"] == pytest.approx(s["posterior"]["p_not_in_model"], abs=1e-6)


# ---------------------------------------------------------------------------
# New endpoints answer 200 and return a DemoState
# ---------------------------------------------------------------------------


def test_state_has_contract_fields(client: TestClient) -> None:
    s = _ok(client.get("/calls/demo/state"))
    for key in ("situation", "policy", "evidence", "facts", "questions", "searched", "found_room_id",
                "scenarios", "agent_trace"):
        assert key in s
    assert s["situation"]["incident_type"] == "unknown" and s["situation"]["call_started_at"]
    assert s["policy"]["route_enabled"] is True
    assert len(s["questions"]) <= 3
    stop = s["plan"]["teams"][0][0]
    assert stop["kind"] == "first" and stop["mode"] == "rapid"
    assert s["plan"]["solver"] in ("exact", "heuristic")
    assert {sc["id"] for sc in s["scenarios"]} >= {"demo"}


def test_situation_patch(client: TestClient) -> None:
    s = _ok(client.post("/calls/demo/situation", json={"incident_type": "cardiac_arrest", "smoke_reported": True,
                                                       "load": "heavy", "witnessed": True, "n_teams": 2}))
    sit = s["situation"]
    assert sit["incident_type"] == "cardiac_arrest" and sit["incident_confirmed"] is True
    assert sit["smoke_reported"] is True and sit["load"] == "heavy" and sit["witnessed"] is True
    assert s["n_teams"] == 2
    s = _ok(client.post("/calls/demo/situation", json={"witnessed": None, "dismiss_suggestion": True}))
    assert s["situation"]["witnessed"] is None
    assert client.post("/calls/demo/situation", json={"incident_type": "volcano"}).status_code == 422
    assert client.post("/calls/demo/situation", json={"nope": 1}).status_code == 422


def test_utterance_speaker_and_answers(client: TestClient) -> None:
    s = _ok(client.post("/calls/demo/utterance", json={"text": "Are you near a window?", "speaker": "dispatcher"}))
    assert s["transcript"][-1]["speaker"] == "dispatcher" and s["transcript"][-1]["parser"] is None
    s = _ok(client.post("/calls/demo/utterance", json={
        "text": "Yes", "speaker": "caller", "answers": {"clue_type": "landmark", "value": "window", "answer": "yes"}}))
    last = s["transcript"][-1]
    assert last["speaker"] == "caller" and last["answer"]["answer"] == "yes" and last["parser"] == "rules"
    # older clients send text only
    s = _ok(client.post("/calls/demo/utterance", json={"text": "I'm in a bathroom"}))
    assert s["transcript"][-1]["speaker"] == "caller"
    assert client.post("/calls/demo/utterance", json={"text": "x", "speaker": "system"}).status_code == 422


def test_ask_logs_question_and_stops_suggesting_it(client: TestClient) -> None:
    s = client.get("/calls/demo/state").json()
    q = s["questions"][0]
    s = _ok(client.post("/calls/demo/ask", json=q))  # a Question can be posted as is
    last = s["transcript"][-1]
    assert last["speaker"] == "dispatcher" and last["question"]["id"] == q["id"]
    assert q["id"] not in {x["id"] for x in s["questions"]}
    nq = s["next_question"]
    assert nq is None or (nq["clue_type"], nq["value"]) != (q["clue_type"], q["value"])


def test_evidence_reject_and_restore(client: TestClient) -> None:
    before = _to_turn(client, 1)["posterior"]
    s = _to_turn(client, 1)  # turn 2: "I'm locked in a bathroom"
    ev = s["evidence"]
    assert [e["id"] for e in ev] == ["t2-e0"] and ev[0]["kind"] == "vocab" and ev[0]["status"] == "active"
    assert ev[0]["matched_room_ids"] and ev[0]["unmatched"] is False
    with_clue = s["posterior"]
    rejected = _ok(client.post("/calls/demo/evidence/t2-e0/reject"))
    assert rejected["evidence"][0]["status"] == "rejected" and rejected["clues"] == []
    assert rejected["posterior"]["room_probs"][PL_RESTROOM] == pytest.approx(before["room_probs"][PL_RESTROOM], abs=1e-12)
    assert rejected["posterior"]["room_probs"][PL_RESTROOM] < with_clue["room_probs"][PL_RESTROOM]
    restored = _ok(client.post("/calls/demo/evidence/t2-e0/restore"))
    assert restored["evidence"][0]["status"] == "active"
    assert restored["posterior"]["room_probs"][PL_RESTROOM] == pytest.approx(with_clue["room_probs"][PL_RESTROOM], abs=1e-12)
    assert client.post("/calls/demo/evidence/nope/reject").status_code == 404
    assert client.post("/calls/demo/evidence/nope/restore").status_code == 404


def test_searched_found_gates_undo(client: TestClient) -> None:
    _to_turn(client, 3)
    s = _ok(client.post(f"/rooms/{PL_RESTROOM}/searched", params={"mode": "thorough"}))
    assert s["searched"][-1]["room_id"] == PL_RESTROOM and s["searched"][-1]["mode"] == "thorough"
    assert s["searched"][-1]["at"] and s["cleared_room_ids"] == [PL_RESTROOM] and s["can_undo"]
    s = _ok(client.post("/rooms/f4-office-405/found"))
    assert s["found_room_id"] == "f4-office-405"
    s = _ok(client.post("/calls/demo/gates", json={"closed_room_ids": ["f4-office-406"], "closed_entrance_ids": ["ent-rear"]}))
    assert s["situation"]["closed_room_ids"] == ["f4-office-406"]
    assert s["situation"]["closed_entrance_ids"] == ["ent-rear"]
    assert "f4-office-406" not in {st["room_id"] for t in s["plan"]["teams"] for st in t}
    s = _ok(client.post("/calls/demo/undo"))
    assert s["situation"]["closed_room_ids"] == [] and s["found_room_id"] == "f4-office-405"
    s = _ok(client.post("/calls/demo/undo"))
    assert s["found_room_id"] is None and len(s["searched"]) == 1
    s = _ok(client.post("/calls/demo/undo"))
    assert s["searched"] == [] and not s["can_undo"]
    _ok(client.post("/calls/demo/undo"))  # nothing left: still 200
    assert client.post("/rooms/nope/searched").status_code == 404
    assert client.post("/rooms/nope/found").status_code == 404
    assert client.post(f"/rooms/{PL_RESTROOM}/searched", params={"mode": "slow"}).status_code == 422
    assert client.post("/calls/demo/gates", json={"closed_room_ids": ["nope"]}).status_code == 404
    assert client.post("/calls/demo/gates", json={"closed_entrance_ids": ["nope"]}).status_code == 404


def test_timestamp(client: TestClient) -> None:
    for name in ("wheels_stopped", "building_entry", "floor_arrival", "patient_contact"):
        s = _ok(client.post("/calls/demo/timestamp", params={"name": name}))
        assert s["situation"]["timestamps"][name]
    assert client.post("/calls/demo/timestamp", params={"name": "lunch"}).status_code == 422


def test_scenario(client: TestClient) -> None:
    ids = [sc["id"] for sc in client.get("/calls/demo/state").json()["scenarios"]]
    target = next(i for i in ids if i != "demo")
    s = _ok(client.post("/calls/demo/scenario", params={"id": target}))
    assert s["scenario_id"] == target and s["turn"] == 0
    assert client.post("/calls/demo/scenario", params={"id": "nope"}).status_code == 404
    client.post("/calls/demo/building", params={"id": "ptc-100"})


def test_parser_accepts_agent_mode_without_network(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setattr(server, "llm_available", lambda: False)
    assert client.post("/calls/demo/parser", params={"mode": "agent"}).status_code == 409  # accepted, needs a key
    assert client.post("/calls/demo/parser", params={"mode": "bogus"}).status_code == 422


# ---------------------------------------------------------------------------
# Stub modules have the contract signatures
# ---------------------------------------------------------------------------


def test_stub_signatures() -> None:
    assert list(inspect.signature(interpret_turn).parameters)[:4] == ["building", "transcript", "index", "mode"]
    assert list(inspect.signature(policy_for).parameters)[:3] == ["building", "situation", "facts"]
    assert list(inspect.signature(stake_for).parameters) == ["situation", "elapsed_s"]
    assert list(inspect.signature(survival_for).parameters) == ["situation"]
    assert list(inspect.signature(profile_for).parameters) == ["situation"]
    params = inspect.signature(plan_search).parameters
    for name in ("profile", "survival", "searched", "gates", "exclude"):
        assert name in params


def test_stub_outputs_have_contract_types() -> None:
    building = load_building("ptc-100")
    sit = Situation()
    profile = profile_for(sit)
    assert isinstance(profile, TravelProfile)
    assert profile.walk_mps == pytest.approx(1.2) and profile.stair_s_per_floor == pytest.approx(15.0)
    pol = policy_for(building, sit, [])
    assert isinstance(pol, Policy) and pol.route_enabled is True
    turns = [TranscriptTurn(turn=0, title="t", speaker="caller", text="I'm locked in a bathroom")]
    it = interpret_turn(building, turns, 0, "rules")
    assert isinstance(it, Interpretation)
    assert [(e.id, e.kind, e.clue.value if e.clue else None) for e in it.evidence] == [("t0-e0", "vocab", "restroom")]
    ev = Evidence(id="g", kind="grounded", selector=RoomSelector(room_ids=[PL_RESTROOM]), confidence=0.8)
    m = evidence_match(building, ev)
    assert isinstance(m, EvidenceMatch) and not m.unmatched
    assert m.match[PL_RESTROOM] == 1.0 and m.match["f1-lobby"] == 0.0 and m.matched_room_ids == [PL_RESTROOM]
    none = evidence_match(building, Evidence(id="n", kind="grounded", selector=RoomSelector(room_ids=["nope"]), confidence=0.8))
    assert none.unmatched and set(none.match.values()) == {1.0}


def test_situations_config_lists_every_incident_type() -> None:
    cfg = load_situations()
    assert set(cfg["incident_types"]) == set(INCIDENT_TYPES)
    for section in ("travel", "survival", "search", "policy"):
        assert section in cfg


# ---------------------------------------------------------------------------
# Web fixtures satisfy DemoState and show what each mode needs
# ---------------------------------------------------------------------------

FIXTURE_NAMES = ("unknown", "cardiac_arrest", "fire_smoke", "active_threat")


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_is_a_valid_demo_state(name: str) -> None:
    data = _fixture(name)
    state = DemoState.model_validate(data)
    building = load_building(state.building_id)
    room_ids = {r.id for r in building.rooms}
    assert set(state.posterior.room_probs) == room_ids
    for e in state.evidence:
        assert set(e.matched_room_ids) <= room_ids
    for f in state.facts:
        assert f.verified is False and (f.room_id is None or f.room_id in room_ids)
    for stop in (st for team in state.plan.teams for st in team):
        assert stop.room_id in room_ids
    assert state.situation.incident_type == name
    assert len(state.questions) <= 3


def test_fixtures_cover_the_ui_cases() -> None:
    unk, ca, fire, at = (DemoState.model_validate(_fixture(n)) for n in FIXTURE_NAMES)
    assert unk.situation.suggested_type and not unk.situation.incident_confirmed
    assert ca.policy.stake and ca.policy.stake.quantified and (ca.policy.stake.per_min_low or 0) > 0
    assert unk.policy.stake and not unk.policy.stake.quantified
    statuses = {e.status for f in (ca, fire) for e in f.evidence}
    assert {"active", "rejected", "superseded"} <= statuses
    assert {e.kind for e in ca.evidence} == {"vocab", "grounded"}
    assert {"yes_no", "open"} <= {q.kind for f in (unk, ca) for q in f.questions}
    assert {st.kind for st in ca.plan.teams[0]} == {"first", "re_search"}
    assert ca.agent_trace and any(not t.kept and t.reason for t in ca.agent_trace)
    assert len({f.category for fx in (ca, fire, at) for f in fx.facts}) >= 5
    assert fire.policy.travel.notes and fire.policy.elevator_allowed is False and fire.policy.show_hazard_class
    assert {st.hazard_class for st in fire.plan.teams[0]} >= {"fire_room", "closed_door", "open_path"}
    for f in (unk, ca, fire, at):
        assert f.policy.banners
    assert ca.searched and fire.searched and fire.situation.closed_room_ids
    for f in (unk, ca, fire):
        assert f.plan.p_out is not None and f.plan.cumulative_success is not None
    # active threat: no route, no entrance, no staging, no questions
    assert at.policy.route_enabled is False and at.policy.show_incident_board is True
    assert all(team == [] for team in at.plan.teams)
    assert at.plan.entrance_id == "" and at.approach["entrance_id"] == "" and at.approach["staging_id"] == ""
    assert at.questions == [] and at.next_question is None

"""Step C1: dialog-aware evidence and grounded matching (docs/BUILD_PLAN.md section 6).

Rules mode only; nothing here touches the network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import core.evidence as core_evidence
from core.evidence import evidence_match, normalize_unit, selector_hits
from core.fusion import GROUNDED_AGENT_MAX_CONFIDENCE, posterior_probs
from core.schema import Answer, Clue, Evidence, Question, RoomSelector, TranscriptTurn, WithinM
from ingest.matterport import load_building
from parser.dialog import asked_keys, classify_reply, find_units
from parser.interpret import interpret_turn, wrap_clue

WINDOW_Q = "Is there a window in the room you're in?"

# Fake sidecar metadata for the synthetic building (it has none of its own).
FAKE_META = {
    "f3-office-303": {"suite": "304", "tenant": "Acme Widgets"},
    "f4-office-401": {"suite": "401", "tenant": "Northside Law"},
    "f2-office-201": {"suite": "201", "tenant": "Dr. Patel Family Dental"},
}


@pytest.fixture(scope="module")
def building():
    return load_building("ptc-100")


@pytest.fixture()
def fake_meta(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(core_evidence, "room_meta", lambda b: FAKE_META)
    return FAKE_META


def _turns(*rows) -> list[TranscriptTurn]:
    out = []
    for k, row in enumerate(rows):
        speaker, text, *extra = row
        out.append(TranscriptTurn(turn=k, title=f"t{k}", speaker=speaker, text=text, **(extra[0] if extra else {})))
    return out


def _window_question() -> Question:
    return Question(id="landmark:window", question=WINDOW_Q, clue_type="landmark", value="window")


def _post(building, items):
    return posterior_probs(building, hour=14, clues=items)


def _resolved(interps) -> list[Evidence]:
    """Evidence of several turns with ``supersedes`` applied (what the session does)."""
    sup = {i for it in interps for i in it.supersedes}
    return [e.model_copy(update={"status": "superseded"}) if e.id in sup else e for it in interps for e in it.evidence]


# ---------------------------------------------------------------------------
# Answer resolution
# ---------------------------------------------------------------------------


def test_yes_after_window_question_is_a_positive_window_clue(building) -> None:
    t = _turns(("dispatcher", WINDOW_Q, {"question": _window_question()}), ("caller", "Yeah, there is one"))
    it = interpret_turn(building, t, 1, "rules")
    windows = [e for e in it.evidence if e.clue and e.clue.value == "window"]
    assert len(windows) == 1
    e = windows[0]
    assert e.clue.type == "landmark" and e.clue.polarity is True
    assert e.confidence == pytest.approx(0.85) and e.span == "Yeah" and e.source == "rules" and e.id == "t1-e0"
    base, _ = _post(building, [])
    after, _ = _post(building, it.evidence)
    windowed = [r.id for r in building.rooms if r.has_windows]
    assert sum(after[r] for r in windowed) > sum(base[r] for r in windowed)


def test_no_after_window_question_is_a_negative_window_clue(building) -> None:
    t = _turns(("dispatcher", WINDOW_Q, {"question": _window_question()}), ("caller", "No, I don't think so"))
    it = interpret_turn(building, t, 1, "rules")
    assert len(it.evidence) == 1
    c = it.evidence[0].clue
    assert (c.type, c.value, c.polarity) == ("negative", "landmarks:window", False)
    assert it.evidence[0].confidence == pytest.approx(0.85) and it.evidence[0].span == "No"
    base, _ = _post(building, [])
    after, _ = _post(building, it.evidence)
    windowed = [r.id for r in building.rooms if r.has_windows]
    assert sum(after[r] for r in windowed) < sum(base[r] for r in windowed)


def test_yes_to_a_free_text_question_and_through_system_turns(building) -> None:
    t = _turns(("dispatcher", "Are you on the fourth floor?"), ("system", "Device fix received."), ("caller", "uh huh"))
    it = interpret_turn(building, t, 2, "rules")
    assert [(e.clue.type, e.clue.value, e.clue.polarity) for e in it.evidence] == [("floor", "4", True)]
    assert it.evidence[0].confidence == pytest.approx(0.75)  # question read from free text


def test_unsure_emits_nothing_but_counts_as_asked(building) -> None:
    t = _turns(("dispatcher", WINDOW_Q, {"question": _window_question()}), ("caller", "I don't know, it's dark"))
    assert interpret_turn(building, t, 1, "rules").evidence == []
    assert "landmark:window" in asked_keys(t)
    button = _turns(("caller", "Not sure", {"answer": Answer(clue_type="view", value="street", answer="unsure")}))
    assert interpret_turn(building, button, 0, "rules").evidence == []
    assert "view:street" in asked_keys(button)


def test_answer_buttons_give_dispatcher_evidence(building) -> None:
    t = _turns(("dispatcher", WINDOW_Q, {"question": _window_question()}),
               ("caller", "Yes", {"answer": Answer(clue_type="landmark", value="window", answer="yes")}))
    it = interpret_turn(building, t, 1, "rules")
    assert len(it.evidence) == 1
    e = it.evidence[0]
    assert e.source == "dispatcher" and e.confidence == pytest.approx(0.9)
    assert (e.clue.type, e.clue.value, e.clue.polarity) == ("landmark", "window", True)
    no = _turns(("caller", "no", {"answer": Answer(clue_type="room_type", value="restroom", answer="no")}))
    c = interpret_turn(building, no, 0, "rules").evidence[0].clue
    assert (c.type, c.value, c.polarity) == ("negative", "room_types:restroom", False)


def test_reply_forms() -> None:
    assert classify_reply("yep")[0] == "yes" and classify_reply("Um, yes")[0] == "yes"
    assert classify_reply("nope")[0] == "no" and classify_reply("I don't think so")[0] == "no"
    assert classify_reply("not sure")[0] == "unsure" and classify_reply("no idea")[0] == "unsure"
    assert classify_reply("No wait, the fifth floor") is None
    assert classify_reply("Nothing like that") is None


def test_no_question_means_yes_is_ignored(building) -> None:
    t = _turns(("dispatcher", "Stay on the line with me."), ("caller", "yes"))
    assert interpret_turn(building, t, 1, "rules").evidence == []


# ---------------------------------------------------------------------------
# Unit numbers and grounded matching
# ---------------------------------------------------------------------------


def test_unit_mentions() -> None:
    assert [m.unit for m in find_units("I'm in 304")] == ["304"]
    assert [m.unit for m in find_units("we're in unit 12B")] == ["12B"]
    assert [m.unit for m in find_units("suite 860, the big one")] == ["860"]
    assert [m.unit for m in find_units("apartment 4")] == ["4"]
    assert find_units("I'm on the 3rd floor") == [] and find_units("I'm in 2 minutes") == []
    assert normalize_unit("Suite #0304") == "304" and normalize_unit("12b") == normalize_unit("12B")


def test_im_in_304_raises_that_suite_and_no_other_floor(building, fake_meta) -> None:
    t = _turns(("caller", "I'm in 304"))
    it = interpret_turn(building, t, 0, "rules")
    assert len(it.evidence) == 1  # the unit is on record: no floor fallback
    e = it.evidence[0]
    assert e.kind == "grounded" and e.selector.unit == "304" and e.span == "I'm in 304"
    m = evidence_match(building, e)
    assert not m.unmatched and m.matched_room_ids == ["f3-office-303"]
    base, _ = _post(building, [])
    after, _ = _post(building, it.evidence)
    assert after["f3-office-303"] > base["f3-office-303"]
    for r in building.rooms:
        if r.floor != 3:
            assert after[r.id] <= base[r.id] + 1e-15, r.id


def test_unit_without_metadata_falls_back_to_the_floor(building) -> None:
    it = interpret_turn(building, _turns(("caller", "I'm in 304")), 0, "rules")
    kinds = [(e.kind, e.clue.value if e.clue else e.selector.unit, e.confidence) for e in it.evidence]
    assert kinds == [("grounded", "304", 0.85), ("vocab", "3", 0.5)]
    assert evidence_match(building, it.evidence[0]).unmatched  # UI: could not be placed
    # floor 9 does not exist in a four-storey building: no fallback, and the unit alone changes nothing
    it9 = interpret_turn(building, _turns(("caller", "suite 904")), 0, "rules")
    assert [e.kind for e in it9.evidence] == ["grounded"]
    assert _post(building, it9.evidence) == _post(building, [])


def test_bare_number_answers_an_open_unit_question(building, fake_meta) -> None:
    q = Question(id="open:unit", question="What number is on the door?", kind="open", partition="unit")
    it = interpret_turn(building, _turns(("dispatcher", q.question, {"question": q}), ("caller", "401")), 1, "rules")
    assert [e.selector.unit for e in it.evidence] == ["401"]
    assert evidence_match(building, it.evidence[0]).matched_room_ids == ["f4-office-401"]


def test_real_building_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        bundy = load_building("bundy-olympic")
    except Exception:  # pragma: no cover - processed data not present
        pytest.skip("bundy-olympic not built")
    if not core_evidence.has_unit_metadata(bundy):
        pytest.skip("no suite metadata")
    it = interpret_turn(bundy, _turns(("caller", "We're in suite 860")), 0, "rules")
    m = evidence_match(bundy, it.evidence[0])
    assert m.matched_room_ids and all(r.startswith("f8-s860") for r in m.matched_room_ids)


def test_selector_conditions(building, fake_meta) -> None:
    def hits(**kw):
        return set(selector_hits(building, RoomSelector(**kw)))

    floor3 = {r.id for r in building.rooms if r.floor_label == "3"}
    assert hits(floor_labels=["3"]) == floor3
    assert hits(floor_labels=["3"], room_types=["Office"]) == {"f3-suite-vacant", "f3-office-301", "f3-office-302", "f3-office-303"}
    assert hits(name_contains="suite 310") == {"f3-suite-vacant"}
    assert hits(floor_labels=["3"], window_facing="street", near_stairs=True) == {"f3-office-301", "f3-office-303"}
    assert hits(floor_labels=["3"], has_windows=False, near_elevator=True) == {
        "f3-corridor-n-mid", "f3-corridor-s-mid", "f3-restroom-core", "f3-elevator", "f3-mechanical"}
    assert hits(tenant_contains="patel") == {"f2-office-201"}
    assert hits(unit="#401") == {"f4-office-401"}
    near = hits(within_m_of=WithinM(room_id="f3-office-301", radius_m=12.0))
    assert "f3-office-301" in near and "f3-corridor-s-west" in near and "f3-office-302" in near
    assert all(building.rooms[[r.id for r in building.rooms].index(rid)].floor == 3 for rid in near)
    assert "f3-office-303" not in near
    assert hits() == set() and hits(room_ids=["nope"]) == set()


def test_grounded_values_and_negate(building) -> None:
    sel = RoomSelector(floor_labels=["2"], room_types=["office"])
    e = Evidence(id="g", kind="grounded", selector=sel, default_match=0.2, confidence=0.8)
    m = evidence_match(building, e)
    assert set(m.matched_room_ids) == {"f2-office-201", "f2-office-202"}
    assert m.match["f2-office-201"] == 1.0 and m.match["f1-lobby"] == pytest.approx(0.2)
    neg = evidence_match(building, e.model_copy(update={"negate": True, "default_match": 1.0}))
    assert neg.match["f2-office-201"] == 0.0 and neg.match["f1-lobby"] == 1.0


def test_unmatched_selector_changes_nothing(building, fake_meta) -> None:
    e = Evidence(id="u", kind="grounded", selector=RoomSelector(unit="999"), confidence=0.85)
    m = evidence_match(building, e)
    assert m.unmatched and set(m.match.values()) == {1.0} and m.matched_room_ids == []
    assert _post(building, [e]) == _post(building, [])
    empty = Evidence(id="x", kind="grounded", selector=RoomSelector(), confidence=0.85)
    assert evidence_match(building, empty).unmatched


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_vocab_evidence_equals_the_clue_path(building) -> None:
    clues = [Clue(type="room_type", value="restroom", polarity=True, confidence=1.0, span="bathroom"),
             Clue(type="view", value="parking_lot", polarity=True, confidence=0.9, span="parking lot")]
    wrapped = [wrap_clue(c, 2, k, "rules") for k, c in enumerate(clues)]
    assert _post(building, wrapped) == _post(building, clues)


def test_rejected_evidence_no_longer_moves_the_posterior(building) -> None:
    e = wrap_clue(Clue(type="room_type", value="restroom", polarity=True, confidence=1.0, span="bathroom"), 0, 0, "rules")
    assert _post(building, [e]) != _post(building, [])
    assert _post(building, [e.model_copy(update={"status": "rejected"})]) == _post(building, [])
    assert _post(building, [e.model_copy(update={"status": "superseded"})]) == _post(building, [])


def test_grounded_agent_confidence_is_capped(building) -> None:
    sel = RoomSelector(room_ids=["f3-office-301"])
    agent = Evidence(id="a", kind="grounded", selector=sel, confidence=1.0, source="agent")
    capped = agent.model_copy(update={"confidence": GROUNDED_AGENT_MAX_CONFIDENCE})
    rules = agent.model_copy(update={"source": "rules"})
    assert _post(building, [agent]) == _post(building, [capped])
    assert _post(building, [rules]) != _post(building, [agent])


def test_one_wrong_statement_never_zeroes_a_room(building) -> None:
    e = Evidence(id="g", kind="grounded", selector=RoomSelector(room_ids=["f3-office-301"]), confidence=1.0)
    probs, _ = _post(building, [e])
    assert min(probs.values()) > 0.0


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def test_superseding_floor_statement_replaces_the_first(building) -> None:
    t = _turns(("caller", "I'm on the third floor"), ("dispatcher", "Okay."), ("caller", "Sorry, I meant the fourth floor"))
    first = interpret_turn(building, t, 0, "rules")
    second = interpret_turn(building, t, 2, "rules", prior_evidence=first.evidence)
    assert second.supersedes == [first.evidence[0].id] == ["t0-e0"]
    items = _resolved([first, second])
    assert [e.status for e in items] == ["superseded", "active"]
    only_fourth = [e for e in items if e.clue.value == "4"]
    assert _post(building, items) == _post(building, only_fourth)
    probs, _ = _post(building, items)
    assert probs["f4-office-401"] > probs["f3-office-301"]


def test_correction_inside_one_turn(building) -> None:
    it = interpret_turn(building, _turns(("caller", "I'm on the third floor, no wait, the fourth floor")), 0, "rules")
    assert [(e.clue.value, e.status) for e in it.evidence] == [("3", "superseded"), ("4", "active")]
    assert it.supersedes == ["t0-e0"]


def test_correction_only_touches_its_own_slot(building) -> None:
    t = _turns(("caller", "I'm in a bathroom on the third floor"), ("caller", "Actually it's the second floor"))
    first = interpret_turn(building, t, 0, "rules")
    second = interpret_turn(building, t, 1, "rules", prior_evidence=first.evidence)
    floor_ids = [e.id for e in first.evidence if e.clue.type == "floor"]
    assert second.supersedes == floor_ids
    # no cue -> nothing superseded
    plain = interpret_turn(building, _turns(("caller", t[0].text), ("caller", "It's the second floor")), 1, "rules",
                           prior_evidence=first.evidence)
    assert plain.supersedes == []


def test_unit_correction_supersedes_the_floor(building, fake_meta) -> None:
    t = _turns(("caller", "I'm on the second floor"), ("caller", "no wait, I'm in 304"))
    first = interpret_turn(building, t, 0, "rules")
    second = interpret_turn(building, t, 1, "rules", prior_evidence=first.evidence)
    assert second.supersedes == ["t0-e0"]


# ---------------------------------------------------------------------------
# Through the API (the session already calls interpret_turn)
# ---------------------------------------------------------------------------


def test_session_yes_after_asked_question() -> None:
    import api.server as server

    c = TestClient(server.app)
    c.post("/calls/demo/parser", params={"mode": "rules"})
    c.post("/calls/demo/building", params={"id": "ptc-100"})
    c.post("/calls/demo/ask", json={"id": "landmark:window", "question": WINDOW_Q, "clue_type": "landmark",
                                    "value": "window", "kind": "yes_no"})
    s = c.post("/calls/demo/utterance", json={"text": "yes", "speaker": "caller"}).json()
    ev = [e for e in s["evidence"] if e["turn"] == len(s["transcript"]) - 1]
    assert [(e["clue"]["type"], e["clue"]["value"], e["clue"]["polarity"]) for e in ev] == [("landmark", "window", True)]


def test_negated_unit_mention_is_not_read_as_a_location():
    from parser.dialog import find_units

    assert find_units("we're not in suite 304") == []
    assert find_units("we aren't in unit 12B") == []
    assert [m.unit for m in find_units("we aren't in unit 12B, we're in suite 860")] == ["860"]
    assert [m.unit for m in find_units("I'm in 304")] == ["304"]

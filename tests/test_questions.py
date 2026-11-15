"""Step C3: building-grounded questions (docs/BUILD_PLAN.md section 6).

Offline only: the synthetic building, fake sidecar metadata and (when built) the
processed bundy-olympic building with real suite numbers.
"""

from __future__ import annotations

import random

import pytest

import core.evidence as core_evidence
from core.prior import compute_prior
from core.schema import Policy
from core.uncertainty import entropy_bits
from ingest.matterport import load_building
from parser.next_question import (
    DONT_KNOW,
    OPEN_QUESTIONS,
    is_instruction,
    next_question,
    open_info_gain_bits,
    rank_open_questions,
    top_questions,
)


@pytest.fixture(scope="module")
def ptc():
    return load_building("ptc-100")


@pytest.fixture(scope="module")
def ptc_prior(ptc):
    return compute_prior(ptc)


@pytest.fixture(scope="module")
def suite_meta(ptc):
    """Suite and tenant for every room that is not a corridor, stair or elevator."""
    meta = {}
    shared = {"corridor", "stairwell", "elevator", "lobby"}
    for i, r in enumerate(r for r in ptc.rooms if r.room_type not in shared):
        meta[r.id] = {"suite": f"{r.floor}{i % 20:02d}", "tenant": f"Tenant {r.floor}-{i % 5}"}
    return meta


def _brute_open_gain(room_probs, p_nim, cells, eps=0.1, dont_know=DONT_KNOW):
    """Direct sum over answers, written the slow way."""
    labels = sorted(set(cells.values()))
    K = len(labels)
    total = sum(room_probs.values()) + p_nim
    labelled = sum(p for r, p in room_probs.items() if r in cells)

    def p_ans(a, r):
        return (1 - eps) if cells[r] == a else eps / (K - 1)

    avg = {a: sum(p * p_ans(a, r) for r, p in room_probs.items() if r in cells) / labelled for a in labels}
    h_after = 0.0
    for a in labels:
        w = {r: p * (p_ans(a, r) if r in cells else avg[a]) for r, p in room_probs.items()}
        w_nim = p_nim * avg[a]
        z = sum(w.values()) + w_nim
        h_after += z / total * entropy_bits(w, w_nim)
    return (1 - dont_know) * (entropy_bits(room_probs, p_nim) - h_after)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_open_gain_matches_the_direct_sum() -> None:
    rng = random.Random(7)
    for _ in range(5):
        rp = {f"r{i}": rng.random() for i in range(25)}
        s = sum(rp.values())
        rp = {k: v / s * 0.9 for k, v in rp.items()}
        cells = {f"r{i}": f"c{rng.randint(0, 5)}" for i in range(25) if rng.random() < 0.7}
        cells["no-prob-room"] = "c9"  # a cell nobody is likely in still counts as an answer
        assert open_info_gain_bits(rp, 0.1, cells) == pytest.approx(_brute_open_gain(rp, 0.1, cells), abs=1e-9)


def test_dont_know_scales_the_gain_and_a_clean_partition_gives_log_k() -> None:
    rp = {f"r{i}": 0.25 for i in range(4)}
    cells = {f"r{i}": str(i) for i in range(4)}
    assert open_info_gain_bits(rp, 0.0, cells, eps=0.0) == pytest.approx((1 - DONT_KNOW) * 2.0)
    assert open_info_gain_bits(rp, 0.0, cells, eps=0.0, dont_know=0.0) == pytest.approx(2.0)
    # noise floor: with eps = 0.1 the answer is worth less than a perfect one
    assert open_info_gain_bits(rp, 0.0, cells) < (1 - DONT_KNOW) * 2.0
    # one cell: nothing to learn
    assert open_info_gain_bits(rp, 0.0, {r: "x" for r in rp}) == 0.0


# ---------------------------------------------------------------------------
# Which questions are offered
# ---------------------------------------------------------------------------


def test_door_number_outranks_every_yes_no_at_the_prior(ptc, ptc_prior, suite_meta) -> None:
    rp, p_nim = ptc_prior
    qs = top_questions(ptc, rp, p_nim, meta=suite_meta, k=50)
    unit = next(q for q in qs if q.id == "open:unit")
    best_yes_no = max(q.expected_info_gain_bits for q in qs if q.kind == "yes_no")
    assert unit.kind == "open" and unit.partition == "unit" and unit.clue_type is None
    assert unit.question == OPEN_QUESTIONS["unit"]
    assert unit.expected_info_gain_bits > best_yes_no
    top3 = top_questions(ptc, rp, p_nim, meta=suite_meta)
    assert len(top3) == 3 and "open:unit" in {q.id for q in top3}
    gains = [q.expected_info_gain_bits for q in top3]
    assert gains == sorted(gains, reverse=True)


def test_real_building_door_number_question() -> None:
    try:
        bundy = load_building("bundy-olympic")
    except Exception:  # pragma: no cover - processed data not present
        pytest.skip("bundy-olympic not built")
    if not core_evidence.has_unit_metadata(bundy):
        pytest.skip("no suite metadata")
    rp, p_nim = compute_prior(bundy)
    qs = top_questions(bundy, rp, p_nim, k=100)
    assert qs[0].id == "open:unit"
    assert all(qs[0].expected_info_gain_bits > q.expected_info_gain_bits for q in qs if q.kind == "yes_no")


def test_no_door_number_question_on_the_synthetic_building(ptc, ptc_prior) -> None:
    rp, p_nim = ptc_prior
    ids = {q.id for q in top_questions(ptc, rp, p_nim, k=100)}
    assert "open:unit" not in ids and "open:tenant" not in ids
    assert "open:floor" in ids  # every room has a floor label
    assert {q["id"] for q in rank_open_questions(ptc, rp, p_nim)} == {"open:floor"}


def test_sparse_metadata_is_not_enough(ptc, ptc_prior) -> None:
    """Three labelled offices hold far less than half the probability: no unit question."""
    rp, p_nim = ptc_prior
    meta = {
        "f3-office-303": {"suite": "304", "tenant": "Acme Widgets"},
        "f4-office-401": {"suite": "401", "tenant": "Northside Law"},
        "f2-office-201": {"suite": "201", "tenant": "Dr. Patel Family Dental"},
    }
    ids = {q["id"] for q in rank_open_questions(ptc, rp, p_nim, meta=meta)}
    assert "open:unit" not in ids and "open:tenant" not in ids


def test_metadata_is_read_from_the_sidecar_by_default(ptc, ptc_prior, suite_meta, monkeypatch) -> None:
    monkeypatch.setattr(core_evidence, "room_meta", lambda b: suite_meta)
    rp, p_nim = ptc_prior
    assert "open:unit" in {q.id for q in top_questions(ptc, rp, p_nim, k=10)}


# ---------------------------------------------------------------------------
# Asked questions
# ---------------------------------------------------------------------------


def test_an_asked_question_does_not_return(ptc, ptc_prior, suite_meta) -> None:
    rp, p_nim = ptc_prior
    first = top_questions(ptc, rp, p_nim, meta=suite_meta, k=50)
    for q in first[:5]:
        again = top_questions(ptc, rp, p_nim, asked={q.id}, meta=suite_meta, k=50)
        assert q.id not in {x.id for x in again}


def test_a_given_unit_or_floor_retires_the_open_question(ptc, ptc_prior, suite_meta) -> None:
    rp, p_nim = ptc_prior
    ids = {q.id for q in top_questions(ptc, rp, p_nim, asked={"unit:304", "floor:3"}, meta=suite_meta, k=50)}
    assert "open:unit" not in ids and "open:floor" not in ids
    assert "open:tenant" in ids


def test_asking_which_floor_drops_the_floor_yes_no_questions(ptc, ptc_prior) -> None:
    rp, p_nim = ptc_prior
    before = top_questions(ptc, rp, p_nim, k=100)
    assert any(q.clue_type == "floor" for q in before)
    after = top_questions(ptc, rp, p_nim, asked={"open:floor"}, k=100)
    assert not any(q.clue_type == "floor" or q.id == "open:floor" for q in after)


def test_session_logged_open_question_is_not_offered_again(ptc) -> None:
    from api.session import AskRequest, CallSession

    s = CallSession(ptc)
    q = next(x for x in s.state().questions if x.kind == "open")
    s.ask(AskRequest(**q.model_dump()))
    assert s.revealed[-1].question is not None and s.revealed[-1].question.partition == q.partition
    assert q.id not in {x.id for x in s.state().questions}
    s.say("I don't know", speaker="caller")  # whatever the reply
    assert q.id not in {x.id for x in s.state().questions}


# ---------------------------------------------------------------------------
# Policy and wording
# ---------------------------------------------------------------------------


def test_no_questions_for_an_active_threat(ptc, ptc_prior, suite_meta) -> None:
    rp, p_nim = ptc_prior
    board = Policy(route_enabled=False, show_incident_board=True)
    unconfirmed = Policy(route_enabled=False)  # unconfirmed active-threat suggestion: route off, no board
    for policy in (board, unconfirmed, Policy(show_incident_board=True)):
        assert top_questions(ptc, rp, p_nim, policy=policy, meta=suite_meta) == []
        assert next_question(ptc, rp, p_nim, policy=policy) is None
    assert top_questions(ptc, rp, p_nim, policy=Policy(), meta=suite_meta)
    assert next_question(ptc, rp, p_nim, policy=Policy()) is not None


def test_no_question_tells_the_caller_to_act(ptc, ptc_prior, suite_meta) -> None:
    rp, p_nim = ptc_prior
    qs = top_questions(ptc, rp, p_nim, meta=suite_meta, k=1000)
    assert qs and not any(is_instruction(q.question) for q in qs)
    for text in ("Go to the front door", "Can you open the door for the crew?", "Please bring the AED",
                 "Could you walk to the lobby?"):
        assert is_instruction(text)
    for text in ("What number is on the door nearest you?", "Which floor did you get off at?",
                 "If you look outside, can you see the street?"):
        assert not is_instruction(text)


def test_open_question_shape_matches_the_dialog_reader(ptc, ptc_prior, suite_meta) -> None:
    """parser/dialog.py reads a bare number after partition "unit" or "floor"."""
    rp, p_nim = ptc_prior
    for q in top_questions(ptc, rp, p_nim, meta=suite_meta, k=1000):
        if q.kind == "open":
            assert q.partition in ("unit", "tenant", "floor") and q.id == f"open:{q.partition}"
            assert q.p_yes is None and q.value is None
        else:
            assert q.id == f"{q.clue_type}:{q.value}" and q.partition is None


def test_next_question_still_returns_the_best_yes_no(ptc, ptc_prior) -> None:
    rp, p_nim = ptc_prior
    nq = next_question(ptc, rp, p_nim)
    assert nq is not None and {"question", "clue_type", "value", "expected_info_gain_bits", "p_yes"} <= set(nq)
    best_yes_no = next(q for q in top_questions(ptc, rp, p_nim, k=100) if q.kind == "yes_no")
    assert (nq["clue_type"], nq["value"]) == (best_yes_no.clue_type, best_yes_no.value)


# ---------------------------------------------------------------------------
# Layout questions and questions the caller already answered
# ---------------------------------------------------------------------------


def _ids(qs) -> set[str]:
    return {q.id for q in qs}


def test_layout_questions_are_offered_and_score_the_current_belief(ptc, ptc_prior) -> None:
    from parser.layout import LAYOUT_QUESTIONS, layout_features

    feats = layout_features(ptc)
    assert feats["corner"] and feats["interior"] and not (feats["corner"] & feats["interior"])
    rp, p_nim = ptc_prior
    layout_ids = {q.id for q in top_questions(ptc, rp, p_nim, k=200) if q.clue_type == "layout"}
    assert layout_ids and layout_ids <= {f"layout:{k}" for k in LAYOUT_QUESTIONS}

    # once nearly all the probability sits on corner rooms, "is it a corner room?" tells us little
    corner_heavy = {rid: (1.0 if rid in feats["corner"] else 1e-4) for rid in rp}
    total = sum(corner_heavy.values())
    corner_heavy = {rid: p / total for rid, p in corner_heavy.items()}
    gain = {q.id: q.expected_info_gain_bits for q in top_questions(ptc, rp, p_nim, k=200)}
    gain_after = {q.id: q.expected_info_gain_bits for q in top_questions(ptc, corner_heavy, 0.0, k=200)}
    assert gain_after.get("layout:corner", 0.0) < gain["layout:corner"]
    assert _ids(top_questions(ptc, rp, p_nim)) != _ids(top_questions(ptc, corner_heavy, 0.0))


def test_a_layout_answer_retires_its_question_and_the_opposite_one(ptc, ptc_prior) -> None:
    rp, p_nim = ptc_prior
    yes = _ids(top_questions(ptc, rp, p_nim, asked={"layout:corner"}, k=200))
    assert "layout:corner" not in yes and "layout:interior" not in yes
    no = _ids(top_questions(ptc, rp, p_nim, asked={"negative:layout:corner"}, k=200))
    assert "layout:corner" not in no and "layout:interior" in no


def test_a_question_the_caller_answered_in_their_own_words_is_not_suggested(ptc, ptc_prior) -> None:
    from core.clue_likelihood import clue_match
    from core.schema import Clue

    rp, p_nim = ptc_prior
    street = Clue(type="view", value="street", polarity=True, confidence=1.0, span="")
    street_rooms = frozenset(rid for rid, m in clue_match(ptc, street).items() if m >= 0.5)
    assert "view:street" in _ids(top_questions(ptc, rp, p_nim, k=200))
    # the reader recorded "I can see the street" as a clue over the street-facing rooms
    assert "view:street" not in _ids(top_questions(ptc, rp, p_nim, k=200, answered=[street_rooms]))
    # a clue about something else leaves it alone
    other = frozenset(list(rp)[:3])
    assert "view:street" in _ids(top_questions(ptc, rp, p_nim, k=200, answered=[other]))

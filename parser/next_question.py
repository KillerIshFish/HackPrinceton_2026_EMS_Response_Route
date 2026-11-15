"""Information-gain next question for the dispatcher.

For every candidate yes/no question derived from the controlled vocabulary
(room types, views, landmarks, floors) we model the caller's answer as a noisy
channel built from the same match vector and epsilon floor the fusion step uses:

    l_yes(r) = eps + (1 - eps) * m(r)          # what fusion multiplies in on "yes"
    l_no(r)  = eps + (1 - eps) * (1 - m(r))    # ... and on "no"
    P(yes | r) = l_yes(r) / (l_yes(r) + l_no(r))

The "not in model" bucket answers at the posterior base rate of m. The score is
the expected entropy reduction (mutual information between answer and room):

    IG = H(p) - [ P(yes) * H(p | yes) + P(no) * H(p | no) ]      (bits)

``next_question`` returns the best not-yet-asked yes/no question (older callers).

Layout questions
----------------
Besides the vocabulary, every building gets yes/no questions about its own shape, from the
layout map the reasoning reader uses (``parser.layout.layout_features``): a corner room, an
inside room, one of the bigger or smaller rooms on its floor, at the end of a hallway, a short
walk from the entrance. They are scored with the same information gain, so which of them is
offered follows the rooms still likely. Answers are ``layout`` clues.

A yes/no question is also dropped once a statement on the console already answers it: a
descriptive clue from the caller's own words picking out nearly the same rooms
(``already_answered``), e.g. "I can see the street" against "can you see the street?".

Open questions (Step C3)
------------------------
``top_questions`` also scores open questions built from *this* building, each with a
partition of the rooms into answer cells:

* ``open:unit``   "What number is on the door nearest you?"  cells = suite numbers
                  (sidecar ``rooms[id]["suite"]``, normalised with ``normalize_unit``)
* ``open:tenant`` "What name or sign can you see?"           cells = tenant names
* ``open:floor``  "Which floor did you get off at?"          cells = floor labels

An open question is offered only when its labelled rooms hold at least
``MIN_LABELLED_SHARE`` of the in-building probability and it has two or more cells.
Answer model, with the fusion noise floor ``eps``:

    P(don't know)            = DONT_KNOW = 0.2   (independent of the room: no information)
    P(answer a | r, answers) = 1 - eps            if a is r's cell
                             = eps / (K - 1)      for each of the other K - 1 cells
    unlabelled rooms and "not in model" answer like the average labelled room.

    IG = (1 - DONT_KNOW) * [ H(p) - sum_a P(a | answers) * H(p | a) ]   (bits)

The top ``k`` questions of both kinds, best gain first, go to ``DemoState.questions``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

import core.evidence as _evidence
from core.clue_likelihood import clue_match
from core.schema import Building, Clue, Policy, Question
from core.uncertainty import entropy_bits
from parser.extract import SECTION_FOR_TYPE, TYPE_FOR_SECTION, load_vocab
from parser.layout import LAYOUT_EXCLUDES, LAYOUT_QUESTIONS, layout_features

# Answer noise floor: must equal core.fusion.EPS (0.1), so the scored answer model is the update fusion
# really applies (docs/BUILD_PLAN.md step C3). Duplicated here, not imported.
EPS = 0.1
# Numerical zero for a gain: a question at or below this cannot change the posterior, so it is not offered.
MIN_GAIN_BITS = 1e-6

# A yes / no question counts as already answered when a descriptive statement on the console
# (the reasoning reader's clue from the caller's own words) picks out nearly the same rooms:
# overlap / union of the two room sets at least this much. Tuned heuristic: 3 rooms in 4 shared is
# "the same question"; a lower bar would hide questions that still split the remaining rooms.
ANSWERED_OVERLAP = 0.75

# Open questions: probability the caller cannot answer, and the share of the in-building
# probability the labelled rooms must hold before the question is offered.
# Both from docs/BUILD_PLAN.md step C3: "don't know" has probability 0.2 (assumed, not measured), and an
# open question needs data for "the rooms holding most of the probability" (read as at least half).
DONT_KNOW = 0.2
MIN_LABELLED_SHARE = 0.5

OPEN_QUESTIONS: dict[str, str] = {
    "unit": "What number is on the door nearest you?",
    "tenant": "What name or sign can you see?",
    "floor": "Which floor did you get off at?",
}

# Room types a dispatcher would not normally ask about directly.
_SKIP_ROOM_TYPES: set[str] = set()

_ROOM_TYPE_PHRASES = {
    "restroom": "a restroom or bathroom",
    "office": "a private office",
    "open_office": "an open office area with desks or cubicles",
    "conference": "a conference or meeting room",
    "break_room": "a break room or kitchen",
    "lobby": "the lobby",
    "reception": "a reception or waiting area",
    "corridor": "a hallway",
    "stairwell": "a stairwell",
    "elevator": "an elevator",
    "storage": "a storage room or closet",
    "server_room": "a server or IT room",
    "mechanical": "a mechanical or utility room",
}
_VIEW_PHRASES = {
    "parking_lot": "a parking lot",
    "street": "the street",
}
_LANDMARK_QUESTIONS = {
    "window": "Is there a window in the room you're in?",
    "elevator": "Are you close to the elevators?",
    "stairs": "Are you close to a stairwell or exit stairs?",
    "vending_machine": "Is there a vending machine where you are?",
    "printer": "Is there a printer or copier near you?",
    "reception_desk": "Can you see a reception or security desk?",
    "sink": "Is there a sink in the room?",
}


def question_key(clue_type: str, value: str) -> str:
    """Stable id for a question, used in the ``asked`` set: ``"<type>:<value>"``."""
    return f"{clue_type}:{value}"


def asked_from_clues(clues: Iterable[Clue]) -> set[str]:
    """``asked`` keys for clues the caller already gave.

    Asserted clues -> ``"<type>:<value>"``; denied clues (type "negative" or
    polarity False) -> ``"negative:<section>:<key>"``. ``rank_questions`` expands
    these into everything they make redundant (see ``_excluded_keys``).
    """
    keys: set[str] = set()
    for clue in clues:
        if clue.type == "negative":
            keys.add(f"negative:{clue.value}")
        elif not clue.polarity:
            keys.add(f"negative:{SECTION_FOR_TYPE.get(clue.type, clue.type)}:{clue.value}")
        else:
            keys.add(question_key(clue.type, clue.value))
    return keys


def _excluded_keys(asked: Iterable[str], candidates: Iterable[tuple[str, str]], vocab: dict) -> set[str]:
    """Question keys that are already answered or made redundant by ``asked``.

    * ``room_type:X`` (caller said they are in X) -> every room-type question, and
      landmark questions that are only a proxy for a room type (sink, vending machine).
    * ``view:X`` -> every view question and "is there a window" (a view implies one).
    * ``floor:X`` -> every floor question.
    * ``negative:<section>:<key>`` -> just that one question.
    Anything else in ``asked`` excludes only the identical key.
    """
    asked = set(asked)
    candidates = list(candidates)
    out = set(asked)
    landmarks = vocab.get(SECTION_FOR_TYPE["landmark"]) or {}
    room_proxy_landmarks = {
        str(k) for k, v in landmarks.items() if set(((v or {}).get("match") or {}).keys()) == {"room_type"}
    }
    for key in asked:
        head, _, rest = key.partition(":")
        if head == "negative":
            section, _, value = rest.partition(":")
            if value and section == "layout":  # "no" to a layout question
                out.add(question_key("layout", value))
            elif value and section in TYPE_FOR_SECTION:
                out.add(question_key(TYPE_FOR_SECTION[section], value))
                if (section, value) == ("landmarks", "window"):  # no window -> no view
                    out.update(question_key(t, v) for t, v in candidates if t == "view")
            else:  # bare key: exclude it under every type it could belong to
                out.update(question_key(t, v) for t, v in candidates if v == rest)
        elif head == "room_type":
            out.update(question_key(t, v) for t, v in candidates if t == "room_type")
            out.update(question_key("landmark", v) for v in room_proxy_landmarks)
        elif head == "view":
            out.update(question_key(t, v) for t, v in candidates if t == "view")
            out.add(question_key("landmark", "window"))
        elif head == "floor":
            out.update(question_key(t, v) for t, v in candidates if t == "floor")
        elif head == "layout":  # "yes, a corner room" makes "an inside room?" pointless
            out.update(question_key("layout", v) for v in LAYOUT_EXCLUDES.get(rest, ()))
    return out


def _phrase_question(clue_type: str, value: str) -> str:
    pretty = value.replace("_", " ")
    if clue_type == "room_type":
        return f"Are you in {_ROOM_TYPE_PHRASES.get(value, 'a ' + pretty)}?"
    if clue_type == "view":
        return f"If you look outside, can you see {_VIEW_PHRASES.get(value, 'the ' + pretty)}?"
    if clue_type == "landmark":
        return _LANDMARK_QUESTIONS.get(value, f"Can you see a {pretty} near you?")
    if clue_type == "floor":
        return f"Do you think you're on floor {value}?"
    if clue_type == "layout" and value in LAYOUT_QUESTIONS:
        return LAYOUT_QUESTIONS[value]
    return f"Does this apply to where you are: {pretty}?"


def candidate_questions(building: Building, vocab: dict | None = None) -> list[tuple[str, str]]:
    """(clue_type, value) pairs worth asking about for this building."""
    vocab = vocab or load_vocab()
    present_types = {r.room_type for r in building.rooms}
    out: list[tuple[str, str]] = []
    for key in vocab.get(SECTION_FOR_TYPE["room_type"]) or {}:
        if key in present_types and key not in _SKIP_ROOM_TYPES:
            out.append(("room_type", str(key)))
    for ctype in ("view", "landmark"):
        for key in vocab.get(SECTION_FOR_TYPE[ctype]) or {}:
            out.append((ctype, str(key)))
    # Questions about the shape of this building: offered when the feature splits the rooms
    # (some have it, some do not). Whether it splits the rooms still likely is up to the gain.
    n_rooms = len(building.rooms)
    for key, rooms in layout_features(building).items():
        if 0 < len(rooms) < n_rooms:
            out.append(("layout", key))
    seen_labels: set[str] = set()
    for room in sorted(building.rooms, key=lambda r: r.floor):
        if room.floor_label not in seen_labels:
            seen_labels.add(room.floor_label)
            out.append(("floor", room.floor_label))
    return out


def expected_info_gain_bits(
    room_probs: Mapping[str, float], p_nim: float, match: Mapping[str, float], eps: float = EPS
) -> tuple[float, float]:
    """Return (expected entropy reduction in bits, P(yes)) for one yes/no question."""
    in_model = sum(room_probs.values())
    if in_model <= 0.0:
        return 0.0, 0.5  # no room mass: no information, and no basis for P(yes) either way

    def p_yes_given(m: float) -> float:
        m = min(1.0, max(0.0, m))
        l_yes = eps + (1.0 - eps) * m
        l_no = eps + (1.0 - eps) * (1.0 - m)
        return l_yes / (l_yes + l_no)

    # "not in model" answers like the posterior-weighted average room; an unscored room has match 1.0
    m_nim = sum(p * match.get(rid, 1.0) for rid, p in room_probs.items()) / in_model
    yes_room = {rid: p * p_yes_given(match.get(rid, 1.0)) for rid, p in room_probs.items()}
    yes_nim = p_nim * p_yes_given(m_nim)
    p_yes = sum(yes_room.values()) + yes_nim
    total = in_model + p_nim
    p_no = total - p_yes
    if p_yes <= 0.0 or p_no <= 0.0:
        return 0.0, p_yes / total if total > 0 else 0.5

    h_now = entropy_bits({rid: p / total for rid, p in room_probs.items()}, p_nim / total)
    h_yes = entropy_bits({rid: v / p_yes for rid, v in yes_room.items()}, yes_nim / p_yes)
    h_no = entropy_bits(
        {rid: (room_probs[rid] - v) / p_no for rid, v in yes_room.items()}, (p_nim - yes_nim) / p_no
    )
    gain = h_now - (p_yes / total) * h_yes - (p_no / total) * h_no
    return max(0.0, gain), p_yes / total


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def already_answered(match: Mapping[str, float], answered: Iterable[frozenset[str]], all_ids: frozenset[str]) -> bool:
    """True when a statement already on the console settles this yes / no question.

    The question splits the rooms into the ones a "yes" points to (match at least 0.5) and the
    rest. If a described set of rooms is nearly either side, the caller has in effect answered
    it ("I can see the street out the window" answers "can you see the street?").
    """
    yes = frozenset(rid for rid, m in match.items() if m >= 0.5)  # m >= 0.5: rooms where l_yes >= l_no
    no = all_ids - yes
    if not yes or not no:
        return False
    return any(_overlap(s, yes) >= ANSWERED_OVERLAP or _overlap(s, no) >= ANSWERED_OVERLAP for s in answered)


def rank_questions(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
    vocab: dict | None = None,
    answered: Iterable[frozenset[str]] = (),
) -> list[dict]:
    """All not-yet-asked candidate questions, best expected information gain first.

    ``answered``: room sets the caller has already described in their own words (see
    ``already_answered``); a question splitting out nearly the same rooms is left out.
    """
    vocab = vocab or load_vocab()
    candidates = candidate_questions(building, vocab)
    excluded = _excluded_keys(asked or set(), candidates, vocab)
    answered = [a for a in answered if a]
    all_ids = frozenset(r.id for r in building.rooms)
    ranked: list[dict] = []
    for clue_type, value in candidates:
        if question_key(clue_type, value) in excluded:
            continue
        probe = Clue(type=clue_type, value=value, polarity=True, confidence=1.0, span="")  # type: ignore[arg-type]
        match = clue_match(building, probe, vocab)
        if answered and already_answered(match, answered, all_ids):
            continue
        gain, p_yes = expected_info_gain_bits(room_probs, p_nim, match)
        ranked.append(
            {
                "question": _phrase_question(clue_type, value),
                "clue_type": clue_type,
                "value": value,
                "expected_info_gain_bits": round(gain, 4),
                "p_yes": round(p_yes, 4),
            }
        )
    ranked.sort(key=lambda q: -q["expected_info_gain_bits"])
    return ranked


def prompts_allowed(policy: Policy | None) -> bool:
    """False when the console must not suggest questions to the caller.

    That is whenever the route is off (``route_enabled`` false, e.g. an unconfirmed
    active-threat suggestion) or the incident board is shown: during an active threat the
    console does not prompt contact with callers.
    """
    return policy is None or (policy.route_enabled and not policy.show_incident_board)


def next_question(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
    *,
    policy: Policy | None = None,
    answered: Iterable[frozenset[str]] = (),
) -> dict | None:
    """Best next yes/no question, or None when nothing left would reduce uncertainty.

    Returns ``{"question", "clue_type", "value", "expected_info_gain_bits", "p_yes"}``.
    ``asked`` holds keys to exclude: ``"<type>:<value>"`` for things the caller
    asserted or was already asked, ``"negative:<section>:<key>"`` for things they
    denied (``asked_from_clues(clues)`` builds this). Questions made redundant by
    those answers are skipped too - nobody asks "are you in an office?" after
    "I'm in a bathroom". Open questions are only in ``top_questions``. With a
    ``policy`` that forbids prompts (see ``prompts_allowed``) this returns None.
    """
    if not prompts_allowed(policy):
        return None
    ranked = rank_questions(building, room_probs, p_nim, asked, answered=answered)
    if not ranked or ranked[0]["expected_info_gain_bits"] <= MIN_GAIN_BITS:
        return None
    return ranked[0]


# ---------------------------------------------------------------------------
# Open questions built from the building
# ---------------------------------------------------------------------------


def _norm_tenant(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def open_partitions(
    building: Building, meta: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, dict[str, str]]:
    """``partition -> {room id: answer cell}``; rooms without that data are left out.

    ``unit`` from sidecar ``suite`` (``normalize_unit``), ``tenant`` from sidecar ``tenant``
    (case / space folded), ``floor`` from ``Room.floor_label``. ``meta`` overrides the
    building's sidecar metadata (tests).
    """
    meta = _evidence.room_meta(building) if meta is None else meta
    units: dict[str, str] = {}
    tenants: dict[str, str] = {}
    for r in building.rooms:
        m = meta.get(r.id) or {}
        unit = _evidence.normalize_unit(m.get("suite")) if m.get("suite") else ""
        if unit:
            units[r.id] = unit
        tenant = _norm_tenant(m.get("tenant"))
        if tenant:
            tenants[r.id] = tenant
    floors = {r.id: str(r.floor_label) for r in building.rooms}
    return {"unit": units, "tenant": tenants, "floor": floors}


def labelled_share(room_probs: Mapping[str, float], cells: Mapping[str, str]) -> float:
    """Share of the in-building probability held by rooms that have a cell."""
    in_model = sum(p for p in room_probs.values() if p > 0.0)
    if in_model <= 0.0:
        return 0.0
    return sum(p for rid, p in room_probs.items() if p > 0.0 and rid in cells) / in_model


def _sum_vlogv(f: float, s: float, g: float) -> float:
    """sum over a group of (f * q) * log2(f * q), given s = sum q and g = sum q log2 q."""
    if f <= 0.0 or s <= 0.0:
        return 0.0
    return f * (s * math.log2(f) + g)


def open_info_gain_bits(
    room_probs: Mapping[str, float],
    p_nim: float,
    cells: Mapping[str, str],
    eps: float = EPS,
    dont_know: float = DONT_KNOW,
) -> float:
    """Expected entropy reduction (bits) of an open question with answer cells ``cells``.

    See the module docstring for the answer model. Rooms in one cell share a likelihood,
    so each answer's posterior entropy comes from per-cell sums in O(1); the whole score
    is O(rooms + cells).
    """
    labels = set(cells.values())
    K = len(labels)
    total = sum(p for p in room_probs.values() if p > 0.0) + max(p_nim, 0.0)
    if K < 2 or total <= 0.0:
        return 0.0

    s_cell: dict[str, float] = defaultdict(float)  # sum of q per cell (q = p / total)
    g_cell: dict[str, float] = defaultdict(float)  # sum of q log2 q per cell
    s_u = g_u = 0.0  # rooms without a cell
    for rid, p in room_probs.items():
        if p <= 0.0:
            continue
        q = p / total
        qlog = q * math.log2(q)
        c = cells.get(rid)
        if c is None:
            s_u += q
            g_u += qlog
        else:
            s_cell[c] += q
            g_cell[c] += qlog
    s_l = sum(s_cell.values())
    if s_l <= 0.0:
        return 0.0
    g_l = sum(g_cell.values())
    n = max(p_nim, 0.0) / total
    h_now = -(g_l + g_u + (n * math.log2(n) if n > 0.0 else 0.0))

    hit, miss = 1.0 - eps, eps / (K - 1)
    all_miss = _sum_vlogv(miss, s_l, g_l)  # every labelled room at the miss rate
    expected_h = 0.0
    for a in labels:
        s_a, g_a = s_cell.get(a, 0.0), g_cell.get(a, 0.0)
        labelled_a = hit * s_a + miss * (s_l - s_a)
        q_a = labelled_a / s_l  # P(answer a) for an average labelled room
        z = labelled_a + (s_u + n) * q_a  # P(answer a | the caller answers)
        if z <= 0.0:
            continue
        vlogv = (
            all_miss - _sum_vlogv(miss, s_a, g_a) + _sum_vlogv(hit, s_a, g_a)
            + _sum_vlogv(q_a, s_u, g_u)
            + (n * q_a * math.log2(n * q_a) if n > 0.0 and q_a > 0.0 else 0.0)
        )
        expected_h += z * (math.log2(z) - vlogv / z)
    return max(0.0, (1.0 - dont_know) * (h_now - expected_h))


def _open_excluded(asked: Iterable[str]) -> set[str]:
    """Open partitions already asked or answered.

    ``open:<partition>`` (logged through POST /calls/demo/ask, whatever the reply) and
    ``unit:<x>`` / ``tenant:<x>`` / ``floor:<x>`` (the caller already gave one).
    """
    out: set[str] = set()
    for key in asked:
        head, _, rest = str(key).partition(":")
        if not rest:
            continue
        if head == "open":
            out.add(rest)
        elif head in OPEN_QUESTIONS:
            out.add(head)
    return out


def rank_open_questions(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Open questions this building supports and not yet asked, best gain first.

    Each: ``{"id", "question", "partition", "n_answers", "labelled_share",
    "expected_info_gain_bits"}``. A partition is offered only with two or more cells and
    when its labelled rooms hold at least ``MIN_LABELLED_SHARE`` of the probability.
    """
    excluded = _open_excluded(asked or ())
    out: list[dict] = []
    for partition, cells in open_partitions(building, meta).items():
        if partition in excluded:
            continue
        n_answers = len(set(cells.values()))
        if n_answers < 2:
            continue
        share = labelled_share(room_probs, cells)
        if share < MIN_LABELLED_SHARE:
            continue
        gain = open_info_gain_bits(room_probs, p_nim, cells)
        out.append(
            {
                "id": f"open:{partition}",
                "question": OPEN_QUESTIONS[partition],
                "partition": partition,
                "n_answers": n_answers,
                "labelled_share": round(share, 4),
                "expected_info_gain_bits": round(gain, 4),
            }
        )
    out.sort(key=lambda q: -q["expected_info_gain_bits"])
    return out


# Questions must not tell the caller to move, fetch something or open a door; those are
# protocol prompts (Policy.protocol_prompts), not location questions.
_INSTRUCTION = re.compile(
    r"^\s*(?:please\s+)?(?:go|move|walk|run|leave|head|get|fetch|bring|grab|open|unlock|close|shut|take|carry|pick)\b"
    r"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:go|move|walk|run|leave|head|get|fetch|bring|grab|open|unlock|carry|pick)\b",
    re.IGNORECASE,
)


def is_instruction(text: str) -> bool:
    """True for text that tells the caller to do something rather than asking where they are."""
    return bool(_INSTRUCTION.search(text or ""))


def top_questions(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
    policy: Policy | None = None,
    k: int = 3,  # docs/BUILD_PLAN.md step C3: "return the top three"
    *,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
    vocab: dict | None = None,
    answered: Iterable[frozenset[str]] = (),
) -> list[Question]:
    """The best ``k`` questions for DemoState.questions, yes/no and open, best gain first.

    * yes/no: vocabulary questions from ``rank_questions`` (``id = "<type>:<value>"``;
      ``clue_type`` / ``value`` are what the answer buttons send back).
    * open: ``rank_open_questions`` (``id = "open:<partition>"``, ``partition`` set).

    ``asked`` holds every key that counts as asked: logged question ids (an asked open
    question is never offered again, whatever the reply), asserted / denied clue keys, and
    ``unit:<x>`` / ``tenant:<x>`` for a unit or tenant the caller already gave. Once
    ``open:floor`` is asked the yes/no floor questions are dropped too.

    Returns nothing when ``prompts_allowed(policy)`` is false (route off or incident board
    shown). Protocol prompts are never produced here; they come from Policy.protocol_prompts.
    """
    if not prompts_allowed(policy):
        return []
    asked = set(asked or ())
    candidates: list[Question] = []
    for q in rank_questions(building, room_probs, p_nim, asked, vocab, answered=answered):
        if q["clue_type"] == "floor" and "open:floor" in asked:
            continue
        candidates.append(
            Question(
                id=question_key(q["clue_type"], q["value"]),
                question=q["question"],
                kind="yes_no",
                expected_info_gain_bits=q["expected_info_gain_bits"],
                clue_type=q["clue_type"],
                value=q["value"],
                p_yes=q["p_yes"],
            )
        )
    for q in rank_open_questions(building, room_probs, p_nim, asked, meta):
        candidates.append(
            Question(
                id=q["id"],
                question=q["question"],
                kind="open",
                expected_info_gain_bits=q["expected_info_gain_bits"],
                partition=q["partition"],
            )
        )
    candidates = [
        q for q in candidates
        if q.expected_info_gain_bits > MIN_GAIN_BITS and q.id not in asked and not is_instruction(q.question)
    ]
    candidates.sort(key=lambda q: -q.expected_info_gain_bits)  # stable: ties keep yes/no first
    return candidates[: max(k, 0)]

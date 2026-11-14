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

The best not-yet-asked question is returned.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from core.clue_likelihood import clue_match
from core.schema import Building, Clue
from core.uncertainty import entropy_bits
from parser.extract import SECTION_FOR_TYPE, TYPE_FOR_SECTION, load_vocab

EPS = 0.1
MIN_GAIN_BITS = 1e-6

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
            if value and section in TYPE_FOR_SECTION:
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
        return 0.0, 0.5

    def p_yes_given(m: float) -> float:
        m = min(1.0, max(0.0, m))
        l_yes = eps + (1.0 - eps) * m
        l_no = eps + (1.0 - eps) * (1.0 - m)
        return l_yes / (l_yes + l_no)

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


def rank_questions(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
    vocab: dict | None = None,
) -> list[dict]:
    """All not-yet-asked candidate questions, best expected information gain first."""
    vocab = vocab or load_vocab()
    candidates = candidate_questions(building, vocab)
    excluded = _excluded_keys(asked or set(), candidates, vocab)
    ranked: list[dict] = []
    for clue_type, value in candidates:
        if question_key(clue_type, value) in excluded:
            continue
        probe = Clue(type=clue_type, value=value, polarity=True, confidence=1.0, span="")  # type: ignore[arg-type]
        match = clue_match(building, probe, vocab)
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


def next_question(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    asked: set[str] | None = None,
) -> dict | None:
    """Best next yes/no question, or None when nothing left would reduce uncertainty.

    Returns ``{"question", "clue_type", "value", "expected_info_gain_bits", "p_yes"}``.
    ``asked`` holds keys to exclude: ``"<type>:<value>"`` for things the caller
    asserted or was already asked, ``"negative:<section>:<key>"`` for things they
    denied (``asked_from_clues(clues)`` builds this). Questions made redundant by
    those answers are skipped too - nobody asks "are you in an office?" after
    "I'm in a bathroom".
    """
    ranked = rank_questions(building, room_probs, p_nim, asked)
    if not ranked or ranked[0]["expected_info_gain_bits"] <= MIN_GAIN_BITS:
        return None
    return ranked[0]

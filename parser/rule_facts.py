"""Facts and an incident-type suggestion read from one caller turn by fixed patterns.

The offline counterpart of the agent's ``record_fact`` / ``suggest_incident_type`` tools
(step C2), so the "rules" and "llm" parser modes also put caller reports on the board.
``api.session.CallSession`` merges this into the Interpretation of a caller turn whenever
the turn was not read by the agent.

Deliberately narrow: a pattern fires only on plain wording ("smoke", "not breathing",
"he has a gun", "door's shut", "Dave will meet you out front"), a negated hazard or threat
("no smoke", "I didn't hear shots") is skipped, and the fact text is the caller's own
sentences, never a paraphrase. Every fact is ``verified=False``. One fact per category per
turn (sentences joined), split only when sentences name different rooms. A fact carries a
``room_id`` only when its sentence names exactly one room of the building (by a room number
such as "303" or a unique room name such as "Records Room"); a closed-door report may take
the one room named anywhere in the same turn ("I'm in 302. I shut the door.").
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Sequence

from core.schema import Building, Fact, IncidentType, TranscriptTurn

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_NEGATION = re.compile(r"(?<!\w)(?:no|not|never|nobody|nothing|without)(?!\w)|n'?t(?!\w)", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[,;:]|(?<!\w)(?:but|and|so|then)(?!\w)", re.IGNORECASE)

# category -> pattern. Patient and people patterns carry their own negation ("not breathing",
# "not hurt"), so only threat and hazard mentions are checked for a preceding negation.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "threat": re.compile(
        r"(?<!\w)(?:shooter|gunman|gunshots?|shots(?:\s+fired)?|shot\s+fired|shooting|gunfire|stabbing"
        r"|(?:has|had|with|got|carrying)\s+(?:a\s+|the\s+)?(?:gun|rifle|pistol|knife|weapon))(?!\w)",
        re.IGNORECASE,
    ),
    "hazard": re.compile(
        r"(?<!\w)(?:smoke|smoky|fire|flames?|burning|fire\s+alarm|alarms?\s+(?:is\s+|are\s+)?going\s+off)(?!\w)",
        re.IGNORECASE,
    ),
    "patient": re.compile(
        r"(?<!\w)(?:not\s+breathing|isn'?t\s+breathing|stopped\s+breathing|no\s+pulse|can'?t\s+find\s+a\s+pulse"
        r"|unresponsive|collapsed|passed\s+out|won'?t\s+wake\s+up|(?:been|got|was|is|he'?s|she'?s)\s+shot"
        r"|stabbed|bleeding|wounded|cpr|compressions|turning\s+(?:blue|gray|grey))(?!\w)",
        re.IGNORECASE,
    ),
    "access": re.compile(
        r"(?<!\w)(?:door(?:'s|\s+is|\s+was)?\s+(?:closed|shut|locked)|(?:closed|shut|locked)\s+(?:the|my|our)\s+door"
        r"|barricaded|locked\s+(?:in|inside|ourselves|myself))(?!\w)",
        re.IGNORECASE,
    ),
    "people": re.compile(
        r"(?<!\w)(?:hiding|there(?:'s|\s+are|\s+is)\s+\w+\s+of\s+us|(?:two|three|four|five|six|\d+)\s+of\s+us"
        r"|\w+\s+(?:people|others|coworkers)\s+(?:with\s+me|in\s+here)"
        r"|(?:i'?m|we'?re)\s+(?:okay|ok|fine|not\s+hurt)|(?:nobody|no\s+one)(?:'s|\s+is)\s+hurt)(?!\w)",
        re.IGNORECASE,
    ),
}

_MEET = re.compile(
    r"(?<!\w)(?:meet\s+(?:you|them|the\s+(?:ambulance|crew|medics|paramedics|firefighters))"
    r"|wait(?:ing)?\s+(?:for\s+(?:you|them|the\s+(?:ambulance|crew|medics))|(?:out\s+front|at\s+the\s+(?:front|door|entrance|lobby)))"
    r"|flag\s+(?:you|them)\s+down|(?:send|sending)\s+\w+\s+(?:down|out))(?!\w)",
    re.IGNORECASE,
)

_SUGGEST: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("active_threat", _PATTERNS["threat"]),
    ("fire_smoke", re.compile(r"(?<!\w)(?:smoke|smoky|fire|flames?)(?!\w)", re.IGNORECASE)),
    ("cardiac_arrest", re.compile(
        r"(?<!\w)(?:not\s+breathing|isn'?t\s+breathing|stopped\s+breathing|no\s+pulse|can'?t\s+find\s+a\s+pulse"
        r"|cpr|compressions)(?!\w)",
        re.IGNORECASE,
    )),
)
_NEGATABLE = {"threat", "hazard", "active_threat", "fire_smoke"}

_ROOM_NUMBER = re.compile(r"(?<!\w)(?<!\d\.)(\d{3,4}[A-Za-z]?)(?!\w)(?!\.\d)")
_NUMBER_IN_NAME = re.compile(r"(?<!\w)(\d{2,4}[A-Za-z]?)(?!\w)")


def _negated(sentence: str, start: int) -> bool:
    head = sentence[max(0, start - 30):start]
    breaks = list(_CLAUSE_BREAK.finditer(head))
    clause = head[breaks[-1].end():] if breaks else head
    return bool(_NEGATION.search(clause))


def _clean(sentence: str) -> str:
    s = " ".join(sentence.split()).strip()
    return s[:1].upper() + s[1:] if s else s


def room_named(building: Building, text: str) -> Optional[str]:
    """The one room ``text`` names by number ("303") or by a unique room name, else None."""
    low = text.lower()
    hits: set[str] = set()
    numbers = {m.group(1).lower() for m in _ROOM_NUMBER.finditer(text)}
    if numbers:
        for r in building.rooms:
            tokens = {m.group(1).lower() for m in _NUMBER_IN_NAME.finditer(r.name)}
            if tokens & numbers:
                hits.add(r.id)
    if not hits:
        names: dict[str, list[str]] = {}
        for r in building.rooms:
            name = re.sub(r"\s*\(.*?\)\s*", " ", r.name).strip().lower()
            if len(name) >= 6:
                names.setdefault(name, []).append(r.id)
        for name, ids in names.items():
            if len(ids) == 1 and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", low):
                hits.add(ids[0])
    return next(iter(hits)) if len(hits) == 1 else None


def read_facts(
    building: Building,
    transcript: Sequence[TranscriptTurn],
    index: int,
    *,
    start_k: int = 0,
) -> tuple[list[Fact], Optional[IncidentType], Optional[str]]:
    """``(facts, suggested_type, suggested_span)`` for caller turn ``transcript[index]``.

    Fact ids are ``t<index>-f<k>`` from ``k = start_k``. Suggestions: a threat report ->
    ``active_threat`` (wins over everything); smoke or fire -> ``fire_smoke``; "not
    breathing", "no pulse", CPR -> ``cardiac_arrest``. Non-caller turns give nothing.
    """
    turn = transcript[index]
    if turn.speaker != "caller":
        return [], None, None
    text = turn.text or ""
    observed: Optional[datetime] = turn.at
    turn_room: Optional[str] = None
    turn_room_read = False
    # (category, premise_key, room_id) -> sentences, in first-seen order
    groups: dict[tuple[str, Optional[str], Optional[str]], list[tuple[str, str]]] = {}
    suggestion: dict[str, str] = {}

    for m in _SENTENCE.finditer(text):
        raw = m.group(0)
        sentence = _clean(raw)
        if not sentence:
            continue
        keys: list[tuple[str, Optional[str]]] = []
        for category, rx in _PATTERNS.items():
            hit = rx.search(raw)
            if hit is not None and not (category in _NEGATABLE and _negated(raw, hit.start())):
                keys.append((category, None))
        meet = _MEET.search(raw)
        if meet is not None and not _negated(raw, meet.start()):
            keys = [k for k in keys if k[0] != "people"] + [("people", "meet_person")]
        if keys:
            room = room_named(building, raw)
            for category, premise in keys:
                room_id: Optional[str] = None
                if category == "hazard":
                    room_id = room
                elif category == "access":
                    if room is None and not turn_room_read:
                        turn_room, turn_room_read = room_named(building, text), True
                    room_id = room or turn_room
                groups.setdefault((category, premise, room_id), []).append((sentence, raw.strip()))
        for kind, rx in _SUGGEST:
            hit = rx.search(raw)
            if hit is not None and kind not in suggestion and not (kind in _NEGATABLE and _negated(raw, hit.start())):
                suggestion[kind] = " ".join(raw.split())

    facts = [
        Fact(
            id=f"t{index}-f{start_k + k}", category=category,  # type: ignore[arg-type]
            text=" ".join(s for s, _ in sentences), span=sentences[0][1], turn=index, source="rules", observed_at=observed, room_id=room_id,
            verified=False, premise_key=premise,  # type: ignore[arg-type]
        )
        for k, ((category, premise, room_id), sentences) in enumerate(groups.items())
    ]
    for kind, _ in _SUGGEST:
        if kind in suggestion:
            return facts, kind, suggestion[kind]  # type: ignore[return-value]
    return facts, None, None


__all__ = ["read_facts", "room_named"]

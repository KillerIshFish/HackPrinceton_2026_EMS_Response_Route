"""Deterministic dialog rules that run before any extractor or model.

They read a caller turn *in the context of the call*, which the single-utterance
extractor (``parser.extract``) cannot:

1. **Answer resolution.** A caller reply that opens with a yes / no form ("yes", "yeah",
   "uh huh", "no", "nope", "I don't think so") right after a dispatcher question about
   ``(clue_type, value)`` becomes that clue with the matching polarity, confidence 0.85,
   span = the reply's opening words. "I don't know" / "not sure" emit nothing (the
   question still counts as asked, see ``asked_keys``). The Yes / No / Not sure buttons
   (``TranscriptTurn.answer``) do the same at confidence 0.9 with source "dispatcher" and
   win over the text. A dispatcher turn without a structured ``Question`` whose text ends
   in "?" and names exactly one vocabulary clue ("Are you on the fourth floor?") is read
   as that question, at confidence 0.75.
2. **Numbers and names printed on rooms.** "I'm in 304", "unit 12B", "suite 860",
   "apartment 4", "my office, 302", "the door says 302" -> grounded evidence with
   ``selector.unit``. Its confidence is the stated share of ``core.fusion``: 0.95 for a
   plain, exact statement down to 0.40 for a hesitant, vague one, from the rule rating of
   the caller's words (``parser.statement_rating.rate_statement``; in Claude Haiku mode the
   model's rating replaces it). The number is resolved against the building's own labels: the
   sidecar suite records and the numbers in room names ("Office 302", "Unit 202"). Every
   room carrying the number is selected; when no room does, the statement is kept as
   unmatched and a floor clue is added from the leading digits (3+ digit numbers: 304 ->
   floor 3, 1204 -> floor 12) if the building has that floor, at confidence 0.5. A room
   or tenant name said as the caller's own place ("I'm in the server room", "we're at
   Nadel Architects", "the sign says Boardroom") selects the rooms that carry the name
   (``find_names``). A place that is not the caller's is left alone: "next to 301", "the
   fire's in 301", "we're not in 304". After the dispatcher asked for the unit (console
   question or "What suite are you in?") a bare number ("860") is a unit; after a floor
   question a bare number or number word is a floor; after "Where are you?" a number
   standing alone in its clause is read when it is a label in this building.
3. **Corrections.** A correction cue ("no wait", "sorry, I meant", "actually", "I mean",
   "scratch that", ...) followed by a statement of a slot already stated supersedes the
   earlier evidence of that slot: earlier turns (from ``prior_evidence``) and earlier in
   the same turn. Slots are ``evidence_slots``: floor, room_type, view, landmark, sound,
   smell, route, unit (a unit also implies a floor), selector.

Public API: ``read_dialog`` (all three rules for one turn), ``resolve_answer``,
``find_units``, ``unit_statements``, ``find_names``, ``name_statements``, ``reply_context``,
``correction_cue``, ``evidence_slots``, ``superseded_ids``, ``asked_keys``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from core.evidence import normalize_unit, room_labels, selector_hits
from core.evidence import room_meta as _room_meta
from core.schema import Building, Clue, Evidence, EvidenceSource, Question, RoomSelector, TranscriptTurn
from parser.extract import SECTION_FOR_TYPE, TYPE_FOR_SECTION, extract_clues_rules
from parser.statement_rating import PLAIN, Rating, rate_statement

ANSWER_CONFIDENCE = 0.85
BUTTON_CONFIDENCE = 0.9
INFERRED_QUESTION_CONFIDENCE = 0.75
HEDGED_YES_FACTOR = 0.7  # "I think so"
# A number or a name printed on a room, and a floor, are specific statements (core.fusion):
# their confidence is the stated share, mapped from how sure and how exact the caller sounds
# (parser.statement_rating; configs/statements.yaml). A floor read off the leading digits of
# a number no room carries is an inference, not a statement, and keeps this fixed share.
UNIT_FLOOR_FALLBACK_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@dataclass
class DialogStatement:
    """One statement read by a dialog rule; ``parser.interpret`` gives it an id and wraps it.

    Exactly one of ``clue`` (vocab) / ``selector`` (grounded) is set. ``start`` / ``end``
    are character offsets of ``span`` in the turn text (-1 when it came from a button).
    ``rating`` is set on a specific statement in the caller's own words (a number, a name, a
    floor): the rule estimate of how sure and how exact it sounds, which ``confidence`` (the
    stated share) was mapped from. The model reader may replace it (``with_rating``).
    """

    confidence: float
    span: str
    source: EvidenceSource = "rules"
    clue: Optional[Clue] = None
    selector: Optional[RoomSelector] = None
    rationale: str = ""
    start: int = -1
    end: int = -1
    rating: Optional[Rating] = None

    @property
    def kind(self) -> str:
        return "vocab" if self.clue is not None else "grounded"

    def with_rating(self, rating: Rating) -> "DialogStatement":
        """This statement with another rating (and the stated share that follows from it)."""
        share = rating.share()
        clue = self.clue.model_copy(update={"confidence": share}) if self.clue is not None else None
        return DialogStatement(confidence=share, span=self.span, source=self.source, clue=clue,
                               selector=self.selector, rationale=self.rationale, start=self.start, end=self.end,
                               rating=rating)


def rating_note(rating: Rating, by: str) -> str:
    """Short text for a rationale or a trace: how the stated share came about."""
    return (f"stated share {rating.share():.2f} (certainty {rating.certainty:g}, specificity {rating.specificity:g}, "
            f"rated by {by})")


@dataclass
class DialogReading:
    """What the dialog rules read from one caller turn.

    * ``statements``   clues / selectors to add, in order (answer first, then units)
    * ``question_key`` ``"<clue_type>:<value>"`` of the question this turn answered, if any
    * ``answer``       "yes" | "no" | "unsure" | None
    * ``correction_at`` offset in the text after which statements correct earlier ones
                       (None: no correction cue)
    """

    statements: list[DialogStatement] = field(default_factory=list)
    question_key: Optional[str] = None
    answer: Optional[str] = None
    correction_at: Optional[int] = None


# ---------------------------------------------------------------------------
# 1. Answer resolution
# ---------------------------------------------------------------------------

_FILLER = r"(?:(?:well|um+|uh+|er+|oh|ok(?:ay)?|so|hmm+)\s*[,.]?\s+)*"
# Order matters: "not sure" / "no idea" before "no"; "I don't think so" before "I don't know".
_UNSURE = re.compile(
    rf"^\s*{_FILLER}(?P<w>i\s+(?:do\s+not|don'?t)\s+know|i'?m\s+not\s+(?:sure|certain)|not\s+(?:sure|certain)"
    r"|no\s+idea|dunno|i\s+(?:can'?t|cannot)\s+tell|i\s+have\s+no\s+idea|maybe|i'?m\s+unsure|unsure)(?!\w)",
    re.IGNORECASE,
)
_NO = re.compile(
    rf"^\s*{_FILLER}(?P<w>i\s+(?:do\s+not|don'?t)\s+think\s+so|not\s+really|uh[\s-]+uh|nope|nah|negative"
    r"|no(?![\s,.!-]*(?:wait|sorry|hold\s+on|actually))(?:\s*,?\s*no)?)(?!\w)",
    re.IGNORECASE,
)
_YES = re.compile(
    rf"^\s*(?P<w>uh[\s-]*huh|mm[\s-]*hmm|mhm)(?!\w)|^\s*{_FILLER}(?P<w2>yes|yeah|yep|yup|yah|ya|correct|affirmative"
    r"|that'?s\s+right|i\s+think\s+so|i\s+believe\s+so|definitely|absolutely)(?!\w)",
    re.IGNORECASE,
)
_HEDGED_YES = re.compile(r"i\s+(?:think|believe)\s+so", re.IGNORECASE)


@dataclass(frozen=True)
class _AskedQuestion:
    clue_type: Optional[str]
    value: Optional[str]
    partition: Optional[str]
    inferred: bool  # read from the dispatcher's free text, not a structured Question


def _question_of(turn: TranscriptTurn, vocab: dict | None) -> Optional[_AskedQuestion]:
    q: Optional[Question] = turn.question
    if q is not None:
        if q.kind == "yes_no" and q.clue_type and q.value:
            return _AskedQuestion(q.clue_type, q.value, None, False)
        if q.kind == "open":
            partition = q.partition or (q.id.split(":", 1)[1] if q.id.startswith("open:") else None)
            return _AskedQuestion(None, None, partition, False)
        return None
    text = (turn.text or "").strip()
    if not text.endswith("?"):
        return None
    clues = [c for c in extract_clues_rules(text, vocab=vocab) if c.polarity and c.type != "negative"]
    if len(clues) != 1:
        return None
    return _AskedQuestion(clues[0].type, clues[0].value, None, True)


def previous_question(
    transcript: Sequence[TranscriptTurn], index: int, vocab: dict | None = None
) -> Optional[_AskedQuestion]:
    """The question the dispatcher asked just before ``transcript[index]`` (system turns skipped)."""
    for j in range(index - 1, -1, -1):
        t = transcript[j]
        if t.speaker == "system":
            continue
        if t.speaker != "dispatcher":
            return None
        return _question_of(t, vocab)
    return None


def answer_clue(clue_type: str, value: str, yes: bool, confidence: float, span: str) -> Clue:
    """The clue a yes / no answer to the question ``(clue_type, value)`` stands for.

    A "no" is written the way the extractor writes a denial: ``type="negative"``,
    ``value="<section>:<key>"``, ``polarity=False``.
    """
    if yes:
        return Clue(type=clue_type, value=value, polarity=True, confidence=confidence, span=span)  # type: ignore[arg-type]
    if clue_type in SECTION_FOR_TYPE:
        return Clue(type="negative", value=f"{SECTION_FOR_TYPE[clue_type]}:{value}", polarity=False,
                    confidence=confidence, span=span)
    return Clue(type=clue_type, value=value, polarity=False, confidence=confidence, span=span)  # type: ignore[arg-type]


def classify_reply(text: str) -> Optional[tuple[str, int, int]]:
    """``(answer, start, end)`` for a reply opening with a yes / no / unsure form, else None."""
    for answer, rx in (("unsure", _UNSURE), ("no", _NO), ("yes", _YES)):
        m = rx.match(text or "")
        if m:
            grp = "w" if m.group("w") is not None else "w2"
            return answer, m.start(grp), m.end(grp)
    return None


def resolve_answer(
    transcript: Sequence[TranscriptTurn], index: int, vocab: dict | None = None
) -> tuple[Optional[str], Optional[str], Optional[DialogStatement]]:
    """``(question_key, answer, statement)`` for a caller turn answering a yes / no question.

    Buttons (``turn.answer``) win over the text. "unsure" gives a key but no statement.
    """
    turn = transcript[index]
    text = turn.text or ""
    reply = classify_reply(text)
    if turn.answer is not None:
        a = turn.answer
        key = f"{a.clue_type}:{a.value}"
        if a.answer == "unsure":
            return key, "unsure", None
        start, end = (reply[1], reply[2]) if reply else (-1, -1)
        span = text[start:end] if reply else text.strip()[:60]
        clue = answer_clue(a.clue_type, a.value, a.answer == "yes", BUTTON_CONFIDENCE, span)
        return key, a.answer, DialogStatement(
            confidence=BUTTON_CONFIDENCE, span=span, source="dispatcher", clue=clue, start=start, end=end,
            rationale=f"dispatcher marked the answer '{a.answer}' to {key}",
        )

    q = previous_question(transcript, index, vocab)
    if q is None or q.clue_type is None or q.value is None or reply is None:
        return None, None, None
    key = f"{q.clue_type}:{q.value}"
    answer, start, end = reply
    if answer == "unsure":
        return key, "unsure", None
    span = text[start:end]
    conf = INFERRED_QUESTION_CONFIDENCE if q.inferred else ANSWER_CONFIDENCE
    if answer == "yes" and _HEDGED_YES.fullmatch(span.strip()):
        conf *= HEDGED_YES_FACTOR
    conf = round(conf, 3)
    clue = answer_clue(q.clue_type, q.value, answer == "yes", conf, span)
    return key, answer, DialogStatement(
        confidence=conf, span=span, clue=clue, start=start, end=end,
        rationale=f"'{span}' answers the question just asked ({key})",
    )


# ---------------------------------------------------------------------------
# 2. Numbers and names printed on rooms
# ---------------------------------------------------------------------------

_UNIT_TOKEN = r"(?P<unit>\d{1,5}[A-Za-z]?|[A-Za-z]\d{1,4})"
_NOT_A_UNIT_AFTER = r"(?!\s*(?:st|nd|rd|th)?\s*(?:floor|level|stor(?:y|ey)|minutes?|mins?|seconds?|hours?|years?|people|persons?|percent|%|am|pm|o'?clock)\b)"
_UNIT_WORD = r"(?P<kw>suite|ste\.?|unit|apartment|apt\.?|room|rm\.?|office|condo|flat|number)"
_UNIT_KEYWORD = re.compile(
    r"(?<!\w)" + _UNIT_WORD + r"\s*#?\s*" + _UNIT_TOKEN + r"(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)
# "my office, 302" / "the suite is 860" / "room number 12" / "apartment - 4B": the number is
# separated from the word. Read only when the number is a label in this building
# (``needs_label``), so "the room. 20 of us" stays out.
_LABEL_NUMBER = r"(?P<unit>\d{2,5}[A-Za-z]?|\d[A-Za-z]|[A-Za-z]\d{1,4})"
_UNIT_KEYWORD_LOOSE = re.compile(
    r"(?<!\w)" + _UNIT_WORD
    + r"(?:\s+number)?\s*(?:[,:.–—-]\s*|\s+is\s+|\s+was\s+|\s+it'?s\s+|\s+number\s+)(?:number\s+)?#?\s*"
    + _LABEL_NUMBER + r"(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)
# "the door says 302", "the sign reads 12B", "the number on the door is 302"
_UNIT_SIGN = re.compile(
    r"(?<!\w)(?:(?:door|sign|plate|plaque|placard|label)\s+(?:says|reads|is|has)"
    r"|number\s+on\s+the\s+(?:door|sign|wall)\s+(?:is|says|reads)"
    r"|it\s+says|says)\s+(?:(?:room|suite|unit|apartment|office|number)\s+)?#?\s*"
    + _LABEL_NUMBER + r"(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)
_UNIT_HASH = re.compile(r"(?<![\w#])#\s*" + _UNIT_TOKEN + r"(?!\w)" + _NOT_A_UNIT_AFTER, re.IGNORECASE)
_SUBJECT_BE = (r"(?:i'?m|i\s+am|we'?re|we\s+are|(?:s?he|it)'?s|(?:s?he|it)\s+is|they'?re|they\s+are|you'?re"
               r"|(?:s?he|it|they|we|i)\s+(?:was|were))")
# "we're in 530", "I'm at 2B" (one digit needs a letter: "I'm in 2" says nothing about a unit).
_LOCATIVE_UNIT = r"(?P<unit>\d{2,5}[A-Za-z]?|\d[A-Za-z])"
_UNIT_LOCATIVE = re.compile(
    r"(?<!\w)" + _SUBJECT_BE + r"\s+(?:in|at|inside)\s+(?:number\s+)?" + _LOCATIVE_UNIT + r"(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)
# "we're across the hall in 530" / "across the hall in 530". Without a subject the phrase must
# open its clause (checked in code), so "the AED is across the hall in 530" is not a location.
_UNIT_ACROSS = re.compile(
    r"(?<!\w)(?:(?P<subj>" + _SUBJECT_BE + r")\s+(?:(?:right|just)\s+)?)?"
    r"across\s+the\s+(?:hall|hallway|corridor)\s+(?:in|at)\s+(?:(?:suite|unit|number|room)\s+#?\s*)?"
    + _LOCATIVE_UNIT + r"(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)
# A number that opens its clause and ends it ("302.", "uh, 302, third floor") in a reply to
# "where are you?". Read only when the number is a label in this building.
_CLAUSE_NUMBER = re.compile(
    r"(?:^|(?<=[.,;!?]))\s*(?:(?:um+|uh+|er+|oh|okay|ok|it'?s|its|number|room)\s+)*#?\s*"
    r"(?P<unit>\d{2,5}[A-Za-z]?|\d[A-Za-z])\s*(?=[.,;!?]|$)",
    re.IGNORECASE,
)
# A number after "room" that follows a room-type word names a room ("conference room 2",
# "break room 3"), not a unit; name matching handles those.
_ROOM_NAME_WORD = re.compile(
    r"(?<!\w)(?:conference|break|meeting|rest|bath|class|board|bed|living|dining|waiting|exam|examination"
    r"|treatment|therapy|operating|recovery|procedure|consult(?:ation)?|interview|training|computer|server|copy"
    r"|mail|supply|storage|stock|utility|laundry|locker|mechanical|boiler|electrical|equipment|family|guest"
    r"|lunch|music|game|play|common|community|party|sitting|work|study|reading|media|fitness|weight|green"
    r"|wash|change|changing|dressing|show|control|data|file|print|nursing|break-out|breakout)\s+$",
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"^\s*(?:it'?s\s+|its\s+|we'?re\s+in\s+|i'?m\s+in\s+)?#?\s*" + _UNIT_TOKEN + r"\s*[.!]?\s*$", re.IGNORECASE)
_BARE_FLOOR = re.compile(
    r"^\s*(?:it'?s\s+|we'?re\s+on\s+|i'?m\s+on\s+)?(?:the\s+)?(?:floor\s+)?(?P<n>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:st|nd|rd|th)?(?:\s+floor)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}

# What the dispatcher just asked, read from free text (a structured Question says it itself).
_ASKS_UNIT = re.compile(
    r"(?<!\w)(?:(?:which|what)(?:'s|\s+is)?\s+(?:the\s+|your\s+)?(?:room|suite|unit|apartment|apt|office|door)(?:\s+number)?"
    r"|(?:room|suite|unit|apartment|apt|office|door)\s+number|number\s+(?:is\s+)?on\s+(?:the|your)\s+door)(?!\w)",
    re.IGNORECASE,
)
_ASKS_FLOOR = re.compile(r"(?<!\w)(?:which|what)\s+(?:floor|level|story|storey)(?!\w)", re.IGNORECASE)
_ASKS_WHERE = re.compile(
    r"(?<!\w)(?:where(?:\s+exactly|abouts)?\s+(?:are|r)\s+(?:you|u|they|we)|where\s+(?:is|are)\s+(?:he|she|the\s+patient|they)"
    r"|where\s+(?:exactly\s+)?in\s+the\s+building|where\s+(?:exactly|abouts)|tell\s+me\s+(?:exactly\s+)?where"
    r"|(?:what(?:'s|\s+is)|give\s+me)\s+your\s+(?:exact\s+)?location|your\s+location)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UnitMention:
    unit: str  # as spoken ("12B")
    span: str
    start: int
    end: int
    keyword: Optional[str] = None  # the word it came with ("office" in "office 302"), lower case
    needs_label: bool = False  # only a statement when the number is a label in this building


_NEGATION = re.compile(r"(?:(?<!\w)not(?!\w)|n'?t(?!\w)|(?<!\w)never(?!\w)|(?<!\w)no\s+longer(?!\w))", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[.,;!?]|(?<!\w)(?:but|and|so)(?!\w)", re.IGNORECASE)
# A place the caller mentions relative to their own: "next to 301", "across from suite 520",
# "just off the main lobby", "two doors past the server room". Not where the caller is.
_RELATIVE_BEFORE = re.compile(
    r"(?<!\w)(?:next\s+(?:door\s+)?to|across(?:\s+the\s+\w+)?\s+from|opposite|beside|near(?:\s+to)?|nearby|close\s+to|by"
    r"|past|behind|above|below|over|under(?:neath)?|outside(?:\s+of)?|in\s+front\s+of|off(?:\s+of)?"
    r"|(?:down|up)\s+(?:the\s+\w+\s+)?from|around\s+the\s+corner\s+from|doors?\s+(?:down|up|past|from|before|after)(?:\s+from)?"
    r"|before|after|between|towards?)\s+(?:(?:the|my|our|a|an|that|this)\s+)?"
    r"(?:(?!(?:in|at|on|inside|and|i'?m|we'?re|is|it'?s)(?!\w))[\w'-]+\s+){0,2}$",
    re.IGNORECASE,
)
# The clause is about the hazard or the threat, not about the caller: "the fire's in 301",
# "smoke coming from suite 310", "the shooter went into room 12".
_OTHER_SUBJECT = re.compile(
    r"(?<!\w)(?:fire|smoke|flames?|blaze|burning|shooter|gunman|shooting|gunshots?|shots?|suspect|intruder"
    r"|attacker|alarm|explosion|noise|sound|bang|pop|smell|leak)(?:'s|s)?(?!\w)",
    re.IGNORECASE,
)
_CALLER_WORD = re.compile(r"(?<!\w)(?:my|our|i'?m|i\s+am|we'?re|we\s+are)(?!\w)", re.IGNORECASE)
# Someone other than the caller going somewhere: "he went up to the third floor", "they ran
# down to the lobby", "she is heading for the roof". Where they went is not where the caller is.
_MOVING = (r"(?:went|goes|going|gone|ran|runs|running|headed|heading|walked|walking|fled|fleeing|took\s+off|moved|moving"
           r"|came|coming|climbed|climbing|left|leaving|took|taking|carried|carrying|brought|bringing)")
_OTHER_MOVED = re.compile(
    r"(?<!\w)(?:he|she|they|it|him|her|them|someone|somebody|everyone|everybody|people|the\s+[\w'-]+|a\s+[\w'-]+"
    r"|(?:this|that|some|my|our|his|her|their)\s+[\w'-]+)(?:'s|\s+is|\s+are|\s+was|\s+were|\s+has|\s+have|\s+just|\s+already)*\s+"
    + _MOVING + r"(?!\w)",
    re.IGNORECASE,
)
_CALLER_MOVED = re.compile(
    r"(?<!\w)(?:i|we)(?:'ve|'m|'re|\s+am|\s+are|\s+have|\s+just|\s+already)*\s+" + _MOVING + r"(?!\w)", re.IGNORECASE)


def _clause_head(text: str, start: int, window: int = 70) -> str:
    """The text of the mention's clause before ``start``."""
    head = text[max(0, start - window):start]
    breaks = list(_CLAUSE_BREAK.finditer(head))
    return head[breaks[-1].end():] if breaks else head


def _negated(text: str, start: int) -> bool:
    """True when a negation ("not", "aren't", "never", "no longer") precedes ``start`` in its clause.

    "We're not in suite 304" must not point at suite 304. The rules path has no way to
    express a denial of a unit, so a negated mention is dropped (a wrong room is worse than
    no room); the agent path records the denial with ``negate``.
    """
    return bool(_NEGATION.search(_clause_head(text, start, 40)))


def someone_elses_place(text: str, start: int) -> bool:
    """True when the place mentioned at ``start`` is not where the caller is: it is named
    relative to the caller ("next to 301", "just off the lobby") or its clause is about the
    fire, the smoke or the threat ("the fire's in 301")."""
    if _RELATIVE_BEFORE.search(text[max(0, start - 40):start]):
        return True
    head = _clause_head(text, start)
    # "there's smoke in my office, 302" is still the caller's own office
    return bool(_OTHER_SUBJECT.search(head)) and not _CALLER_WORD.search(head)


_someone_elses_place = someone_elses_place


def someone_elses_floor(text: str, start: int) -> bool:
    """True when the floor mentioned at ``start`` is not a statement of where the caller is:
    its clause is about the fire, the smoke or the threat ("the fire is on the third floor")
    or about someone else going there ("he went up to the third floor"). A floor the caller
    themselves went to ("we ran up to the third floor") is theirs."""
    head = _clause_head(text, start)
    if _OTHER_SUBJECT.search(head) and not _CALLER_WORD.search(head):
        return True
    return bool(_OTHER_MOVED.search(head)) and not _CALLER_MOVED.search(head)


# Numbers as speech recognition writes them: "three oh two", "four seventy five", "six sixty".
_DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9",
                # homophones speech recognition writes for digits; never accepted on their own
                "won": "1", "too": "2", "to": "2", "for": "4", "fore": "4", "ate": "8"}
_AMBIGUOUS_DIGIT_WORDS = {"o", "won", "too", "to", "for", "fore", "ate"}
_TEEN_WORDS = {"ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
               "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19"}
_TENS_WORDS = {"twenty": 2, "thirty": 3, "forty": 4, "fourty": 4, "fifty": 5, "sixty": 6, "seventy": 7, "eighty": 8,
               "ninety": 9}
_NUMBER_WORD = "|".join(sorted(set(_DIGIT_WORDS) | set(_TEEN_WORDS) | set(_TENS_WORDS) | {"hundred"}, key=len, reverse=True))
_UNIT_SPOKEN = re.compile(
    r"(?<!\w)(?:(?P<kw>suite|sweet|unit|apartment|apt|room|office|number)\s+(?:number\s+)?|"
    + _SUBJECT_BE + r"\s+(?:in|at)\s+(?:(?:suite|sweet|unit|apartment|room|office|number)\s+)?)"
    r"(?P<words>(?:" + _NUMBER_WORD + r")(?:[\s-]+(?:" + _NUMBER_WORD + r")){1,5})(?!\w)",
    re.IGNORECASE,
)


def spoken_number(words: str) -> Optional[str]:
    """Digits for a number said in words, the way people say room numbers, else None.

    "three oh two" -> "302", "four seventy five" -> "475", "six sixty" -> "660", "twelve oh
    four" -> "1204", "three hundred two" -> "302". Homophones ("too", "for", "ate") count only
    next to at least two plain number words.
    """
    toks = [t for t in re.split(r"[\s-]+", words.strip().lower()) if t]
    plain = [t for t in toks if t not in _AMBIGUOUS_DIGIT_WORDS]
    if len(toks) < 2 or len(plain) < 2 or len(plain) < len(toks) - 1:
        return None
    out = ""
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "hundred":
            if not out or i + 1 >= len(toks):
                out += "00"
            elif toks[i + 1] in _DIGIT_WORDS and i + 2 >= len(toks):
                out += "0"  # "three hundred two" -> 302
        elif t in _TENS_WORDS:
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if nxt in _DIGIT_WORDS and _DIGIT_WORDS[nxt] != "0":
                out += str(_TENS_WORDS[t]) + _DIGIT_WORDS[nxt]
                i += 1
            else:
                out += str(_TENS_WORDS[t]) + "0"
        elif t in _TEEN_WORDS:
            out += _TEEN_WORDS[t]
        elif t in _DIGIT_WORDS:
            out += _DIGIT_WORDS[t]
        else:
            return None
        i += 1
    return out if 2 <= len(out) <= 5 else None


def _opens_clause(text: str, start: int) -> bool:
    """True when only a clause break (or nothing, or a filler) comes before ``start``."""
    head = re.sub(r"(?:(?<!\w)(?:um+|uh+|oh|okay|ok|well|so|and|no|now)\s*)+$", "", text[:start].rstrip(), flags=re.IGNORECASE)
    head = head.rstrip()
    return not head or head[-1] in ",.;:!?-"


def find_units(text: str, *, labels: Optional[set[str]] = None, where_reply: bool = False) -> list[UnitMention]:
    """Unit / suite / room numbers in one utterance, in text order, without overlaps or repeats.

    Left out: mentions negated within their clause ("we're not in 304") and places that are
    not the caller's own ("next to 301", "the fire's in 301").

    ``labels`` (normalised labels of this building's rooms, ``core.evidence.room_labels``)
    turns on the readings that are only safe when the number is printed on a room here:
    "my office, 302", "the suite is 860", and, with ``where_reply`` (the dispatcher just
    asked where the caller is), a number standing alone in its clause ("302, third floor").
    """
    found: list[UnitMention] = []
    patterns: list[tuple[re.Pattern[str], bool]] = [(_UNIT_KEYWORD, False), (_UNIT_HASH, False), (_UNIT_ACROSS, False),
                                                    (_UNIT_LOCATIVE, False)]
    if labels is not None:
        patterns += [(_UNIT_KEYWORD_LOOSE, True), (_UNIT_SIGN, True)]
        if where_reply:
            patterns.append((_CLAUSE_NUMBER, True))
    for rx, needs_label in patterns:
        for m in rx.finditer(text or ""):
            if any(not (m.end() <= f.start or m.start() >= f.end) for f in found):
                continue
            unit = m.group("unit")
            if needs_label and normalize_unit(unit) not in (labels or ()):
                continue
            if _negated(text, m.start()):
                continue
            if rx is not _UNIT_ACROSS and _someone_elses_place(text, m.start()):
                continue  # "next to suite 520", "the fire's in 301"
            keyword = (m.groupdict().get("kw") or "").lower().rstrip(".") or None
            if (rx in (_UNIT_KEYWORD, _UNIT_KEYWORD_LOOSE) and keyword in ("room", "rm")
                    and _ROOM_NAME_WORD.search(text[max(0, m.start() - 30):m.start()])):
                continue  # "conference room 2" is a room name
            if rx is _UNIT_ACROSS and m.group("subj") is None and not _opens_clause(text, m.start()):
                continue  # "the AED is across the hall in 530": not where the caller is
            start = m.start("unit") if rx is _CLAUSE_NUMBER else m.start()
            found.append(UnitMention(unit, text[start:m.end()], start, m.end(), keyword, needs_label))
    if labels is not None:  # "office three oh two", "sweet four seventy five": only numbers this building has
        for m in _UNIT_SPOKEN.finditer(text or ""):
            if any(not (m.end() <= f.start or m.start() >= f.end) for f in found):
                continue
            unit = spoken_number(m.group("words"))
            if unit is None or normalize_unit(unit) not in labels:
                continue
            if _negated(text, m.start()) or someone_elses_place(text, m.start()):
                continue
            keyword = (m.group("kw") or "").lower() or None
            found.append(UnitMention(unit, text[m.start():m.end()], m.start(), m.end(),
                                     "suite" if keyword == "sweet" else keyword, True))
    out: list[UnitMention] = []
    seen: set[str] = set()
    for f in sorted(found, key=lambda f: f.start):
        if f.unit.lower() not in seen:
            seen.add(f.unit.lower())
            out.append(f)
    return out


def _floor_label_exists(building: Building, label: str) -> bool:
    return any(r.floor_label.strip().lower() == label.lower() for r in building.rooms)


def unit_floor_label(building: Building, unit: str) -> Optional[str]:
    """Floor a 3+ digit unit number names by its leading digits (304 -> "3"), if the building has it."""
    m = re.fullmatch(r"(\d{3,5})[A-Za-z]?", unit.strip())
    if not m:
        return None
    label = str(int(m.group(1)[:-2]))
    return label if _floor_label_exists(building, label) else None


_KEYWORD_IN_NAME = {"ste": "suite", "apt": "apartment", "rm": "room"}


def _room_names(building: Building, room_ids: Sequence[str], limit: int = 3) -> str:
    names = {r.id: r.name for r in building.rooms}
    shown = [names[rid] for rid in room_ids[:limit] if rid in names]
    more = len(room_ids) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def unit_statements(building: Building, mention: UnitMention, *, meta: dict | None = None,
                    text: str = "") -> list[DialogStatement]:
    """Grounded evidence for one number the caller gave as their unit, suite or room.

    The number is looked up in the building's own labels (``core.evidence.unit_room_ids``:
    suite records, else numbers printed in room names). With a word ("office 302") the rooms
    whose name also has that word are preferred when there are any. Every room carrying the
    label is selected, so a number two rooms share raises both. A number no room carries is
    kept as an unmatched statement (the UI says it could not be placed), plus a floor clue
    from its leading digits at confidence 0.5 when the building has that floor.

    The confidence is the stated share for how the caller said it (``rate_statement``).
    """
    rating = rate_statement(text, mention.start, mention.end) if text else PLAIN
    confidence = rating.share()
    selector = RoomSelector(unit=mention.unit)
    hits = selector_hits(building, selector, meta)
    if hits and mention.keyword and mention.keyword != "number":
        word = _KEYWORD_IN_NAME.get(mention.keyword, mention.keyword)
        narrower = RoomSelector(unit=mention.unit, name_contains=word)
        narrow_hits = selector_hits(building, narrower, meta)
        if narrow_hits:
            selector, hits = narrower, narrow_hits
    rationale = (f"{mention.unit} is the number on {_room_names(building, hits)}" if hits
                 else f"no room of this building is numbered {mention.unit}")
    out = [DialogStatement(
        confidence=confidence, span=mention.span, selector=selector, start=mention.start, end=mention.end,
        rationale=rationale, rating=rating if hits else None,
    )]
    if not hits:
        label = unit_floor_label(building, mention.unit)
        if label is not None:
            out.append(DialogStatement(
                confidence=UNIT_FLOOR_FALLBACK_CONFIDENCE, span=mention.span,
                clue=Clue(type="floor", value=label, polarity=True, confidence=UNIT_FLOOR_FALLBACK_CONFIDENCE,
                          span=mention.span),
                start=mention.start, end=mention.end,
                rationale=f"no room numbered {mention.unit}; leading digits read as floor {label}",
            ))
    return out


def _open_answer_statements(
    building: Building, text: str, partition: Optional[str], *, meta: dict | None = None
) -> list[DialogStatement]:
    """A bare number replying to an open unit / floor question."""
    if partition == "unit":
        m = _BARE_NUMBER.match(text or "")
        if m:
            mention = UnitMention(m.group("unit"), text[m.start("unit"):m.end("unit")], m.start("unit"), m.end("unit"))
            return unit_statements(building, mention, meta=meta, text=text)
    if partition == "floor":
        m = _BARE_FLOOR.match(text or "")
        if m:
            tok = m.group("n").lower()
            label = str(int(tok)) if tok.isdigit() else str(_NUMBER_WORDS[tok])
            if _floor_label_exists(building, label):
                span = text.strip()
                start = text.find(span)
                rating = rate_statement(text, start, start + len(span))  # the caller's own statement of the floor
                share = rating.share()
                return [DialogStatement(
                    confidence=share, span=span, start=start, end=start + len(span),
                    clue=Clue(type="floor", value=label, polarity=True, confidence=share, span=span),
                    rationale=f"'{span}' answers the floor question just asked", rating=rating,
                )]
    return []


def reply_context(transcript: Sequence[TranscriptTurn], index: int) -> set[str]:
    """What the dispatcher asked just before ``transcript[index]``: any of "unit", "floor", "where".

    From a structured open ``Question`` (``partition``) or from the dispatcher's own words
    ("What suite are you in?", "What floor?", "Where are you?"). Empty when the previous
    turn is not a dispatcher turn.
    """
    for j in range(index - 1, -1, -1):
        t = transcript[j]
        if t.speaker == "system":
            continue
        if t.speaker != "dispatcher":
            return set()
        out: set[str] = set()
        q = t.question
        if q is not None and q.kind == "open":
            partition = q.partition or (q.id.split(":", 1)[1] if q.id.startswith("open:") else None)
            if partition in ("unit", "floor"):
                out.add(partition)
            if partition in ("unit", "tenant"):
                out.add("where")
        text = t.text or ""
        if _ASKS_UNIT.search(text):
            out |= {"unit", "where"}
        if _ASKS_FLOOR.search(text):
            out.add("floor")
        if _ASKS_WHERE.search(text):
            out.add("where")
        return out
    return set()


# ---------------------------------------------------------------------------
# 2b. Room and tenant names
# ---------------------------------------------------------------------------

_NAME_STOP = {"the", "a", "an", "of", "and"}
_CORPORATE_SUFFIX = {"inc", "llc", "llp", "lp", "ltd", "co", "corp", "corporation", "company", "pc", "pllc", "plc"}
# One-word names that say nothing about which room ("Room", "Other", "Unit").
_GENERIC_NAME = {"room", "rooms", "other", "unknown", "unit", "units", "suite", "office", "offices", "space", "area",
                 "floor", "level", "building", "vacant", "storage", "closet"}
# Words that put the caller at the place named next: "I'm in the ...", "calling from ...",
# "the sign says ...". Checked on the clause text just before the name.
_PLACE_FRAME = re.compile(
    r"(?:(?<!\w)(?:in|at|inside|into|from|says|reads|called|labell?ed|marked)\s+(?:(?:the|my|our|a|an|his|her|their|this)\s+)?"
    r"|(?<!\w)(?:it'?s|its|it\s+is|this\s+is|that'?s|i'?m|we'?re|he'?s|she'?s|they'?re)\s+(?:(?:the|my|our|a|an)\s+)?)$",
    re.IGNORECASE,
)
_FILLER_WORD = re.compile(r"(?<!\w)(?:um+|uh+|er+|erm|hmm+|like|you\s+know)(?!\w)[,.]?\s*", re.IGNORECASE)
_NAME_LEAD_IN = re.compile(r"(?:(?<!\w)(?:um+|uh+|er+|oh|okay|ok|well|so|and|no|now|yes|yeah|the|my|our|a|an)\s*)+$",
                           re.IGNORECASE)
# After a name that opens its clause, the caller must go on about the place, not say
# something else about it ("the elevator is stuck" does not put the caller in the elevator).
_NAME_FOLLOW_OK = re.compile(
    r"^\s*(?:$|[.,;!?-]|(?:office|offices|suite|room|place|area|downstairs|upstairs|on|in|at|by|near|next|just|right|off"
    r"|across|here|please|hurry|i|we|it'?s|and)(?!\w))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NameMention:
    """A room or tenant name of this building found in the caller's words."""

    phrase: str  # normalised words of the name that matched
    room_ids: tuple[str, ...]
    span: str
    start: int
    end: int


def _name_words(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"'s(?!\w)", "", text)
    return [w for w in re.split(r"[^a-z0-9]+", text) if w and w not in _NAME_STOP]


def _phrases_of(name: str, tenant: bool = False) -> dict[tuple[str, ...], bool]:
    """Ways a caller may say a room's (or tenant's) name: ``word tuple -> is the whole name``.

    Whole names: the name without parentheses, each side of a "/", and a tenant name
    without its corporate suffix. Shortened forms (not whole): a part of an "&" name
    ("Supply & Copy Room 3" -> "copy room 3") and any form without its trailing number.
    """
    base = re.sub(r"\(.*?\)", " ", name or "")
    parts: list[tuple[str, bool]] = [(base, True)]
    if "/" in base:
        parts += [(p, True) for p in re.split(r"\s*/\s*", base)]
    for p, _ in list(parts):
        conj = re.split(r"\s*(?:&|(?<!\w)and(?!\w))\s*", p)
        if len(conj) > 1 and not tenant:
            parts += [(c, False) for c in conj if len(_name_words(c)) >= 2]
    out: dict[tuple[str, ...], bool] = {}
    for p, whole in parts:
        words = _name_words(p)
        if tenant:
            while words and words[-1] in _CORPORATE_SUFFIX:
                words = words[:-1]
        if not words:
            continue
        out[tuple(words)] = out.get(tuple(words), False) or whole
        if len(words) >= 2 and any(ch.isdigit() for ch in words[-1]):
            out.setdefault(tuple(words[:-1]), False)
    return out


_name_index_cache: dict[tuple[str, int, int], dict[tuple[str, ...], tuple[str, ...]]] = {}


def name_index(building: Building, meta: Optional[dict] = None, vocab: Optional[dict] = None
               ) -> dict[tuple[str, ...], tuple[str, ...]]:
    """``phrase (word tuple) -> room ids`` for the names a caller can use to say where they are.

    Kept: phrases of two or more words, or one word of five or more letters that is not a
    generic word ("room", "other"); carried by at most ``max(4, number of floors)`` rooms
    (a name on more rooms than that is a type of room, which the vocabulary scores). A
    phrase that is a room-type word of the vocabulary ("elevator", "server room", "break
    room") is kept only when it really names rooms here: every room of that type carries it,
    or it is the whole name of exactly one room ("Boardroom", "Kitchen").
    """
    key = (building.id, len(building.rooms), id(vocab) if vocab is not None else 0)
    if meta is None and key in _name_index_cache:
        return _name_index_cache[key]
    room_meta_ = _room_meta(building) if meta is None else meta
    carried: dict[tuple[str, ...], list[str]] = {}
    whole_name: dict[tuple[str, ...], bool] = {}
    for r in building.rooms:
        phrases = _phrases_of(r.name)
        tenant = (room_meta_.get(r.id) or {}).get("tenant")
        if tenant:
            for p, whole in _phrases_of(str(tenant), tenant=True).items():
                phrases[p] = phrases.get(p, False) or whole
        for p, whole in phrases.items():
            carried.setdefault(p, []).append(r.id)
            whole_name[p] = whole_name.get(p, False) or whole
    # The last word of a name, when no other room's name has it ("the lounge" for the one
    # "Executive Lounge"): a caller's short form.
    word_rooms: dict[str, set[str]] = {}
    last_word: dict[str, str] = {}
    for r in building.rooms:
        words = _name_words(re.sub(r"\(.*?\)", " ", r.name))
        for w in words:
            word_rooms.setdefault(w, set()).add(r.id)
        if len(words) >= 2:
            last_word[r.id] = words[-1]
    for rid, w in last_word.items():
        if len(word_rooms[w]) == 1 and (w,) not in carried:
            carried[(w,)] = [rid]
            whole_name[(w,)] = False
    n_floors = len({r.floor for r in building.rooms})
    limit = max(4, n_floors)
    by_type: dict[str, set[str]] = {}
    for r in building.rooms:
        by_type.setdefault(r.room_type, set()).add(r.id)
    out: dict[tuple[str, ...], tuple[str, ...]] = {}
    for phrase, ids in carried.items():
        if len(ids) > limit:
            continue
        if len(phrase) == 1 and (len(phrase[0]) < 5 or phrase[0] in _GENERIC_NAME):
            continue
        if all(any(ch.isdigit() for ch in w) for w in phrase):
            continue  # numbers alone are the unit rule's job
        spoken = " ".join(phrase)
        clues = [c for c in extract_clues_rules(spoken, vocab=vocab) if c.type == "room_type" and c.polarity]
        if len(clues) == 1 and len(_name_words(clues[0].span)) == len(phrase):
            others = by_type.get(clues[0].value, set()) - set(ids)
            if others and not (len(ids) == 1 and whole_name[phrase]):
                continue  # a type word: other rooms of that type go by other names
        out[phrase] = tuple(ids)
    if meta is None:
        _name_index_cache[key] = out
    return out


def find_names(building: Building, text: str, *, meta: Optional[dict] = None, vocab: Optional[dict] = None,
               where_reply: bool = False, taken: Sequence[tuple[int, int]] = ()) -> list[NameMention]:
    """Room and tenant names of this building that the caller gives as their own place.

    Longest name first, no overlaps (also not with ``taken`` spans, e.g. unit mentions). A
    name counts only in a frame that puts the caller there: "I'm in the ...", "we're at ...",
    "calling from ...", "the sign says ...", or opening its clause ("the executive lounge",
    a reply to "where are you?"). Negated mentions, places named relative to the caller
    ("just off the main lobby") and clauses about the hazard ("the fire's in the server
    room") are left out.
    """
    index = name_index(building, meta, vocab)
    if not index or not (text or "").strip():
        return []
    low = (text or "").lower().replace("’", "'")
    if len(low) != len(text):  # offsets must line up with the caller's own text
        return []
    said = set(_name_words(low))
    found: list[NameMention] = []
    spans: list[tuple[int, int]] = list(taken)
    for phrase in sorted(index, key=lambda p: (-len(" ".join(p)), p)):
        if not said.issuperset(phrase):
            continue
        body = r"(?:'s)?[^a-z0-9;!?]+(?:(?:the|a|an|of|and)[^a-z0-9;!?]+)*".join(re.escape(w) for w in phrase)
        for m in re.finditer(rf"(?<![a-z0-9]){body}(?![a-z0-9])", low):
            if any(not (m.end() <= s or m.start() >= e) for s, e in spans):
                continue
            if _negated(text, m.start()) or _someone_elses_place(text, m.start()):
                continue
            head = _FILLER_WORD.sub("", _clause_head(text, m.start()))
            framed = bool(_PLACE_FRAME.search(head))
            opens = not _NAME_LEAD_IN.sub("", head).strip() and bool(_NAME_FOLLOW_OK.match(text[m.end():]))
            if not framed and not opens:
                continue
            if len(phrase) == 1 and not framed and not where_reply:
                continue  # one bare word needs a frame, or a question it answers
            spans.append((m.start(), m.end()))
            found.append(NameMention(" ".join(phrase), index[phrase], text[m.start():m.end()], m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def name_statements(building: Building, mention: NameMention, text: str = "") -> DialogStatement:
    """Grounded evidence for a room or tenant name: the rooms that carry the name."""
    rating = rate_statement(text, mention.start, mention.end) if text else PLAIN
    return DialogStatement(
        confidence=rating.share(), span=mention.span, selector=RoomSelector(room_ids=list(mention.room_ids)),
        start=mention.start, end=mention.end,
        rationale=f"'{mention.span}' is the name of {_room_names(building, list(mention.room_ids))}",
        rating=rating,
    )


# ---------------------------------------------------------------------------
# 3. Corrections
# ---------------------------------------------------------------------------

_CORRECTION = re.compile(
    r"(?<!\w)(?:no+\s*,?\s*wait|wait\s*,?\s*no|sorry\s*,?\s*(?:i\s+)?mean[t]?|sorry\s*,?\s*(?:it'?s|i'?m|we'?re)"
    r"|i\s+mean[t]?|i\s+meant\s+to\s+say|actually|correction|scratch\s+that|my\s+mistake|my\s+bad"
    r"|let\s+me\s+correct\s+(?:that|myself)|not\s+\w+\s*,?\s*(?:but|i\s+mean))(?!\w)",
    re.IGNORECASE,
)


def correction_cue(text: str) -> Optional[int]:
    """Offset just after the first correction cue in ``text``, or None."""
    m = _CORRECTION.search(text or "")
    return m.end() if m else None


# "sorry I meant 660", "no wait, it's 402": the corrected value right after the cue.
_CORRECTED_NUMBER = re.compile(
    r"^\s*[,.:-]?\s*(?:(?:it'?s|its|it\s+is|it\s+was|i'?m\s+in|we'?re\s+in|i\s+am\s+in|we\s+are\s+in)\s+)?"
    r"(?:(?:the\s+)?number\s+)?#?\s*(?P<unit>\d{2,5}[A-Za-z]?|\d[A-Za-z]|[A-Za-z]\d{1,4})(?!\w)" + _NOT_A_UNIT_AFTER,
    re.IGNORECASE,
)


def correction_unit(
    transcript: Sequence[TranscriptTurn], index: int, vocab: dict | None = None
) -> Optional[UnitMention]:
    """A bare number right after a correction cue, read as a unit when a unit is what is
    being corrected: an earlier caller turn (or this turn before the cue) named a unit, or
    the dispatcher's open question was about the unit. Single digits ("I meant 4") are left
    alone; they are as likely a floor.
    """
    text = transcript[index].text or ""
    at = correction_cue(text)
    if at is None:
        return None
    m = _CORRECTED_NUMBER.match(text[at:])
    if not m:
        return None
    start, end = at + m.start("unit"), at + m.end("unit")
    context = bool(find_units(text[: at]))
    if not context:
        context = any(t.speaker == "caller" and find_units(t.text or "") for t in transcript[:index])
    if not context:  # the dispatcher asked for the unit (console question or their own words)
        context = any(t.speaker == "dispatcher" and "unit" in reply_context(transcript, j + 1)
                      for j, t in enumerate(transcript[:index]))
    if not context:
        return None
    return UnitMention(m.group("unit"), text[start:end], start, end)


def evidence_slots(e: Evidence | DialogStatement) -> set[str]:
    """What an evidence item is *about*, for corrections.

    vocab: the clue type (a denial counts for the section it denies: "negative" with
    ``floors:3`` is slot "floor"). grounded: "unit" (+ "floor", a unit names a floor, and
    "place") for a unit number; "place" (+ "selector") for rooms named by id, name or
    tenant, so "the server room - no wait, the boardroom" replaces the first with the
    second; else "floor" for floor_labels, "room_type" for room_types, else "selector".
    """
    clue, sel = e.clue, e.selector
    if clue is not None:
        if clue.type == "negative":
            section = clue.value.split(":", 1)[0] if ":" in clue.value else ""
            return {TYPE_FOR_SECTION.get(section, "negative")}
        return {clue.type}
    if sel is None:
        return set()
    if sel.unit is not None:
        return {"unit", "floor", "place"}
    if sel.room_ids is not None or sel.name_contains is not None or sel.tenant_contains is not None:
        return {"place", "selector"}
    if sel.floor_labels is not None:
        return {"floor"}
    if sel.room_types is not None:
        return {"room_type"}
    return {"selector"}


def superseded_ids(
    new_items: Sequence[tuple[Evidence, int]],
    earlier: Iterable[Evidence],
    correction_at: Optional[int],
) -> list[str]:
    """Ids a correction replaces.

    ``new_items`` are this turn's evidence with their text offsets (-1 unknown). Items
    that start at or after ``correction_at`` are corrections; every earlier item sharing
    a slot with one of them (``earlier`` from previous turns, plus this turn's items
    before the cue) is superseded. Rejected items stay rejected and are not listed.
    """
    if correction_at is None:
        return []
    corrections = [e for e, start in new_items if start >= correction_at]
    if not corrections:
        return []
    slots: set[str] = set()
    for e in corrections:
        slots |= evidence_slots(e)
    candidates = list(earlier) + [e for e, start in new_items if 0 <= start < correction_at]
    out: list[str] = []
    for e in candidates:
        if e.status == "rejected" or e.id in out:
            continue
        if evidence_slots(e) & slots:
            out.append(e.id)
    return out


# ---------------------------------------------------------------------------
# Asked questions
# ---------------------------------------------------------------------------


def asked_keys(transcript: Sequence[TranscriptTurn]) -> set[str]:
    """Question keys that count as asked: every logged question and every button answer
    (including "Not sure"), so ``next_question`` does not offer them again."""
    keys: set[str] = set()
    for t in transcript:
        if t.question is not None:
            keys.add(t.question.id)
            if t.question.clue_type and t.question.value:
                keys.add(f"{t.question.clue_type}:{t.question.value}")
        if t.answer is not None:
            keys.add(f"{t.answer.clue_type}:{t.answer.value}")
    return keys


def _confirmed_place(building: Building, transcript: Sequence[TranscriptTurn], index: int, labels: set[str], *,
                     meta: dict | None = None, vocab: dict | None = None) -> Optional[DialogStatement]:
    """"Are you in the security office?" - "yes": the one room number or name the dispatcher's
    question named, confirmed by the caller. None unless the reply opens with a yes and the
    question names exactly one place of this building."""
    text = transcript[index].text or ""
    reply = classify_reply(text)
    if reply is None or reply[0] != "yes":
        return None
    question = ""
    for j in range(index - 1, -1, -1):
        t = transcript[j]
        if t.speaker == "system":
            continue
        question = (t.text or "") if t.speaker == "dispatcher" else ""
        break
    if not question.strip().endswith("?"):
        return None
    units = find_units(question, labels=labels)
    named = find_names(building, question, meta=meta, vocab=vocab, where_reply=True,
                       taken=[(f.start, f.end) for f in units])
    if len(units) + len(named) != 1:
        return None
    start, end = reply[1], reply[2]
    span = text[start:end]
    confidence = INFERRED_QUESTION_CONFIDENCE * (HEDGED_YES_FACTOR if _HEDGED_YES.fullmatch(span.strip()) else 1.0)
    if units:
        base = unit_statements(building, units[0], meta=meta)[0]
        if not selector_hits(building, base.selector, meta):
            return None
        selector, what = base.selector, units[0].span
    else:
        selector, what = RoomSelector(room_ids=list(named[0].room_ids)), named[0].span
    return DialogStatement(
        confidence=round(confidence, 3), span=span, selector=selector, start=start, end=end,
        rationale=f"'{span}' answers the dispatcher's question about {what}",
    )


# ---------------------------------------------------------------------------
# All rules for one turn
# ---------------------------------------------------------------------------


def read_dialog(
    building: Building,
    transcript: Sequence[TranscriptTurn],
    index: int,
    vocab: dict | None = None,
    *,
    meta: dict | None = None,
) -> DialogReading:
    """Run the dialog rules on caller turn ``transcript[index]``.

    ``meta`` overrides the building's sidecar room metadata (tests). Corrections are not
    resolved here (the caller needs the ids): ``correction_at`` says where they start.
    """
    turn = transcript[index]
    text = turn.text or ""
    reading = DialogReading(correction_at=correction_cue(text))

    key, answer, stmt = resolve_answer(transcript, index, vocab)
    reading.question_key, reading.answer = key, answer
    if stmt is not None:
        reading.statements.append(stmt)

    asked = reply_context(transcript, index)
    labels: set[str] = set().union(*room_labels(building, meta).values())
    mentions = find_units(text, labels=labels, where_reply="where" in asked)
    corrected = correction_unit(transcript, index, vocab)
    if corrected is not None and not any(
            not (corrected.end <= f.start or corrected.start >= f.end) for f in mentions):
        mentions = sorted(mentions + [corrected], key=lambda f: f.start)
    for mention in mentions:
        reading.statements.extend(unit_statements(building, mention, meta=meta, text=text))
    names = find_names(building, text, meta=meta, vocab=vocab, where_reply="where" in asked,
                       taken=[(f.start, f.end) for f in mentions])
    for name in names:
        reading.statements.append(name_statements(building, name, text))
    if not mentions and not names and turn.answer is None:
        confirmed = _confirmed_place(building, transcript, index, labels, meta=meta, vocab=vocab)
        if confirmed is not None:
            reading.statements.append(confirmed)
    if not mentions and not names and stmt is None and turn.answer is None:
        for partition in ("unit", "floor"):  # a bare "860" / "third" answering the question just asked
            if partition in asked:
                found = _open_answer_statements(building, text, partition, meta=meta)
                if found:
                    reading.statements.extend(found)
                    break
    return reading

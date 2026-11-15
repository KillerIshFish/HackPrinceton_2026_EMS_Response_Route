"""How sure and how exact a specific caller statement sounds.

A specific statement says where the caller is by a room, unit or suite number, a room or
tenant name, or a floor. Its weight in fusion is the stated share
``core.fusion.stated_share(certainty, specificity)`` (0.40 to 0.95, ``configs/statements.yaml``).
Two readers give the two ratings, and they differ only in who rates:

* model modes ("llm" / "agent", ``parser.agent``): the model gives certainty and specificity
  on each ``record_location`` / ``record_floor`` it makes; ``clean_rating`` checks what it wrote.
* rules mode: ``rate_statement`` estimates the same two things from fixed word patterns.

Ratings are numbers in [0, 1]:

``certainty``    1.0 a plain statement ("I'm in room 302"); 0.5 one hedge ("I think",
                 "probably") or a questioning tone ("third floor?"); 0.25 real doubt
                 ("maybe", "might be", "not sure"); these multiply, so "I think maybe the
                 third floor?" is about 0.06.
``specificity``  1.0 a number or a name given in full; 0.5 an approximate one ("somewhere
                 on three", "around 302", "the third or fourth floor").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from core.fusion import StatementSettings, stated_share, statement_settings

# Rule ratings (see the module docstring). Hand-set judgements, like every number in
# configs/statements.yaml: one hedge or a questioning tone halves certainty, real doubt quarters it,
# an approximate place halves specificity. They match the anchors the record tools give the model
# (``parser.building_tools._CERTAINTY`` / ``_SPECIFICITY``: "about 0.5" for one hedge or approximate).
MILD_HEDGE_FACTOR = 0.5
STRONG_HEDGE_FACTOR = 0.25
QUESTION_FACTOR = 0.5
APPROXIMATE_SPECIFICITY = 0.5

_MILD_HEDGE = re.compile(
    r"(?<!\w)(?:i\s+think|i\s+guess|i\s+believe|i\s+suppose|i'?d\s+say|probably|pretty\s+sure|fairly\s+sure"
    r"|should\s+be|must\s+be|looks\s+like|seems\s+like|it\s+seems|kind\s+of|kinda|sort\s+of|as\s+far\s+as\s+i\s+know)(?!\w)",
    re.IGNORECASE,
)
_STRONG_HEDGE = re.compile(
    r"(?<!\w)(?:maybe|perhaps|possibly|might(?:\s+be)?|could\s+be|may\s+be|not\s+(?:really\s+|too\s+|totally\s+)?(?:sure|certain)"
    r"|unsure|(?:do\s+not|don'?t|dont)\s+(?:really\s+)?know|no\s+idea|(?:can'?t|cannot)\s+tell|can'?t\s+remember"
    r"|if\s+i\s+had\s+to\s+guess)(?!\w)",
    re.IGNORECASE,
)
# Words that make the place approximate, right before the statement ("somewhere on the third
# floor", "around 302") or right after it ("302 or so", "third floor or something", "threeish").
_APPROXIMATE_BEFORE = re.compile(
    r"(?<!\w)(?:somewhere|someplace|around|about|roughly|approximately|something\s+like|close\s+to)\s+"
    r"(?:(?:on|in|at|near|by|around)\s+)?(?:(?:the|a|my|our)\s+)?(?:[\w'-]+\s+)?$",
    re.IGNORECASE,
)
_APPROXIMATE_AFTER = re.compile(
    r"^\s*(?:floor\s+)?,?\s*(?:or\s+so|or\s+something|or\s+somewhere|-?ish(?!\w)|somewhere|something)(?!\w)",
    re.IGNORECASE,
)
_NUMBERISH = (r"(?:\d{1,5}[a-z]?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|one|two|three|four"
              r"|five|six|seven|eight|nine|ten|top|ground|\d{1,2}(?:st|nd|rd|th))")
# "the third or fourth floor", "302 or 303": the caller gave two candidates
_ALTERNATIVE_AFTER = re.compile(r"^\s*(?:floor\s+)?,?\s*or\s+(?:maybe\s+|the\s+|room\s+|suite\s+|unit\s+)*" + _NUMBERISH + r"(?!\w)",
                                re.IGNORECASE)
_ALTERNATIVE_BEFORE = re.compile(r"(?<!\w)" + _NUMBERISH + r"\s*,?\s+or\s+(?:maybe\s+|the\s+)*$", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[.,;!?]|(?<!\w)(?:but|and|so)(?!\w)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.;!?]")
_ONLY_HEDGES = re.compile(
    r"^(?:\s|[,.;!?-]|(?<!\w)(?:um+|uh+|er+|oh|well|like|so|and|but|yeah|i\s+mean)(?!\w)"
    r"|" + _MILD_HEDGE.pattern + r"|" + _STRONG_HEDGE.pattern + r")*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rating:
    """How sure the caller sounds and how exactly the words pin the place down, each in [0, 1]."""

    certainty: float = 1.0
    specificity: float = 1.0

    def share(self, settings: Optional[StatementSettings] = None) -> float:
        """The stated share this rating maps to (``core.fusion.stated_share``)."""
        return stated_share(self.certainty, self.specificity, settings)


PLAIN = Rating(1.0, 1.0)


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Offsets of the clause holding ``text[start:end]``."""
    lo = 0
    for m in _CLAUSE_BREAK.finditer(text, 0, start):
        lo = m.end()
    nxt = _CLAUSE_BREAK.search(text, end)
    return lo, (nxt.start() if nxt else len(text))


def _neighbour_clauses(text: str, lo: int, hi: int) -> str:
    """The clauses just before and just after ``text[lo:hi]`` that hold nothing but hedges and
    fillers: "I think, uh, 302" and "302, I think." hedge the number; "I'm in 302, I think the
    fire is in 301" does not."""
    out = []
    breaks = [m for m in _CLAUSE_BREAK.finditer(text)]
    # walk back over hedge-only clauses (at most 3 each way: "uh, well, I think, 302")
    edge = lo
    for _ in range(3):
        before_lo = max((m.end() for m in breaks if m.end() < edge), default=0)
        if before_lo >= edge:
            break
        before = text[before_lo:edge]
        if not before.strip(" ,.;!?-") or not _ONLY_HEDGES.match(before):
            break
        out.append(before)
        edge = before_lo
        if edge <= 0:
            break
    # and forward
    edge = hi
    for _ in range(3):
        if edge >= len(text):
            break
        after_hi = min((m.start() for m in breaks if m.start() > edge), default=len(text))
        after = text[edge:after_hi]
        if not after.strip(" ,.;!?-") or not _ONLY_HEDGES.match(after):
            break
        out.append(after)
        edge = after_hi
    return " ".join(out)


def rate_statement(text: str, start: int, end: int) -> Rating:
    """Rule estimate for the statement at ``text[start:end]`` (rules mode; in the model modes,
    the fallback ``clean_rating`` uses when the model leaves a rating out).

    Hedges count only in the statement's own clause, or in a neighbouring clause that is
    nothing but a hedge. A question mark closing the statement's sentence counts as a
    questioning tone.
    """
    text = text or ""
    if not text or start < 0 or end > len(text) or start >= end:
        return PLAIN
    lo, hi = _clause_bounds(text, start, end)
    clause = text[lo:hi]
    around = clause + " " + _neighbour_clauses(text, lo, hi)
    certainty = 1.0
    if _MILD_HEDGE.search(around):
        certainty *= MILD_HEDGE_FACTOR
    if _STRONG_HEDGE.search(around):
        certainty *= STRONG_HEDGE_FACTOR
    closing = _SENTENCE_END.search(text, end)
    if closing is not None and closing.group() == "?":
        certainty *= QUESTION_FACTOR
    specificity = 1.0
    if (_APPROXIMATE_BEFORE.search(text[lo:start]) or _APPROXIMATE_AFTER.match(text[end:hi])
            # 40 chars back: room for "the third or" before "fourth floor"
            or _ALTERNATIVE_AFTER.match(text[end:]) or _ALTERNATIVE_BEFORE.search(text[max(0, start - 40):start])):
        specificity = APPROXIMATE_SPECIFICITY
    return Rating(round(certainty, 4), specificity)


def rate_span(text: str, span: str) -> Rating:
    """``rate_statement`` for the first place ``span`` stands in ``text`` (plain when it does not)."""
    if not span:
        return PLAIN
    start = (text or "").find(span)
    if start < 0:
        start = (text or "").lower().find(span.lower())
    if start < 0:
        return PLAIN
    return rate_statement(text, start, start + len(span))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def clean_rating(certainty: Any, specificity: Any, fallback: Rating = PLAIN,
                 settings: Optional[StatementSettings] = None) -> tuple[Rating, list[str]]:
    """The model's rating, checked: ``(rating, notes for the trace)``.

    Each value is clamped to [0, 1]. A value that is missing or not a number is replaced
    conservatively: the rule estimate for the same words (``fallback``), and never more than
    ``missing_rating`` (0.5, ``configs/statements.yaml``).
    """
    cfg = settings or statement_settings()
    notes: list[str] = []
    values = []
    for name, raw, rule in (("certainty", certainty, fallback.certainty), ("specificity", specificity, fallback.specificity)):
        x = _number(raw)
        if x is None:
            x = min(rule, cfg.missing_rating)
            notes.append(f"{name} missing: {x:g} used")
        elif not 0.0 <= x <= 1.0:
            x = min(1.0, max(0.0, x))
            notes.append(f"{name} clamped to {x:g}")
        values.append(round(x, 4))
    return Rating(values[0], values[1]), notes

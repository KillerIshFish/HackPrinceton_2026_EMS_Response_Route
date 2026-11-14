"""Clue extraction: caller utterance -> list[Clue].

v0 is a deterministic keyword/regex extractor driven entirely by
``configs/clue_vocab.yaml``. It makes NO network calls.

LLM swap seam
-------------
``extract_clues(text, use_llm=True)`` dispatches to ``extract_clues_llm``, which
is a stub that raises ``NotImplementedError`` until ``LLM_PROVIDER`` /
``LLM_API_KEY`` are set in ``.env`` and the provider call is implemented. The
system prompt for that call already lives in ``parser/prompts/extract_clues.md``
(see ``build_system_prompt``). Both paths return the same ``list[Clue]``, so
nothing downstream changes when the swap happens.

Rule-based behaviour
--------------------
* Longest synonym wins; a shorter phrase never fires inside a span that a
  longer phrase already consumed ("in the elevator" -> room_type, not landmark).
* Matching is case-insensitive on word boundaries, with an optional plural "s".
* ``Clue.span`` is the exact substring of the input (original casing).
* A negation cue shortly before a phrase, in the same clause, turns it into
  ``type="negative"``, ``value="<vocab section>:<key>"``, ``polarity=False``.
* Hedges ("I think", "maybe", ...) in the same sentence lower confidence.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.schema import Clue

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extract_clues.md"

# Clue.type -> section of clue_vocab.yaml
SECTION_FOR_TYPE: dict[str, str] = {
    "room_type": "room_types",
    "view": "views",
    "landmark": "landmarks",
    "sound": "sounds",
    "smell": "smells",
    "floor": "floors",
}
TYPE_FOR_SECTION: dict[str, str] = {v: k for k, v in SECTION_FOR_TYPE.items()}

# When two phrases have the same length, prefer the more specific clue type.
_TYPE_PRIORITY = {"floor": 0, "room_type": 1, "view": 2, "sound": 3, "smell": 4, "landmark": 5}

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_HEDGES = ("i think", "maybe", "might be", "probably", "i guess", "could be",
           "not sure", "looks like", "kind of", "sort of")
HEDGE_FACTOR = 0.7
# A bare room word ("...the office building...") is weaker evidence than a
# locative statement ("I'm in an office").
NO_LOCATIVE_FACTOR = 0.8
NEGATION_MAX_GAP_WORDS = 3

_LOCATIVE_BEFORE = re.compile(
    r"(?<!\w)(?:in|inside|into|at|from|of|near|by)\s+(?:[\w'-]+\s+){0,3}$", re.IGNORECASE
)
_LOCATIVE_PHRASE_START = re.compile(r"^(?:in|inside|on|my|stuck)\s", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(
    r"[.,;:!?]|(?<!\w)but(?!\w)|(?<!\w)and\s+(?:i|i'm|we|we're|there|there's|it|it's|he|she|they)(?!\w)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[.;!?]")
# "no windows or doors": a negation carries across "or" / "nor" to the next phrase.
_NEGATION_CARRY = re.compile(r"^\s*,?\s*(?:or|nor)\s+(?:(?:a|an|any|the)\s+)?$", re.IGNORECASE)
# "office" in "office building / office park" names the building, not the room.
_NOT_A_ROOM_AFTER = re.compile(r"^\s+(?:building|park|complex|tower|block)(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=4)
def _load_vocab_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_vocab(path: str | os.PathLike | None = None) -> dict:
    """Load ``configs/clue_vocab.yaml`` (cached)."""
    return _load_vocab_cached(str(path or CONFIG_DIR / "clue_vocab.yaml"))


# ---------------------------------------------------------------------------
# Candidate matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    type: str  # Clue.type (never "negative" at this stage)
    value: str

    @property
    def length(self) -> int:
        return self.end - self.start


def _normalize(text: str) -> str:
    """Length-preserving normalisation so match indices map back to `text`."""
    return text.replace("’", "'").replace("‘", "'")


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in phrase.strip().split()]
    body = r"\s+".join(words)
    plural = r"(?:s)?" if phrase[-1:].isalpha() and not phrase.endswith("s") else ""
    return re.compile(rf"(?<!\w){body}{plural}(?!\w)", re.IGNORECASE)


def _vocab_candidates(norm: str, vocab: dict) -> list[_Candidate]:
    out: list[_Candidate] = []
    for clue_type, section in SECTION_FOR_TYPE.items():
        if clue_type == "floor":
            continue
        for key, entry in (vocab.get(section) or {}).items():
            for syn in (entry or {}).get("synonyms") or [str(key).replace("_", " ")]:
                for m in _phrase_regex(str(syn)).finditer(norm):
                    if clue_type == "room_type" and _NOT_A_ROOM_AFTER.match(norm[m.end():]):
                        continue
                    out.append(_Candidate(m.start(), m.end(), clue_type, str(key)))
    return out


def _floor_candidates(norm: str, vocab: dict) -> list[_Candidate]:
    floors = vocab.get("floors") or {}
    ordinals = {str(k).lower(): int(v) for k, v in (floors.get("ordinals") or {}).items()}
    out: list[_Candidate] = []

    # "fourth floor", "4th floor", "22nd level"
    ord_alt = "|".join(sorted((re.escape(k) for k in ordinals), key=len, reverse=True))
    ord_alt = (ord_alt + "|" if ord_alt else "") + r"\d{1,2}(?:st|nd|rd|th)"
    for m in re.finditer(
        rf"(?<!\w)({ord_alt})\s+(?:floor|level|story|storey)(?!\w)", norm, re.IGNORECASE
    ):
        tok = m.group(1).lower()
        n = ordinals.get(tok)
        if n is None:
            n = int(re.match(r"\d+", tok).group())
        out.append(_Candidate(m.start(), m.end(), "floor", str(n)))

    # "floor 4", "level four", "floor number 3"
    num_alt = r"\d{1,2}|" + "|".join(_NUMBER_WORDS)
    for m in re.finditer(
        rf"(?<!\w)(?:floor|level)\s+(?:number\s+)?({num_alt})(?!\w)", norm, re.IGNORECASE
    ):
        tok = m.group(1).lower()
        n = int(tok) if tok.isdigit() else _NUMBER_WORDS[tok]
        out.append(_Candidate(m.start(), m.end(), "floor", str(n)))

    # "top floor", "ground level" -> relative labels resolved by clue_likelihood
    for rel, phrases in (floors.get("relative") or {}).items():
        for phrase in phrases or []:
            for m in _phrase_regex(str(phrase)).finditer(norm):
                out.append(_Candidate(m.start(), m.end(), "floor", str(rel)))
    return out


def _select_non_overlapping(cands: list[_Candidate]) -> list[_Candidate]:
    """Longest-first greedy selection; later candidates may not overlap a winner."""
    ordered = sorted(cands, key=lambda c: (-c.length, _TYPE_PRIORITY[c.type], c.start))
    taken: list[_Candidate] = []
    for c in ordered:
        if all(c.end <= t.start or c.start >= t.end for t in taken):
            taken.append(c)
    return sorted(taken, key=lambda c: c.start)


# ---------------------------------------------------------------------------
# Negation / confidence
# ---------------------------------------------------------------------------


def _negation_regex(vocab: dict) -> re.Pattern[str]:
    cues = [str(c).strip() for c in (vocab.get("negation_cues") or []) if str(c).strip()]
    if not cues:
        cues = ["no", "not"]
    alts = sorted({r"\s+".join(re.escape(w) for w in c.split()) for c in cues}, key=len, reverse=True)
    return re.compile(rf"(?<!\w)(?:{'|'.join(alts)})(?!\w)", re.IGNORECASE)


def _is_negated(norm: str, cand: _Candidate, neg_re: re.Pattern[str]) -> bool:
    """True if a negation cue precedes the phrase closely, within the same clause."""
    window_start = max(0, cand.start - 60)
    for m in neg_re.finditer(norm, window_start, cand.start):
        gap = norm[m.end():cand.start]
        if not gap[:1].isspace() and gap != "":
            continue  # "No," / "not-" : the cue is not attached to what follows
        if _CLAUSE_BREAK.search(gap):
            continue
        if len(gap.split()) <= NEGATION_MAX_GAP_WORDS:
            return True
    return False


def _sentence_of(norm: str, cand: _Candidate) -> str:
    start = 0
    for m in _SENTENCE_BREAK.finditer(norm, 0, cand.start):
        start = m.end()
    nxt = _SENTENCE_BREAK.search(norm, cand.end)
    return norm[start: nxt.start() if nxt else len(norm)].lower()


def _confidence(norm: str, cand: _Candidate, negated: bool, vocab: dict) -> float:
    defaults = vocab.get("default_confidence") or {}
    conf = float(defaults.get("negative" if negated else cand.type, 0.7))
    if cand.type == "room_type" and not negated:
        phrase = norm[cand.start:cand.end]
        located = bool(_LOCATIVE_PHRASE_START.match(phrase)) or bool(
            _LOCATIVE_BEFORE.search(norm[max(0, cand.start - 40):cand.start])
        )
        if not located:
            conf *= NO_LOCATIVE_FACTOR
    sentence = _sentence_of(norm, cand)
    if any(h in sentence for h in _HEDGES):
        conf *= HEDGE_FACTOR
    return round(min(1.0, max(0.0, conf)), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_clues_rules(text: str, vocab: dict | None = None) -> list[Clue]:
    """Deterministic keyword/regex extraction. Clues are returned in text order."""
    if not text or not text.strip():
        return []
    vocab = vocab or load_vocab()
    norm = _normalize(text)
    neg_re = _negation_regex(vocab)

    winners = _select_non_overlapping(_vocab_candidates(norm, vocab) + _floor_candidates(norm, vocab))

    clues: list[Clue] = []
    seen: set[tuple[str, str]] = set()
    prev_end, prev_negated = 0, False
    for cand in winners:
        negated = _is_negated(norm, cand, neg_re) or (
            prev_negated and bool(_NEGATION_CARRY.match(norm[prev_end:cand.start]))
        )
        prev_end, prev_negated = cand.end, negated
        if negated:
            ctype, value = "negative", f"{SECTION_FOR_TYPE[cand.type]}:{cand.value}"
        else:
            ctype, value = cand.type, cand.value
        if (ctype, value) in seen:
            continue
        seen.add((ctype, value))
        clues.append(
            Clue(
                type=ctype,  # type: ignore[arg-type]
                value=value,
                polarity=not negated,
                confidence=_confidence(norm, cand, negated, vocab),
                span=text[cand.start:cand.end],
            )
        )
    return clues


def build_system_prompt(vocab: dict | None = None) -> str:
    """System prompt for the future LLM extractor with the vocabulary injected."""
    vocab = vocab or load_vocab()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    compact: dict[str, Any] = {}
    for section in ("room_types", "views", "landmarks", "sounds", "smells"):
        compact[section] = {
            str(k): list((v or {}).get("synonyms") or []) for k, v in (vocab.get(section) or {}).items()
        }
    compact["floors"] = {"relative": list(((vocab.get("floors") or {}).get("relative") or {}).keys())}
    return template.replace("{{VOCAB}}", yaml.safe_dump(compact, sort_keys=False, width=100).strip())


def extract_clues_llm(text: str, vocab: dict | None = None, provider: str | None = None) -> list[Clue]:
    """LLM-backed extraction. NOT IMPLEMENTED in v0 (no network calls).

    To implement: send ``build_system_prompt(vocab)`` + ``text`` to the provider
    named by ``LLM_PROVIDER`` using ``LLM_API_KEY``, parse the JSON array into
    ``Clue`` objects, drop any clue whose ``span`` is not a substring of ``text``
    or whose ``value`` is outside the vocabulary, and fall back to
    ``extract_clues_rules`` on any error.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower() or "<unset>"
    raise NotImplementedError(
        f"LLM clue extraction (provider={provider}) is not implemented in v0. "
        "Set LLM_PROVIDER and LLM_API_KEY in .env (see .env.example), then implement "
        "parser.extract.extract_clues_llm. Until then call extract_clues(text) without "
        "use_llm=True to use the rule-based extractor."
    )


def extract_clues(text: str, vocab: dict | None = None, use_llm: bool = False) -> list[Clue]:
    """Extract structured clues from one caller utterance.

    Rule-based by default. The LLM path is only taken when ``use_llm=True`` is
    passed explicitly (env vars alone never trigger a network call).
    """
    if use_llm:
        return extract_clues_llm(text, vocab=vocab, provider=os.getenv("LLM_PROVIDER"))
    return extract_clues_rules(text, vocab=vocab)

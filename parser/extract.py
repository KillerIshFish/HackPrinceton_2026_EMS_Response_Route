"""Clue extraction: caller utterance -> list[Clue].

Two extractors return the same ``list[Clue]``:

* ``extract_clues_rules``: deterministic keyword/regex extraction driven by
  ``configs/clue_vocab.yaml``. No network. This is the default everywhere.
* ``extract_clues_llm``: Claude Haiku 5.5 with the system prompt in
  ``parser/prompts/extract_clues.md`` and a JSON-schema-constrained reply. It
  needs ``LLM_PROVIDER=anthropic`` and ``LLM_API_KEY`` in ``.env`` and is only
  used when a caller passes ``use_llm=True``. Replies are validated (span must
  be verbatim in the utterance, value must be in the vocabulary), cached on
  disk, and any failure falls back to the rules.

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

import hashlib
import json
import os
import re
import threading
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


# ---------------------------------------------------------------------------
# LLM extraction (Anthropic). Spend is kept small on purpose:
#   * every (model, prompt, utterance) result is cached on disk, so a scripted
#     demo turn costs one request the first time and nothing afterwards;
#   * thinking is off, effort is low and the reply is capped at a few hundred tokens;
#   * a per-process request budget (LLM_MAX_CALLS) falls back to the rules when hit;
#   * token usage is appended to data/processed/llm_usage.json so it can be audited.
# ---------------------------------------------------------------------------

LLM_MODEL = "claude-haiku-5-5"
LLM_MAX_OUTPUT_TOKENS = 600
LLM_TIMEOUT_S = 20.0
LLM_MIN_CONFIDENCE = 0.4
_REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_CACHE_PATH = _REPO_ROOT / "data" / "processed" / "llm_clue_cache.json"
LLM_USAGE_PATH = _REPO_ROOT / "data" / "processed" / "llm_usage.json"

_CLUE_TYPES = ["room_type", "floor", "landmark", "view", "sound", "smell", "negative"]
_CLUES_SCHEMA = {
    "type": "object",
    "properties": {
        "clues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": _CLUE_TYPES},
                    "value": {"type": "string"},
                    "polarity": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "span": {"type": "string"},
                },
                "required": ["type", "value", "polarity", "confidence", "span"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clues"],
    "additionalProperties": False,
}

_llm_lock = threading.Lock()
_llm_calls_made = 0


class LLMUnavailable(RuntimeError):
    """The LLM path cannot be used (no key, wrong provider, budget spent, API error)."""


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def llm_usage() -> dict:
    """Running totals of LLM requests and tokens: ``{"calls", "input_tokens", "output_tokens", ...}``."""
    return _read_json(LLM_USAGE_PATH, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                       "cache_read_input_tokens": 0, "cache_hits": 0})


def _known_value(clue_type: str, value: str, vocab: dict) -> bool:
    """True when ``value`` is inside the controlled vocabulary for ``clue_type``."""
    if clue_type == "floor":
        return bool(re.fullmatch(r"-?\d{1,3}|[A-Za-z]\d{0,2}", value)) or value in ((vocab.get("floors") or {}).get("relative") or {})
    if clue_type == "negative":
        section, _, key = value.partition(":")
        if section == "floors":
            return _known_value("floor", key, vocab)
        return section in TYPE_FOR_SECTION and key in (vocab.get(section) or {})
    return value in (vocab.get(SECTION_FOR_TYPE.get(clue_type, "")) or {})


def _validate_llm_clues(text: str, raw_clues: list[dict], vocab: dict) -> list[Clue]:
    """Keep only clues whose span is verbatim in ``text`` and whose value is in the vocabulary."""
    out: list[Clue] = []
    seen: set[tuple[str, str]] = set()
    lowered = text.lower()
    for raw in raw_clues:
        ctype = str(raw.get("type", ""))
        value = str(raw.get("value", "")).strip()
        span = str(raw.get("span", ""))
        if ctype not in _CLUE_TYPES or not span or not _known_value(ctype, value, vocab):
            continue
        idx = text.find(span)
        if idx < 0:  # recover the caller's own casing if only the case differs
            idx = lowered.find(span.lower())
            if idx < 0:
                continue
            span = text[idx: idx + len(span)]
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if confidence < LLM_MIN_CONFIDENCE or (ctype, value) in seen:
            continue
        seen.add((ctype, value))
        polarity = False if ctype == "negative" else bool(raw.get("polarity", True))
        out.append(Clue(type=ctype, value=value, polarity=polarity, confidence=round(confidence, 3), span=span))  # type: ignore[arg-type]
    return out


def extract_clues_llm(text: str, vocab: dict | None = None, provider: str | None = None) -> list[Clue]:
    """Claude-backed extraction of one caller utterance. Raises ``LLMUnavailable`` on any failure.

    Results are cached on disk by (model, system prompt, utterance), so repeating
    an utterance never sends a second request.
    """
    global _llm_calls_made
    if not text or not text.strip():
        return []
    vocab = vocab or load_vocab()
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider != "anthropic":
        raise LLMUnavailable(f"LLM_PROVIDER is '{provider or 'unset'}'; only 'anthropic' is implemented")
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        raise LLMUnavailable("LLM_API_KEY is not set in .env")

    system_prompt = build_system_prompt(vocab)
    cache_key = hashlib.sha256(f"{LLM_MODEL}\x00{system_prompt}\x00{text}".encode()).hexdigest()
    with _llm_lock:
        cache = _read_json(LLM_CACHE_PATH, {})
        if cache_key in cache:
            usage = llm_usage()
            usage["cache_hits"] = usage.get("cache_hits", 0) + 1
            _write_json(LLM_USAGE_PATH, usage)
            return _validate_llm_clues(text, cache[cache_key]["clues"], vocab)
        budget = int(os.getenv("LLM_MAX_CALLS", "200"))
        if _llm_calls_made >= budget:
            raise LLMUnavailable(f"LLM request budget for this process is spent ({budget}; raise LLM_MAX_CALLS)")
        _llm_calls_made += 1

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise LLMUnavailable("the 'anthropic' package is not installed (run `uv sync`)") from exc

    client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=1)
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            thinking={"type": "disabled"},
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Utterance: `{text}`"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _CLUES_SCHEMA}},
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("Anthropic rejected LLM_API_KEY") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("Anthropic rate limit reached") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("Anthropic API unreachable") from exc

    if response.stop_reason != "end_turn":
        raise LLMUnavailable(f"model stopped with stop_reason={response.stop_reason}")
    body = next((b.text for b in response.content if b.type == "text"), "")
    try:
        raw_clues = json.loads(body)["clues"]
    except (ValueError, KeyError, TypeError) as exc:
        raise LLMUnavailable("model reply was not the expected JSON") from exc

    with _llm_lock:
        cache = _read_json(LLM_CACHE_PATH, {})
        cache[cache_key] = {"text": text, "model": LLM_MODEL, "clues": raw_clues}
        _write_json(LLM_CACHE_PATH, cache)
        usage = llm_usage()
        usage["calls"] = usage.get("calls", 0) + 1
        usage["input_tokens"] = usage.get("input_tokens", 0) + (response.usage.input_tokens or 0)
        usage["output_tokens"] = usage.get("output_tokens", 0) + (response.usage.output_tokens or 0)
        usage["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0) + (
            response.usage.cache_read_input_tokens or 0
        )
        usage["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens", 0) + (
            response.usage.cache_creation_input_tokens or 0
        )
        _write_json(LLM_USAGE_PATH, usage)
    return _validate_llm_clues(text, raw_clues, vocab)


def extract_clues_with_source(text: str, vocab: dict | None = None, use_llm: bool = False) -> tuple[list[Clue], str]:
    """``(clues, source)`` where source is ``"llm"``, ``"rules"`` or ``"rules (llm unavailable: ...)"``."""
    if use_llm:
        try:
            return extract_clues_llm(text, vocab=vocab), "llm"
        except LLMUnavailable as exc:
            return extract_clues_rules(text, vocab=vocab), f"rules (llm unavailable: {exc})"
    return extract_clues_rules(text, vocab=vocab), "rules"


def extract_clues(text: str, vocab: dict | None = None, use_llm: bool = False) -> list[Clue]:
    """Extract structured clues from one caller utterance.

    Rule-based by default. The LLM path is only taken when ``use_llm=True`` is
    passed explicitly (env vars alone never trigger a network call), and it
    falls back to the rules if the model cannot be reached.
    """
    return extract_clues_with_source(text, vocab=vocab, use_llm=use_llm)[0]

"""Reasoning interpreter: a Claude model reads the call so far against this building's layout.

Entry point (parser/interpret.py, modes "llm" and "agent")::

    interpret(building, transcript, index, *, prior_evidence=(), vocab=None) -> Interpretation

How it works

* The model (``agent_model()``: ``LLM_MODEL`` in the environment, default ``DEFAULT_AGENT_MODEL``)
  gets one cached system block: the briefing (``parser/prompts/reason_call.md``) and the
  building's layout map (``parser.building_tools.layout_roster``, built by ``parser.layout``):
  every hallway with its rooms in door order and what is at each end, and per room its size and
  size rank, position on the floor plate (corner, outer edge, interior), the hallway it opens
  onto, walking metres to stairs / elevator / entrance, windows, neighbours, suite and tenant.
  There is no controlled vocabulary and no word list: the model works out from the layout which
  rooms whatever the caller says fits ("the last door past the stairs", "the big room in the
  corner", "I can hear the elevator").
* Tools (``parser.building_tools.TOOLS``): ``record_location`` for a place the caller named
  (number, name, tenant, sign; weighted as a specific statement), ``record_clue`` for any other
  clue (the rooms it fits plus how well the other rooms still fit; a soft, descriptive
  statement), ``record_floor``, ``record_fact``, ``suggest_incident_type``, ``supersede``, and
  the look-ups ``find_rooms``, ``room_details``, ``rooms_near``.
* One request per caller turn in the usual case; a second only for a look-up or a record that
  cannot be kept as it stands. Never more than ``AGENT_MAX_REQUESTS_PER_TURN`` (4).
* No word rules read the turn first. The one deterministic input kept is the dispatcher's
  Yes / No / Not sure buttons (``parser.dialog.resolve_answer`` with source "dispatcher").
* Validation in code after the loop: the model cannot invent. A record is dropped (and the
  trace says why) when its span is not in the newest caller turn word for word, a room id
  does not exist, a selector or clue matches no room, a floor is not one of the building's,
  confidence is below 0.4, or a fact asserts something as verified / safe.
* Weights. A named place carries the stated share for how the caller said it (``certainty``,
  ``specificity`` -> ``core.fusion.stated_share``, 0.40 to 0.95) when the span carries the
  number or a distinctive name word of every selected room (``_NameCheck``); otherwise it is
  capped at 0.7. A floor the caller stated carries the stated share; one the model worked out,
  0.7 at most. A clue is a descriptive factor (at most 10 to 1 between two rooms) with the
  model's confidence, capped at ``CLUE_CONFIDENCE_CAP``. Nothing ever zeroes a room.
* Spend: raw records are cached on disk keyed by model, settings, system prompt, tools and the
  user message. Token use goes to ``data/processed/llm_usage.json``. ``LLM_MAX_CALLS`` caps the
  requests of one process only when set.
* Settings (``model_settings``): ``LLM_MODEL``, ``LLM_THINKING`` ("adaptive"; "disabled" is
  honoured only on models that accept it), ``LLM_EFFORT`` and ``LLM_STRICT_TOOLS``. On models
  that support it, server-side refusal fallbacks are on (``fallbacks: "default"``).
* Failure: any API error, timeout, refusal, truncated reply, missing key or budget stop before
  the first reply falls back to the rule path with source ``"rules (agent unavailable:
  <reason>)"``. A failure after at least one reply keeps the records of the completed rounds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import parser.extract as _extract
from core.evidence import normalize_unit, room_labels
from core.fusion import CONCLUSION_TAG, is_conclusion
from core.schema import (
    INCIDENT_TYPES,
    AgentTraceEntry,
    Building,
    Clue,
    Evidence,
    Fact,
    Interpretation,
    RoomSelector,
    TranscriptTurn,
)
from parser.dialog import DialogReading, DialogStatement, rating_note, resolve_answer, superseded_ids
from parser.statement_rating import clean_rating, rate_span
from parser.building_tools import (
    LOOKUP_TOOLS,
    PREMISE_KEYS,
    RATE_TOOLS,
    RECORD_TOOLS,
    TOOLS,
    BuildingIndex,
    _STOPWORDS,
    _tokens,
    clue_room_ids,
    layout_roster,
    record_ack,
    run_lookup,
    select_rooms,
    selector_is_empty,
    unknown_room_ids,
)

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "reason_call.md"
_REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_CACHE_PATH = _REPO_ROOT / "data" / "processed" / "llm_agent_cache.json"

AGENT_MAX_REQUESTS_PER_TURN = 4
# Seconds one request may take before the turn falls back (or keeps what earlier rounds recorded).
AGENT_TIMEOUT_S = 45.0
AGENT_MIN_CONFIDENCE = 0.4
# Cap on a denial ("we're not in 304"). A location the caller named by a number or name the
# building has (checked against the span) is no longer capped here: its confidence is the
# stated share mapped from the model's rating of the caller's words (core.fusion.stated_share).
GROUNDED_CONFIDENCE_CAP = 0.85
# A location worked out from a description ("the conference room facing the parking lot").
DESCRIBED_CONFIDENCE_CAP = 0.7
# A record_clue is a descriptive factor; its confidence tempers it (1.0 = up to 10 to 1).
CLUE_CONFIDENCE_CAP = 0.9
CLUE_OTHERS_FIT_MIN = 0.05  # > 0 keeps a clue descriptive (core.fusion.names_rooms)
CLUE_OTHERS_FIT_MAX = 0.5
# record_conclusion: the model's overall judgment of where the caller is, from everything said.
# Its probability is the share the rooms get (core.fusion: never lowers rooms that already hold
# more, never zeroes another room). Only the floor and the ceiling are fixed here.
CONCLUSION_MIN_SHARE = 0.2
CONCLUSION_SHARE_CAP = 0.9
# CONCLUSION_TAG / is_conclusion live in core.fusion, which applies a conclusion last.
RELATIVE_FLOORS = ("top", "ground", "basement")

# Model settings (see the module docstring); the default model is the eval winner
# (eval/run_agent_eval.py on the free-form calls, layout calls included).
DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"
MODEL_LABELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-fable-5-2": "Claude Fable 5.2",
}
# Models that reject thinking {"type": "disabled"}: thinking stays adaptive on them.
_THINKING_ALWAYS_ON = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")
# Models with server-side refusal fallbacks (beta): a declined request is re-run on a fallback model.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_THINKING = "adaptive"
DEFAULT_EFFORT = "low"
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Strict tool schemas are off: every record is checked in code anyway, and strict grammars for
# these tools compiled slowly and were refused with adaptive thinking on claude-haiku-5-5.
# LLM_STRICT_TOOLS=1 turns them back on.
DEFAULT_STRICT_TOOLS = "0"
_MAX_OUTPUT_TOKENS = {"disabled": 2000, "adaptive": 16000}  # thinking counts toward max_tokens

# Fact text that states something as checked, or calls a place safe / clear (ground rule 1).
_ASSERTS_VERIFIED = re.compile(
    r"\b(?:verified|confirmed|certain(?:ly)?|definitely|guaranteed|proven|is safe|are safe|safe to enter|"
    r"all clear|clear of|no known (?:hazard|threat)s?|no (?:threat|danger|hazard)s?)\b",
    re.IGNORECASE,
)

# Diagnostics of the most recent live run (request count, token use and seconds per request).
LAST_RUN: dict[str, Any] = {}


class AgentUnavailable(RuntimeError):
    """The agent cannot run this turn; the rule path is used instead."""


def agent_model() -> str:
    """The model this process reads calls with: ``LLM_MODEL``, else ``DEFAULT_AGENT_MODEL``."""
    return (os.getenv("LLM_MODEL") or DEFAULT_AGENT_MODEL).strip()


def model_label(model: Optional[str] = None) -> str:
    """Display name of ``model`` (default: this process's model), e.g. "Claude Opus 5.5"."""
    m = model or agent_model()
    return MODEL_LABELS.get(m, m)


def model_settings() -> dict[str, Any]:
    """Model, thinking, effort and strict-tool settings for this process (environment overrides the defaults)."""
    model = agent_model()
    thinking = (os.getenv("LLM_THINKING") or DEFAULT_THINKING).strip().lower()
    if thinking not in ("disabled", "adaptive") or model in _THINKING_ALWAYS_ON:
        thinking = "adaptive"
    effort = (os.getenv("LLM_EFFORT") or DEFAULT_EFFORT).strip().lower()
    if effort not in _EFFORTS:
        effort = DEFAULT_EFFORT
    strict = (os.getenv("LLM_STRICT_TOOLS") or DEFAULT_STRICT_TOOLS).strip().lower() in ("1", "true", "yes", "on")
    return {"model": model, "thinking": thinking, "effort": effort, "strict": strict}


def request_tools(strict: bool = True) -> list[dict]:
    """The tool list as sent: ``TOOLS``, without the ``strict`` flag when strict tools are off."""
    if strict:
        return TOOLS
    return [{k: v for k, v in t.items() if k != "strict"} for t in TOOLS]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_system_prompt(idx: BuildingIndex, vocab: Optional[dict] = None) -> str:
    """The briefing with this building's layout map (``vocab`` is accepted for older callers and unused)."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{{BUILDING_LAYOUT}}", layout_roster(idx))


def _rooms_text(building: Building, room_ids: Sequence[str], limit: int = 4) -> str:
    names = {r.id: r.name for r in building.rooms}
    shown = [f"{names.get(rid, '?')} ({rid})" for rid in room_ids[:limit]]
    more = len(room_ids) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def _location_text(building: Building, selector: Optional[RoomSelector], negate: bool = False) -> str:
    if selector is None:
        return "location ?"
    from core.evidence import selector_hits

    hits = selector_hits(building, selector)
    sel = json.dumps(selector.model_dump(exclude_none=True), sort_keys=True)
    where = _rooms_text(building, hits) if hits else "no room of this building"
    return f"location {'NOT ' if negate else ''}{sel} -> {where}"


def _evidence_summary(ev: Evidence, building: Optional[Building] = None) -> str:
    if ev.kind == "vocab" and ev.clue is not None:
        what = f"vocab {ev.clue.type}={ev.clue.value}" + ("" if ev.clue.polarity else " (denied)")
    elif building is not None:
        what = _location_text(building, ev.selector, ev.negate)
    else:
        sel = ev.selector.model_dump(exclude_none=True) if ev.selector is not None else {}
        what = f"location {'NOT ' if ev.negate else ''}{json.dumps(sel, sort_keys=True)}"
    if is_conclusion(ev):
        kind = f"your conclusion, probability {ev.confidence:.2f}"
    elif ev.kind == "grounded" and (ev.negate or ev.default_match > 0.0):
        kind = f"clue, confidence {ev.confidence:.2f}"
    else:
        kind = f"statement, share {ev.confidence:.2f}"
    return f'- {ev.id} [{ev.status}; {kind}] {what} span "{ev.span}"'


def statement_labels(reading: Optional[DialogReading]) -> dict[str, int]:
    """``label -> index in reading.statements`` for the rule-read statements the model is
    asked to rate (``rate_statement``): the specific ones in the caller's own words."""
    if reading is None:
        return {}
    rated = [i for i, st in enumerate(reading.statements) if st.rating is not None]
    return {f"s{n + 1}": i for n, i in enumerate(rated)}


def _statement_summary(st: DialogStatement, building: Optional[Building] = None, label: str = "") -> str:
    if st.clue is not None:
        what = f"vocab {st.clue.type}={st.clue.value}" + ("" if st.clue.polarity else " (denied)")
    elif building is not None:
        what = _location_text(building, st.selector)
    else:
        what = f"location {json.dumps(st.selector.model_dump(exclude_none=True), sort_keys=True)}" if st.selector else "?"
    head = f"{label}: " if label else ""
    return f'- {head}{what} span "{st.span}"'


def build_user_message(building: Building, transcript: Sequence[TranscriptTurn], index: int,
                       prior_evidence: Sequence[Evidence], reading: Optional[DialogReading] = None, *,
                       prior_facts: Sequence[Fact] = (), incident_type: Optional[str] = None) -> str:
    lines = [f"Building: {building.id}", ""]
    if prior_evidence:
        lines.append("Evidence already recorded (ids you may supersede):")
        lines.extend(_evidence_summary(ev, building) for ev in prior_evidence)
    else:
        lines.append("Evidence already recorded: none")
    if prior_facts or (incident_type and incident_type != "unknown"):
        lines += ["", "Already on the console from earlier turns (do not record again):"]
        lines.extend(f"- fact {f.category}: {f.text}" for f in prior_facts)
        if incident_type and incident_type != "unknown":
            lines.append(f"- incident type: {incident_type}")
    lines += ["", "<transcript>"]
    for i, t in enumerate(transcript[: index + 1]):
        if t.speaker == "system":
            continue
        marker = ">>> " if i == index else ""
        who = t.speaker
        q = t.question
        if t.speaker == "dispatcher" and q is not None and q.clue_type and q.value:
            who = f"dispatcher (console question about {q.clue_type}={q.value})"
        lines.append(f"{marker}[{i}] {who}: {t.text}")
    lines.append("</transcript>")
    if reading is not None and reading.statements:
        labels = {i: label for label, i in statement_labels(reading).items()}
        lines += ["", f"Already recorded for turn [{index}] from the dispatcher's answer buttons (do not record again):"]
        lines.extend(_statement_summary(st, building, labels.get(i, "")) for i, st in enumerate(reading.statements))
        if labels:
            lines.append(f"Rate each labelled statement with rate_statement ({', '.join(labels.values())}).")
    lines += ["", f"Read caller turn [{index}] (marked >>>) and record what it adds."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disk cache and usage (same files and shapes as parser/extract.py)
# ---------------------------------------------------------------------------


def _cache_key(system: str, user: str, settings: Optional[dict] = None) -> str:
    s = settings or model_settings()
    tools = json.dumps(request_tools(s["strict"]), sort_keys=True)
    head = f"{s.get('model') or agent_model()}\x00thinking={s['thinking']}\x00effort={s['effort']}"
    return hashlib.sha256(f"{head}\x00{system}\x00{tools}\x00{user}".encode()).hexdigest()


def _record_usage(response_usage: Any = None, cache_hit: bool = False, seconds: float = 0.0, kind: str = "agent") -> None:
    # Same lock as parser/extract.py: the file is read, changed and rewritten as one step.
    with _extract._llm_lock:
        _record_usage_locked(response_usage, cache_hit, seconds, kind)


def _record_usage_locked(response_usage: Any, cache_hit: bool, seconds: float = 0.0, kind: str = "agent") -> None:
    usage = _extract.llm_usage()
    if cache_hit:
        usage["cache_hits"] = usage.get("cache_hits", 0) + 1
    else:
        u = response_usage
        usage["calls"] = usage.get("calls", 0) + 1
        usage[f"{kind}_calls"] = usage.get(f"{kind}_calls", 0) + 1
        usage["input_tokens"] = usage.get("input_tokens", 0) + (getattr(u, "input_tokens", 0) or 0)
        usage["output_tokens"] = usage.get("output_tokens", 0) + (getattr(u, "output_tokens", 0) or 0)
        usage["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0) + (
            getattr(u, "cache_read_input_tokens", 0) or 0)
        usage["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens", 0) + (
            getattr(u, "cache_creation_input_tokens", 0) or 0)
        usage["request_seconds"] = round(usage.get("request_seconds", 0.0) + seconds, 3)
    _extract._write_json(_extract.LLM_USAGE_PATH, usage)


def _budget() -> Optional[int]:
    """Request cap of this process: ``LLM_MAX_CALLS`` when set, else no cap."""
    return _extract.request_budget()


def _budget_remaining() -> Optional[int]:
    """Requests left under ``LLM_MAX_CALLS``, or None when no cap is set."""
    budget = _budget()
    if budget is None:
        return None
    with _extract._llm_lock:
        return max(0, budget - _extract._llm_calls_made)


def _reserve_request() -> None:
    """Count one request against LLM_MAX_CALLS (shared with parser/extract.py)."""
    with _extract._llm_lock:
        budget = _budget()
        if budget is not None and _extract._llm_calls_made >= budget:
            raise AgentUnavailable(f"LLM request budget for this process is spent ({budget}; raise LLM_MAX_CALLS)")
        _extract._llm_calls_made += 1


_clients: dict[str, Any] = {}


def _make_client() -> Any:
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider != "anthropic":
        raise AgentUnavailable(f"LLM_PROVIDER is '{provider or 'unset'}'; only 'anthropic' is implemented")
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        raise AgentUnavailable("LLM_API_KEY is not set in .env")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - declared in pyproject
        raise AgentUnavailable("the 'anthropic' package is not installed (run `uv sync`)") from exc
    # One client per key, kept for the life of the process: its connection pool saves a TLS
    # handshake on every caller turn. max_retries=0: a retry doubles the wait on a live call.
    client = _clients.get(api_key)
    if client is None or not isinstance(client, anthropic.Anthropic):
        client = _clients[api_key] = anthropic.Anthropic(api_key=api_key, timeout=AGENT_TIMEOUT_S, max_retries=0)
    return client


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    """Raw output of the loop: every tool call in order (records are validated later)."""

    calls: list[dict] = field(default_factory=list)  # {"tool", "args", "result_size"}
    requests: int = 0
    capped: bool = False
    rounds: int = 0  # replies whose tool calls were all answered
    cut_short: Optional[str] = None  # reason a later request failed (records of earlier rounds kept)
    seconds: float = 0.0  # wall-clock time spent in model requests

    @property
    def records(self) -> int:
        return sum(1 for c in self.calls if c["tool"] in RECORD_TOOLS or c["tool"] in RATE_TOOLS)


def _block_attr(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _assistant_content(content: Sequence[Any]) -> list[Any]:
    """Echo the assistant turn back unchanged (SDK blocks are accepted as params; thinking
    blocks must go back exactly as they came)."""
    return list(content)


def _request_kwargs(system: str, settings: dict) -> dict[str, Any]:
    """Everything of a request except ``messages`` and ``max_tokens`` (shared with ``prewarm``)."""
    return {
        "model": settings.get("model") or agent_model(),
        "thinking": {"type": "adaptive"} if settings["thinking"] == "adaptive" else {"type": "disabled"},
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "tools": request_tools(settings["strict"]),
        "output_config": {"effort": settings["effort"]},
    }


def _create(client: Any, settings: dict, **kwargs: Any) -> Any:
    """``messages.create``, with server-side refusal fallbacks on models that have them.

    A request the safety classifiers decline is re-run on a fallback model inside the same call
    (``fallbacks: "default"``, beta). Test stubs without a ``beta`` namespace get the plain call.
    """
    beta = getattr(client, "beta", None)
    if (settings.get("model") or agent_model()) in _FALLBACK_MODELS and beta is not None and hasattr(beta, "messages"):
        return beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
    return client.messages.create(**kwargs)


def _call_model(client: Any, system: str, messages: list[dict], settings: Optional[dict] = None) -> Any:
    settings = settings or model_settings()
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        anthropic = None  # type: ignore[assignment]
    try:
        return _create(client, settings, max_tokens=_MAX_OUTPUT_TOKENS[settings["thinking"]], messages=messages,
                       **_request_kwargs(system, settings))
    except Exception as exc:  # map SDK errors to a short reason; anything else is also a fallback
        if anthropic is not None:
            if isinstance(exc, anthropic.AuthenticationError):
                raise AgentUnavailable("Anthropic rejected LLM_API_KEY") from exc
            if isinstance(exc, anthropic.RateLimitError):
                raise AgentUnavailable("Anthropic rate limit reached") from exc
            if isinstance(exc, anthropic.APITimeoutError):
                raise AgentUnavailable("Anthropic API timed out") from exc
            if isinstance(exc, anthropic.APIStatusError):
                raise AgentUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
            if isinstance(exc, anthropic.APIConnectionError):
                raise AgentUnavailable("Anthropic API unreachable") from exc
        raise AgentUnavailable(f"unexpected error: {type(exc).__name__}: {exc}") from exc


def _run_loop(client: Any, idx: BuildingIndex, system: str, user: str, turn_text: Optional[str] = None,
              settings: Optional[dict] = None, earlier_texts: Sequence[str] = ()) -> _Run:
    """Run the tool loop. A failure before the first completed round raises
    ``AgentUnavailable``; a failure after it returns the completed rounds with ``cut_short``
    set (requests already paid for are not thrown away)."""
    run = _Run()
    try:
        _loop(run, client, idx, system, user, turn_text, settings or model_settings(), earlier_texts)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, AgentUnavailable) else f"unexpected error: {type(exc).__name__}: {exc}"
        if run.rounds == 0:
            if isinstance(exc, AgentUnavailable):
                raise
            raise AgentUnavailable(reason) from exc
        run.cut_short = reason
    return run


def _loop(run: _Run, client: Any, idx: BuildingIndex, system: str, user: str, turn_text: Optional[str],
          settings: dict, earlier_texts: Sequence[str] = ()) -> None:
    messages: list[dict] = [{"role": "user", "content": user}]
    LAST_RUN.clear()
    LAST_RUN["requests"] = []
    LAST_RUN["settings"] = dict(settings)
    record_no = 0
    corrected = False  # one round of corrections at most: a second reply for a bad record, not a third
    while run.requests < AGENT_MAX_REQUESTS_PER_TURN:
        _reserve_request()
        started = time.perf_counter()
        response = _call_model(client, system, messages, settings)
        seconds = time.perf_counter() - started
        run.requests += 1
        run.seconds += seconds
        u = getattr(response, "usage", None)
        _record_usage(u, seconds=seconds)
        LAST_RUN["requests"].append({
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "stop_reason": getattr(response, "stop_reason", None),
            "seconds": round(seconds, 3),
        })
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise AgentUnavailable("model refused")
        if stop == "max_tokens":
            raise AgentUnavailable("model reply was cut off (max_tokens)")
        content = list(getattr(response, "content", None) or [])
        tool_uses = [b for b in content if _block_attr(b, "type") == "tool_use"]
        # Tool calls count whatever the stop reason says: without strict tools the API returns
        # a reply that ends in tool calls with stop_reason "end_turn" (seen live on
        # claude-haiku-5-5), and dropping it would throw the whole reading away.
        if not tool_uses:
            return
        results = []
        round_calls: list[dict] = []
        needs_reply = False  # a look-up to answer, or a location record the model should correct
        fixable = False
        for block in tool_uses:
            name = str(_block_attr(block, "name"))
            args = dict(_block_attr(block, "input") or {})
            if name in LOOKUP_TOOLS:
                result = run_lookup(idx, name, args)
                needs_reply = True
            elif name in RATE_TOOLS:
                result = {"noted": "rating"}
            elif name in RECORD_TOOLS:
                record_no += 1
                result = record_ack(idx, name, _clean_record_args(name, args)[0], record_no, turn_text, earlier_texts)
                # Only a location that cannot be kept is worth a second request on a live call.
                fixable = fixable or (name in ("record_location", "record_conclusion")
                                      and ("warning" in result or "error" in result))
            else:
                result = {"error": f"unknown tool '{name}'"}
                needs_reply = True
            text = json.dumps(result, separators=(",", ":"))
            round_calls.append({"tool": name, "args": args, "result_size": len(text)})
            results.append({"type": "tool_result", "tool_use_id": _block_attr(block, "id"), "content": text,
                            **({"is_error": True} if "error" in result else {})})
        run.calls.extend(round_calls)  # a round counts once all its tool calls were answered
        run.rounds += 1
        if fixable and not corrected:
            needs_reply, corrected = True, True
        if not needs_reply:
            return  # records only, every location usable: one request was enough
        messages.append({"role": "assistant", "content": _assistant_content(content)})
        messages.append({"role": "user", "content": results})  # all results of one model turn together
    run.capped = True  # the fourth reply asked for more; its records are kept, no fifth request


# ---------------------------------------------------------------------------
# Selector hygiene
# ---------------------------------------------------------------------------

def _clean_selector(raw: Any) -> tuple[dict, list[str]]:
    """The selector without empty conditions, plus notes for the trace.

    The strict tool schema lets the model send ``within_m_of: {room_id: "", radius_m: 0}``
    or ``unit: ""`` for "not set". Those count as absent instead of as an unknown room id
    or a condition that matches nothing (which would drop the record and invite a retry).
    """
    if isinstance(raw, str):  # the selector written as a JSON string
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw, []
    if not isinstance(raw, dict):
        return raw, []
    out: dict = {}
    notes: list[str] = []
    for key, value in raw.items():
        if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none")):
            continue
        if key in ("room_ids", "floor_labels", "room_types") and isinstance(value, (str, int)):
            value = [str(value)]  # one value where a list was expected
        if key == "within_m_of":
            room_id, radius = "", 0.0
            if isinstance(value, dict):
                room_id = str(value.get("room_id") or "").strip()
                try:
                    radius = float(value.get("radius_m") or 0)
                except (TypeError, ValueError):
                    radius = 0.0
            if not room_id or radius <= 0:
                notes.append("empty within_m_of ignored")
                continue
            out[key] = value
        elif isinstance(value, str):
            if value.strip():
                out[key] = value
        elif isinstance(value, list):
            items = [v for v in value if not (isinstance(v, str) and not v.strip()) and v is not None]
            if items:
                out[key] = items
        else:
            out[key] = value
    return out, notes


def _clean_record_args(name: str, args: dict) -> tuple[dict, list[str]]:
    if name != "record_location":
        return args, []
    selector, notes = _clean_selector(args.get("selector"))
    if not isinstance(selector, dict):
        return args, []
    return {**args, "selector": selector}, notes


# ---------------------------------------------------------------------------
# Validation (code decides what is kept)
# ---------------------------------------------------------------------------


def _norm_with_map(text: str) -> tuple[str, list[int]]:
    """``text`` lower-cased with straight quotes and single spaces, and the index in ``text`` of each character."""
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        c = "'" if ch in "’‘`" else '"' if ch in "“”" else ch
        if c.isspace():
            if in_space:
                continue
            c, in_space = " ", True
        else:
            in_space = False
        low = c.lower()
        out.append(low if len(low) == 1 else c)
        index.append(i)
    return "".join(out), index


def _verbatim(span: str, caller_texts: Sequence[str]) -> Optional[str]:
    """The span as it stands in a caller turn (the caller's own casing, quotes and spacing), else None.

    Case, curly quotes and runs of spaces are not held against the model; anything else is.
    """
    span = (span or "").strip()
    if not span:
        return None
    for text in caller_texts:
        if span in text:
            return span
    want = _norm_with_map(span)[0].strip()
    if not want:
        return None
    for text in caller_texts:
        norm, index = _norm_with_map(text)
        k = norm.find(want)
        if k >= 0:
            return text[index[k]: index[k + len(want) - 1] + 1]
    return None


def _confidence(args: dict) -> Optional[float]:
    try:
        return min(1.0, max(0.0, float(args.get("confidence"))))
    except (TypeError, ValueError):
        return None


# Without strict tool schemas the arguments are whatever the model wrote, so every field is
# read defensively: a malformed value is treated as absent or drops the record, never raises.
_NOT_SET = ("", "null", "none", "n/a", "undefined")


def _opt_str(value: Any) -> Optional[str]:
    """A string argument, or None for the ways a model writes "not set" (null, "", "null", "none")."""
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return None if text.lower() in _NOT_SET else text


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        return low in ("true", "yes", "1") if low not in _NOT_SET else default
    return default if value is None else bool(value)


class _NameCheck:
    """Does the caller's span carry the number or a distinctive name word of the rooms selected?"""

    def __init__(self, idx: BuildingIndex):
        self.idx = idx
        self.labels = room_labels(idx.building, idx.meta_rooms)
        words: dict[str, set[str]] = {}
        for r in idx.building.rooms:
            toks = set(_tokens(r.name)) | set(_tokens(idx.tenant(r.id) or ""))
            words[r.id] = {t for t in toks if len(t) >= 4 and t not in _STOPWORDS and not t.isdigit()}
        count = Counter(w for ws in words.values() for w in ws)
        limit = max(4, len({r.floor for r in idx.building.rooms}))
        # a word on more rooms than that ("office", "corridor", "executive") names a kind of room
        self.words = {rid: {w for w in ws if count[w] <= limit} for rid, ws in words.items()}

    def names(self, span: str, room_ids: Sequence[str]) -> bool:
        toks = _tokens(span)
        said = set(toks)
        # speech recognition splits words ("ware house" for "Warehouse")
        said |= {a + b for a, b in zip(toks, toks[1:])}
        units = {normalize_unit(w) for w in said}
        if not room_ids:
            return False
        for rid in room_ids:
            if self.labels.get(rid, frozenset()) & units:
                continue
            words = self.words.get(rid, set())
            if words & said:
                continue
            # a short form of the name word, or the other way round ("roof" for "Rooftop")
            if any(len(s) >= 4 and (w.startswith(s) or s.startswith(w)) for w in words for s in said):
                continue
            return False
        return True


def _validate(run_calls: list[dict], idx: BuildingIndex, transcript: Sequence[TranscriptTurn], index: int,
              prior_evidence: Sequence[Evidence], vocab: dict, reading: DialogReading,
              source: str = "agent", model: Optional[str] = None) -> Interpretation:
    from parser.interpret import evidence_id, fact_id

    turn = transcript[index]
    caller_texts = [t.text for t in transcript[: index + 1] if t.speaker == "caller"]
    prior_ids = {ev.id for ev in prior_evidence}
    prior_vocab = {(ev.clue.type, ev.clue.value): ev.id for ev in prior_evidence
                   if ev.kind == "vocab" and ev.clue is not None and ev.status == "active"}
    prior_grounded = {(json.dumps(ev.selector.model_dump(exclude_none=True), sort_keys=True), ev.negate): ev.id
                      for ev in prior_evidence if ev.kind == "grounded" and ev.selector is not None
                      and ev.status == "active"}
    # rooms each earlier / rule-read location statement selects: the same rooms again is the same statement
    prior_rooms = {(frozenset(select_rooms(idx, ev.selector)), ev.negate): ev.id
                   for ev in prior_evidence if ev.kind == "grounded" and ev.selector is not None
                   and ev.status == "active"}
    prior_rooms.pop((frozenset(), False), None)
    prior_rooms.pop((frozenset(), True), None)
    # rooms of earlier statements that place the caller, other than earlier conclusions (which a
    # new conclusion replaces): a conclusion repeating one of them adds nothing
    prior_named_rooms = {(frozenset(select_rooms(idx, ev.selector)), False) for ev in prior_evidence
                         if ev.kind == "grounded" and ev.selector is not None and ev.status == "active"
                         and not ev.negate and ev.default_match <= 0.0 and not is_conclusion(ev)}
    conclusion_kept = False
    observed = turn.at or datetime.now(timezone.utc)
    name_check = _NameCheck(idx)
    who = model_label(model)

    evidence: list[Evidence] = []
    offsets: list[int] = []  # start of each evidence span in this turn's text (-1 unknown)
    facts: list[Fact] = []
    supersedes: list[str] = []
    suggested: Optional[tuple[str, str]] = None
    trace: list[AgentTraceEntry] = []
    seen_vocab: set[tuple[str, str]] = set()
    seen_grounded: set[tuple[str, bool]] = set()
    seen_rooms: set[tuple[frozenset, bool]] = set()  # rooms of this turn's clues
    seen_named: set[tuple[frozenset, bool]] = set()  # rooms of this turn's named places
    dialog_grounded: dict[str, tuple[str, str]] = {}  # selector json -> (evidence id, span)
    dialog_rooms: dict[frozenset, tuple[str, str]] = {}  # rooms selected -> (evidence id, span)

    # 1. The deterministic dialog rules run first and are kept as they are (one exception
    #    below: the agent's denial of the very place a rule read as positive). How much a
    #    specific statement weighs is the model's call: its rate_statement for the label
    #    replaces the rule estimate (the first usable rating per label counts).
    labels = statement_labels(reading)
    rated_by_model: dict[int, tuple[Any, list[str]]] = {}
    for call in run_calls:
        if call["tool"] not in RATE_TOOLS:
            continue
        i = labels.get(str(call["args"].get("statement", "")).strip().lower())
        if i is not None and i not in rated_by_model:
            rated_by_model[i] = clean_rating(call["args"].get("certainty"), call["args"].get("specificity"),
                                             reading.statements[i].rating)
    for i, st in enumerate(reading.statements):
        rationale = st.rationale
        if i in rated_by_model:
            st = st.with_rating(rated_by_model[i][0])
            rationale = "; ".join(x for x in (rationale, rating_note(st.rating, who)) if x)
        elif st.rating is not None:
            rationale = "; ".join(x for x in (rationale, rating_note(st.rating, "rules")) if x)
        evidence.append(Evidence(
            id=evidence_id(index, len(evidence)), kind=st.kind, clue=st.clue, selector=st.selector,  # type: ignore[arg-type]
            default_match=0.0, confidence=st.confidence, span=st.span, turn=index, source=st.source,
            rationale=rationale,
        ))
        offsets.append(st.start)
        if st.clue is not None:
            seen_vocab.add((st.clue.type, st.clue.value))
        elif st.selector is not None:
            sel_key = json.dumps(st.selector.model_dump(exclude_none=True), sort_keys=True)
            seen_grounded.add((sel_key, False))
            dialog_grounded[sel_key] = (evidence[-1].id, st.span)
            rooms = frozenset(select_rooms(idx, st.selector))
            if rooms:
                seen_named.add((rooms, False))
                dialog_rooms[rooms] = (evidence[-1].id, st.span)

    def offset_of(span: str) -> int:
        i = turn.text.find(span)
        return i if i >= 0 else turn.text.lower().find(span.lower())

    # 2. What the agent recorded, checked in code.

    for call in run_calls:
        name, args = call["tool"], call["args"]
        entry = AgentTraceEntry(turn=index, tool=name, args=args, result_size=call["result_size"])
        trace.append(entry)
        if name in RATE_TOOLS:
            label = str(args.get("statement", "")).strip().lower()
            i = labels.get(label)
            if i is None:
                entry.kept = False
                entry.reason = f"unknown statement label '{label}'"
            else:
                rating, fixes = clean_rating(args.get("certainty"), args.get("specificity"), reading.statements[i].rating)
                entry.reason = "; ".join([f"{label} ({reading.statements[i].span!r}): {rating_note(rating, who)}"]
                                         + fixes)
            continue
        if name not in RECORD_TOOLS:
            continue

        notes: list[str] = []

        def note(msg: str) -> None:
            notes.append(msg)
            entry.reason = "; ".join(notes)

        def drop(reason: str) -> None:
            entry.kept = False
            entry.reason = "; ".join([reason] + notes)

        if name == "record_conclusion":
            # The model's overall judgment from the whole call: it may rest on words from any turn.
            raw = args.get("spans")
            raws = [raw] if isinstance(raw, str) else [str(x) for x in (raw or []) if x is not None]
            if not raws and args.get("span"):
                raws = [str(args["span"])]
            spans = [s for s in (_verbatim(r, caller_texts) for r in raws) if s]
            ids = list(dict.fromkeys(clue_room_ids(args)))
            bad = [i for i in ids if i not in idx.rooms]
            try:
                share = float(args.get("probability"))
            except (TypeError, ValueError):
                share = None
            if conclusion_kept:
                drop("one conclusion per turn (the first is kept)")
            elif not spans:
                drop("none of its spans is verbatim in a caller turn")
            elif bad:
                drop(f"unknown room id(s) {bad}")
            elif not ids:
                drop("conclusion lists no room")
            elif share is None:
                drop("missing probability")
            elif share < CONCLUSION_MIN_SHARE:
                drop(f"probability {share:.2f} below {CONCLUSION_MIN_SHARE}")
            elif (frozenset(ids), False) in seen_named or (frozenset(ids), False) in prior_named_rooms:
                drop("the same rooms are already on the console as a statement")
            else:
                conclusion_kept = True
                if share > CONCLUSION_SHARE_CAP:
                    note(f"probability capped at {CONCLUSION_SHARE_CAP}")
                share = min(share, CONCLUSION_SHARE_CAP)
                if len(spans) < len(raws):
                    note(f"{len(raws) - len(spans)} span(s) not verbatim in a caller turn left out")
                in_turn = [s for s in spans if _verbatim(s, [turn.text])]
                span = (in_turn or spans)[0]
                what = str(args.get("summary", "")).strip()
                why = str(args.get("reasoning", "")).strip()
                quoted = " · ".join(f'"{s}"' for s in spans)
                rationale = f"{CONCLUSION_TAG}: " + "; ".join(x for x in (what, why, f"from {quoted}") if x)
                evidence.append(Evidence(
                    id=evidence_id(index, len(evidence)), kind="grounded", selector=RoomSelector(room_ids=ids),
                    default_match=0.0, confidence=round(share, 3), span=span, turn=index, source=source,  # type: ignore[arg-type]
                    rationale=rationale[:400], matched_room_ids=ids,
                ))
                offsets.append(offset_of(span))
                for ev in prior_evidence:  # a new conclusion replaces the earlier one
                    if is_conclusion(ev) and ev.status == "active" and ev.id not in supersedes:
                        supersedes.append(ev.id)
                        note(f"replaces the earlier conclusion {ev.id}")
            continue

        raw_span = str(args.get("span", ""))
        span = _verbatim(raw_span, [turn.text])
        if span is None:
            if _verbatim(raw_span, caller_texts) is not None:
                drop("span is from an earlier caller turn (already read then)")
            else:
                drop("span is not verbatim in any caller turn")
            continue

        if name == "supersede":
            eid = str(args.get("evidence_id", ""))
            if eid not in prior_ids:
                drop(f"unknown evidence id '{eid}'")
            elif eid in supersedes:
                drop("already superseded this turn")
            else:
                supersedes.append(eid)
            continue

        if name == "suggest_incident_type":
            itype = str(args.get("type", ""))
            if itype not in INCIDENT_TYPES or itype == "unknown":
                drop(f"unknown incident type '{itype}'")
            elif suggested is not None:
                drop("only one incident type suggestion per turn")
            else:
                suggested = (itype, span)
            continue

        if name == "record_fact":
            text = str(args.get("text", "")).strip()
            category = str(args.get("category", ""))
            room_id = _opt_str(args.get("room_id"))
            premise = _opt_str(args.get("premise_key"))
            if premise not in PREMISE_KEYS:
                premise = None
            if not text:
                drop("empty fact text")
            elif category not in ("hazard", "access", "patient", "people", "threat", "resource"):
                drop(f"unknown fact category '{category}'")
            elif room_id is not None and room_id not in idx.rooms:
                drop(f"unknown room id '{room_id}'")
            elif _ASSERTS_VERIFIED.search(text):
                drop("fact text asserts something as verified or safe")
            else:
                facts.append(Fact(
                    id=fact_id(index, len(facts)), category=category, text=text, span=span,  # type: ignore[arg-type]
                    turn=index, source=source, observed_at=observed, room_id=room_id, verified=False,  # type: ignore[arg-type]
                    premise_key=premise,  # type: ignore[arg-type]
                ))
            continue

        conf = _confidence(args)
        if conf is None:
            drop("missing confidence")
            continue
        if conf < AGENT_MIN_CONFIDENCE:
            drop(f"confidence {conf:.2f} below {AGENT_MIN_CONFIDENCE}")
            continue

        if name == "record_floor":
            value = str(args.get("floor", "")).strip()
            if value.lower() in RELATIVE_FLOORS:
                value = value.lower()
            elif value not in idx.floor_labels:
                drop(f"'{value}' is not a floor of this building")
                continue
            polarity = _as_bool(args.get("polarity"), True)
            ctype, cvalue = ("floor", value) if polarity else ("negative", f"floors:{value}")
            if (ctype, cvalue) in prior_vocab and not polarity:
                drop(f"duplicate of earlier evidence {prior_vocab[(ctype, cvalue)]}")
                continue
            if (ctype, cvalue) in seen_vocab:
                drop("duplicate within this turn")
                continue
            seen_vocab.add((ctype, cvalue))
            rationale = ""
            if polarity and _as_bool(args.get("stated_by_caller"), False):
                # The caller's own statement of their floor: its weight is the stated share for
                # how they said it. Said again in a later turn it is kept: agreement reinforces.
                rating, fixes = clean_rating(args.get("certainty"), args.get("specificity"), rate_span(turn.text, span))
                conf = rating.share()
                rationale = rating_note(rating, who)
                note("; ".join([rationale] + fixes))
            elif polarity:
                # Worked out from something else (a room number, a neighbour): capped like a description.
                conf = min(conf, DESCRIBED_CONFIDENCE_CAP)
                rationale = f"floor worked out by {who}, not stated by the caller"
                note(f"{rationale}: share capped at {DESCRIBED_CONFIDENCE_CAP}")
            else:
                conf = min(conf, GROUNDED_CONFIDENCE_CAP)
            clue = Clue(type=ctype, value=cvalue, polarity=polarity, confidence=round(conf, 3), span=span)  # type: ignore[arg-type]
            evidence.append(Evidence(id=evidence_id(index, len(evidence)), kind="vocab", clue=clue,
                                     confidence=clue.confidence, span=span, turn=index, source=source,  # type: ignore[arg-type]
                                     rationale=rationale))
            offsets.append(offset_of(span))
            continue

        if name == "record_clue":
            ids = clue_room_ids(args)
            bad = [i for i in ids if i not in idx.rooms]
            if bad:
                drop(f"unknown room id(s) {bad}")
                continue
            if not ids:
                drop("clue lists no room")
                continue
            ids = list(dict.fromkeys(ids))
            negate = _as_bool(args.get("negate"), False)
            try:
                others = float(args.get("others_fit", 0.2))
            except (TypeError, ValueError):
                others = 0.2
            others = min(CLUE_OTHERS_FIT_MAX, max(CLUE_OTHERS_FIT_MIN, others))
            default_match = 1.0 if negate else others  # a denial leaves every other room as it was
            rooms_key = (frozenset(ids), negate)
            if rooms_key in prior_rooms:
                drop(f"duplicate of earlier evidence {prior_rooms[rooms_key]}")
                continue
            if rooms_key in seen_rooms:
                drop("duplicate within this turn")
                continue
            seen_rooms.add(rooms_key)
            if conf > CLUE_CONFIDENCE_CAP:
                note(f"confidence capped at {CLUE_CONFIDENCE_CAP}")
            conf = min(conf, CLUE_CONFIDENCE_CAP)
            what = str(args.get("clue", "")).strip()
            why = str(args.get("reasoning", "")).strip()
            rationale = (f"{what}: {why}" if what and why else what or why)[:400]
            evidence.append(Evidence(
                id=evidence_id(index, len(evidence)), kind="grounded", selector=RoomSelector(room_ids=ids),
                default_match=default_match, negate=negate, confidence=round(conf, 3), span=span, turn=index,
                source=source, rationale=rationale, matched_room_ids=ids,  # type: ignore[arg-type]
            ))
            offsets.append(offset_of(span))
            continue

        # record_location
        raw_selector, cleaned = _clean_selector(args.get("selector") or {})
        for msg in cleaned:
            note(msg)
        try:
            selector = RoomSelector.model_validate(raw_selector)
        except (ValueError, TypeError):
            drop("invalid selector")
            continue
        if selector_is_empty(selector):
            drop("selector has no condition")
            continue
        bad = unknown_room_ids(idx, selector)
        if bad:
            drop(f"unknown room id(s) {bad}")
            continue
        hits = select_rooms(idx, selector)
        if not hits:
            drop("selector matches no room")
            continue
        negate = _as_bool(args.get("negate"), False)
        # A location record names rooms outright (default_match 0, core.fusion.naming_floor).
        # ``default_match`` is no longer part of the tool; a value in an older cached run or a
        # test script is still honoured and makes the statement a softer, descriptive one.
        try:
            default_match = min(1.0, max(0.0, float(args.get("default_match", 0.0))))
        except (TypeError, ValueError):
            default_match = 0.0
        if negate:
            default_match = 1.0  # a denial leaves every other room as it was
        key = (json.dumps(selector.model_dump(exclude_none=True), sort_keys=True), negate)
        rooms_key = (frozenset(hits), negate)
        named = negate or name_check.names(span, hits)
        stated = named and not negate and default_match <= 0.0  # a specific statement (core.fusion)
        if (key in prior_grounded or rooms_key in prior_rooms) and not stated:
            drop(f"duplicate of earlier evidence {prior_grounded.get(key) or prior_rooms[rooms_key]}")
            continue
        if key in seen_grounded or rooms_key in seen_named:
            drop("duplicate within this turn (or already read by the dialog rules)")
            continue
        seen_grounded.add(key)
        seen_named.add(rooms_key)
        rationale = str(args.get("rationale", ""))[:300]
        if stated:
            # The caller named these rooms: the weight is the stated share for how they said
            # it. Named again in a later turn the statement is kept: agreement reinforces.
            rating, fixes = clean_rating(args.get("certainty"), args.get("specificity"), rate_span(turn.text, span))
            capped = rating.share()
            rationale = "; ".join(x for x in (rationale, rating_note(rating, who)) if x)
            note("; ".join([rating_note(rating, who)] + fixes))
        else:
            cap = GROUNDED_CONFIDENCE_CAP if named else DESCRIBED_CONFIDENCE_CAP
            capped = min(conf, cap)
            if capped < conf:
                note(f"confidence capped at {cap}" + ("" if named else
                                                       " (rooms worked out from a description, not named by the caller)"))
        # "we're not in suite 304": a unit rule that read the same words as a positive
        # statement misread a denial. Keep the denial, mark the rule's reading superseded.
        misread = (dialog_grounded.get(key[0]) or dialog_rooms.get(frozenset(hits))) if negate else None
        if misread is not None and (misread[1] in span or span in misread[1]) and misread[0] not in supersedes:
            supersedes.append(misread[0])
            note(f"denial replaces the dialog rules' positive reading {misread[0]}")
        evidence.append(Evidence(
            id=evidence_id(index, len(evidence)), kind="grounded", selector=selector, default_match=default_match,
            negate=negate, confidence=round(capped, 3), span=span, turn=index, source=source,  # type: ignore[arg-type]
            rationale=rationale[:400], matched_room_ids=hits,
        ))
        offsets.append(offset_of(span))

    # 3. Corrections: the agent's explicit supersede calls, plus the dialog rule (earlier
    #    evidence of a slot restated after "no wait" / "sorry, I meant" / "actually").
    for eid in superseded_ids(list(zip(evidence, offsets)), prior_evidence, reading.correction_at):
        if eid not in supersedes:
            supersedes.append(eid)
    own = set(supersedes) & {e.id for e in evidence}
    if own:
        evidence = [e.model_copy(update={"status": "superseded"}) if e.id in own else e for e in evidence]

    return Interpretation(
        evidence=evidence, facts=facts,
        suggested_type=suggested[0] if suggested else None,  # type: ignore[arg-type]
        suggested_span=suggested[1] if suggested else None,
        source=source, trace=trace, supersedes=supersedes,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def button_reading(transcript: Sequence[TranscriptTurn], index: int, vocab: Optional[dict]) -> DialogReading:
    """The dispatcher's Yes / No / Not sure button answer for this turn, if any, as a DialogReading.

    A typed or spoken "yes" is left to the model, which reads it against the question asked.
    """
    key, answer, stmt = resolve_answer(transcript, index, vocab)
    reading = DialogReading()
    if stmt is not None and stmt.source == "dispatcher":
        reading.statements.append(stmt)
        reading.question_key, reading.answer = key, answer
    return reading


def _checked(run_calls: list[dict], idx: BuildingIndex, building: Building, transcript: Sequence[TranscriptTurn],
             index: int, prior_evidence: Sequence[Evidence], vocab: dict, reading: DialogReading,
             source: str) -> Interpretation:
    """``_validate``, with the rule path as the net under it: whatever the model wrote, a caller
    turn is never lost to an exception."""
    try:
        return _validate(run_calls, idx, transcript, index, prior_evidence, vocab, reading, source, agent_model())
    except Exception as exc:  # a record shape nobody foresaw
        return _fallback(building, transcript, index, f"could not read the model's records ({type(exc).__name__})",
                         prior_evidence=prior_evidence, vocab=vocab, source=source)


def _fallback(building: Building, transcript: Sequence[TranscriptTurn], index: int, reason: str, *,
              prior_evidence: Sequence[Evidence], vocab: Optional[dict], source: str = "agent") -> Interpretation:
    from parser.interpret import interpret_turn

    out = interpret_turn(building, transcript, index, "rules", prior_evidence=prior_evidence, vocab=vocab)
    trace = list(out.trace) + [AgentTraceEntry(turn=index, tool="agent", kept=False, reason=reason)]
    return out.model_copy(update={"source": f"rules ({source} unavailable: {reason})", "trace": trace})


def interpret(
    building: Building,
    transcript: Sequence[TranscriptTurn],
    index: int,
    *,
    prior_evidence: Sequence[Evidence] = (),
    vocab: dict | None = None,
    client: Any = None,
    meta: dict | None = None,
    prior_facts: Sequence[Fact] = (),
    incident_type: Optional[str] = None,
    source: str = "agent",
) -> Interpretation:
    """Read ``transcript[index]`` with the grounded interpreter.

    ``source`` is the label the result carries: "llm" when the console's model mode
    asked, "agent" under the older mode name. It becomes ``Interpretation.source`` (and so
    ``TranscriptTurn.parser``) and the ``source`` of every record the model made; the reading
    itself and the disk cache are the same for both.

    ``prior_facts`` and ``incident_type`` (what earlier turns already put on the console) are
    shown to the model so it does not spend a reply repeating them.

    ``client`` (an ``anthropic.Anthropic``-like object) and ``meta`` (``room id -> {"suite",
    "tenant"}``, default ``core.evidence.room_meta``) are injectable for tests; the hook passes
    neither. No word rules run: the one deterministic input is the dispatcher's Yes / No / Not
    sure buttons (``button_reading``), kept as they are and shown to the model.
    """
    turn = transcript[index]
    if turn.speaker != "caller":
        return Interpretation(source="")
    source = source if source in ("agent", "llm") else "agent"
    vocab = vocab or _extract.load_vocab()
    idx = BuildingIndex(building, meta, vocab=vocab)
    reading = button_reading(transcript, index, vocab)  # the dispatcher's answer buttons; no word rules
    settings = model_settings()
    system = build_system_prompt(idx, vocab)
    user = build_user_message(building, transcript, index, prior_evidence, reading,
                              prior_facts=prior_facts, incident_type=incident_type)
    key = _cache_key(system, user, settings)

    with _extract._llm_lock:
        cached = _extract._read_json(AGENT_CACHE_PATH, {}).get(key)
    if cached is not None:
        _record_usage(cache_hit=True)
        return _checked(cached["calls"], idx, building, transcript, index, prior_evidence, vocab, reading, source)

    remaining = _budget_remaining()
    if remaining is not None and remaining <= 0:  # do not start a turn that cannot send its first request
        return _fallback(building, transcript, index,
                         f"LLM request budget for this process is spent ({_budget()}; raise LLM_MAX_CALLS)",
                         prior_evidence=prior_evidence, vocab=vocab, source=source)
    try:
        if client is None:
            client = _make_client()
        earlier = [t.text or "" for t in transcript[:index] if t.speaker == "caller"]
        run = _run_loop(client, idx, system, user, turn.text or "", settings, earlier)
    except AgentUnavailable as exc:
        return _fallback(building, transcript, index, str(exc), prior_evidence=prior_evidence, vocab=vocab,
                         source=source)

    if run.cut_short is not None:
        # Requests already paid for are kept: validate the completed rounds as usual, but do
        # not cache a partial result (the same transcript should get a full run next time).
        reason = f"cut short after {run.requests} request(s): {run.cut_short}"
        if run.records == 0:  # nothing recorded yet: the rule path has more to offer
            return _fallback(building, transcript, index, reason, prior_evidence=prior_evidence, vocab=vocab,
                             source=source)
        out = _checked(run.calls, idx, building, transcript, index, prior_evidence, vocab, reading, source)
        if out.source != source:  # the records could not be read: the rule path's reading stands
            return out
        trace = list(out.trace) + [AgentTraceEntry(turn=index, tool="agent", kept=False, reason=reason)]
        return out.model_copy(update={"source": f"{source} (cut short: {run.cut_short})", "trace": trace})

    with _extract._llm_lock:
        cache = _extract._read_json(AGENT_CACHE_PATH, {})
        cache[key] = {"building_id": building.id, "turn": index, "text": turn.text, "model": settings["model"],
                      "settings": settings, "requests": run.requests, "seconds": round(run.seconds, 3),
                      "capped": run.capped, "calls": run.calls}
        _extract._write_json(AGENT_CACHE_PATH, cache)
    return _checked(run.calls, idx, building, transcript, index, prior_evidence, vocab, reading, source)


def prewarm(building: Building, *, vocab: dict | None = None, client: Any = None, meta: dict | None = None
            ) -> Optional[dict]:
    """Write this building's prompt-cache entry so the first caller turn reads it instead of writing it.

    One ``max_tokens=0`` request with the same tools, system block, thinking and effort as a
    real turn: the API writes the cache and returns at once, with no output. Call it when a
    call starts on a building (``api.server`` does, in a background thread). Returns the
    request's token usage, or None when the model is not configured or the request failed;
    it never raises.
    """
    try:
        vocab = vocab or _extract.load_vocab()
        idx = BuildingIndex(building, meta, vocab=vocab)
        settings = model_settings()
        system = build_system_prompt(idx, vocab)
        if client is None:
            client = _make_client()
        _reserve_request()
        started = time.perf_counter()
        response = _create(client, settings, max_tokens=0, messages=[{"role": "user", "content": "warmup"}],
                           **_request_kwargs(system, settings))
        seconds = time.perf_counter() - started
        u = getattr(response, "usage", None)
        _record_usage(u, seconds=seconds, kind="prewarm")
        return {
            "seconds": round(seconds, 3),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
    except Exception:  # a warm-up must never break a call
        return None

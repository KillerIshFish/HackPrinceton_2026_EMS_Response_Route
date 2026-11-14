"""Parser hardening after the interpreter evaluation (C4). Stub clients only: no network."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import parser.agent as agent
import parser.extract as extract
from core.schema import TranscriptTurn
from ingest.matterport import generate_synthetic_office
from core.evidence import selector_hits
from parser.dialog import correction_unit, find_units
from parser.interpret import interpret_turn


@pytest.fixture(scope="module")
def building():
    return generate_synthetic_office()


@pytest.fixture(scope="module")
def patel_room(building):
    # f3-suite-vacant: an office on floor 3 that is NOT near the stairs
    room = next(r for r in building.rooms if r.floor_label == "3" and r.room_type == "office")
    assert room.near_stairs is False and room.window_facing == "parking_lot"
    return room


@pytest.fixture(scope="module")
def meta(patel_room):
    return {patel_room.id: {"suite": "304", "tenant": "Dr. Anika Patel, Family Medicine"}}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_CACHE_PATH", tmp_path / "agent_cache.json")
    monkeypatch.setattr(extract, "LLM_USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(extract, "LLM_CACHE_PATH", tmp_path / "clue_cache.json")
    monkeypatch.setattr(extract, "_llm_calls_made", 0)
    monkeypatch.setenv("LLM_MAX_CALLS", "200")


_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def tool_use(name: str, **args):
    tool_use.n += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_h{tool_use.n}", name=name, input=args)


tool_use.n = 0


def reply(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    usage = SimpleNamespace(input_tokens=50, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


def done():
    return reply(SimpleNamespace(type="text", text="done"))


class StubClient:
    """Pops scripted replies; an Exception in the script is raised instead."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        item = self.script.pop(0) if self.script else done()
        if isinstance(item, BaseException):
            raise item
        return item


def turns(*pairs):
    return [TranscriptTurn(turn=i, title=f"t{i}", speaker=s, text=t) for i, (s, t) in enumerate(pairs)]


CALL = turns(("dispatcher", "911, what is the address of your emergency?"),
             ("caller", "I'm in Dr. Patel's office, my coworker collapsed and he's not breathing"))
PATEL = dict(negate=False, default_match=0.2, confidence=0.9, span="Dr. Patel's office", rationale="tenant")


# ---------------------------------------------------------------------------
# 1. One budget slot is one HTTP request
# ---------------------------------------------------------------------------


class _FakeAnthropic:
    made: list[dict] = []

    def __init__(self, **kwargs):
        _FakeAnthropic.made.append(kwargs)
        self.messages = self

    def create(self, **kwargs):
        body = SimpleNamespace(type="text", text='{"clues": []}')
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(content=[body], stop_reason="end_turn", usage=usage)


def test_clients_are_built_without_retries(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    _FakeAnthropic.made = []
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    assert extract.extract_clues_llm("hardening test: one request per slot") == []
    agent._make_client()
    assert [m["max_retries"] for m in _FakeAnthropic.made] == [0, 0]


# ---------------------------------------------------------------------------
# 2. Budget stops and mid-loop failures
# ---------------------------------------------------------------------------


def test_no_request_is_started_when_the_budget_is_already_spent(building, meta, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS", "3")
    monkeypatch.setattr(extract, "_llm_calls_made", 3)
    client = StubClient()
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert client.calls == []
    assert out.source.startswith("rules (agent unavailable: LLM request budget")


def test_timeout_after_a_completed_round_keeps_its_records(building, meta, patel_room):
    client = StubClient([
        reply(tool_use("search_rooms", query="Patel"),
              tool_use("record_location", selector={"tenant_contains": "Patel"}, **PATEL)),
        anthropic.APITimeoutError(request=_REQ),
    ])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert len(client.calls) == 2
    assert out.source == "agent (cut short: Anthropic API timed out)"
    [ev] = [e for e in out.evidence if e.source == "agent"]
    assert ev.matched_room_ids == [patel_room.id] and ev.confidence == 0.85  # still validated (cap)
    assert out.trace[-1].tool == "agent" and "cut short after 1 request" in out.trace[-1].reason
    # a partial run is never written to the final-result cache
    assert extract._read_json(agent.AGENT_CACHE_PATH, {}) == {}


def test_budget_running_out_mid_loop_keeps_the_paid_round(building, meta, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS", "1")
    # the look-up makes the loop want a second request, which the budget does not allow
    client = StubClient([reply(tool_use("search_rooms", query="Patel"),
                               tool_use("record_location", selector={"tenant_contains": "Patel"}, **PATEL))])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert len(client.calls) == 1
    assert out.source.startswith("agent (cut short: LLM request budget")
    assert any(e.source == "agent" for e in out.evidence)
    assert extract._read_json(agent.AGENT_CACHE_PATH, {}) == {}


def test_failure_after_lookups_only_falls_back_to_rules(building, meta):
    error = anthropic.APIConnectionError(request=_REQ)
    client = StubClient([reply(tool_use("search_rooms", query="Patel")), error])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert out.source.startswith("rules (agent unavailable: cut short after 1 request(s): Anthropic API unreachable")
    assert extract._read_json(agent.AGENT_CACHE_PATH, {}) == {}


# ---------------------------------------------------------------------------
# 3. Empty within_m_of is absent
# ---------------------------------------------------------------------------


def test_empty_within_m_of_counts_as_absent(building, meta, patel_room):
    sel = {"tenant_contains": "Patel", "within_m_of": {"room_id": "", "radius_m": 0}}
    client = StubClient([reply(tool_use("record_location", selector=sel, **PATEL)), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert len(client.calls) == 1  # the record is usable as it stands: no warning, so no second request
    [ev] = out.evidence
    assert ev.selector.within_m_of is None and ev.matched_room_ids == [patel_room.id]
    assert out.trace[0].kept and "empty within_m_of ignored" in out.trace[0].reason


# ---------------------------------------------------------------------------
# 4. Conditions nobody said are stripped
# ---------------------------------------------------------------------------


def test_unsupported_feature_conditions_are_stripped(building, meta, patel_room):
    sel = {"tenant_contains": "Patel", "near_stairs": True, "near_elevator": False, "window_facing": "street"}
    client = StubClient([reply(tool_use("record_location", selector=sel, **PATEL)), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    [ev] = out.evidence  # near_stairs=True alone would have excluded the caller's room
    assert ev.matched_room_ids == [patel_room.id]
    assert ev.selector.near_stairs is None and ev.selector.near_elevator is None and ev.selector.window_facing is None
    reason = out.trace[0].reason
    assert "stripped" in reason and "near_stairs=True" in reason and "window_facing='street'" in reason


def test_supported_feature_conditions_are_kept(building, meta, patel_room):
    call = turns(("dispatcher", "Where are you?"),
                 ("caller", "Dr. Patel's office, not by the stairs, the window looks out on the parking lot"))
    sel = {"tenant_contains": "Patel", "near_stairs": False, "window_facing": "parking_lot", "has_windows": True}
    client = StubClient([reply(tool_use("record_location", selector=sel, **PATEL)), done()])
    out = agent.interpret(building, call, 1, client=client, meta=meta)
    [ev] = [e for e in out.evidence if e.source == "agent"]
    assert ev.selector.near_stairs is False and ev.selector.window_facing == "parking_lot"
    assert ev.selector.has_windows is True
    assert not (out.trace[0].reason or "").count("stripped")


def test_selector_with_only_unsupported_conditions_is_dropped(building, meta):
    client = StubClient([reply(tool_use("record_location", selector={"near_elevator": True}, **PATEL)), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert [e for e in out.evidence if e.source == "agent"] == []
    assert out.trace[0].reason.startswith("selector has no condition") and "stripped" in out.trace[0].reason


# ---------------------------------------------------------------------------
# 5. Usage file under the lock; timeout and unexpected errors in extract.py
# ---------------------------------------------------------------------------


def test_agent_usage_is_written_under_the_shared_lock(monkeypatch):
    monkeypatch.setattr(extract, "_llm_lock", threading.Lock())
    real_write = extract._write_json
    held = []

    def write(path, data):
        held.append(extract._llm_lock.locked())
        real_write(path, data)

    monkeypatch.setattr(extract, "_write_json", write)
    agent._record_usage(SimpleNamespace(input_tokens=3, output_tokens=1))
    agent._record_usage(cache_hit=True)
    assert held == [True, True]
    assert extract.llm_usage()["agent_calls"] == 1 and extract.llm_usage()["cache_hits"] == 1


def _raising_anthropic(error):
    class Client:
        def __init__(self, **kwargs):
            self.messages = self

        def create(self, **kwargs):
            raise error

    return Client


def test_extract_reports_timeout_and_survives_unexpected_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", _raising_anthropic(anthropic.APITimeoutError(request=_REQ)))
    clues, source = extract.extract_clues_with_source("I'm on the third floor", use_llm=True)
    assert source == "rules (llm unavailable: Anthropic API timed out)"
    assert [c.value for c in clues] == ["3"]

    monkeypatch.setattr(anthropic, "Anthropic", _raising_anthropic(ValueError("boom")))
    _, source = extract.extract_clues_with_source("I'm on the fourth floor", use_llm=True)
    assert source == "rules (llm unavailable: unexpected error: ValueError: boom)"

    class NoUsage(_FakeAnthropic):  # a reply that breaks after the request: still the rules
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"clues": []}')],
                                   stop_reason="end_turn", usage=None)

    monkeypatch.setattr(anthropic, "Anthropic", NoUsage)
    _, source = extract.extract_clues_with_source("I'm on the second floor", use_llm=True)
    assert source.startswith("rules (llm unavailable: unexpected error: AttributeError")


# ---------------------------------------------------------------------------
# 6. Dialog rules
# ---------------------------------------------------------------------------


def _units(it):
    return [(e.selector.unit, e.status) for e in it.evidence if e.selector is not None]


def test_bare_number_correction_supersedes_the_earlier_unit(building):
    # 350 and 450 are numbers no room of the synthetic building carries: each is kept as an
    # unmatched unit with a floor fallback from its leading digit
    call = turns(("dispatcher", "What's the suite number?"), ("caller", "I'm in suite 350"),
                 ("dispatcher", "Three fifty?"), ("caller", "sorry I meant 450, four fifty"))
    first = interpret_turn(building, call, 1, "rules")
    second = interpret_turn(building, call, 3, "rules", prior_evidence=first.evidence)
    assert _units(second) == [("450", "active")]
    assert set(second.supersedes) == {e.id for e in first.evidence}  # unit 350 and its floor-3 fallback
    floors = [e.clue.value for e in second.evidence if e.clue is not None and e.clue.type == "floor"]
    assert floors == ["4"]


def test_corrected_number_printed_on_a_room_needs_no_floor_fallback(building):
    # 310 ("Suite 310 (vacant)") and 402 ("Executive Office 402") are printed on rooms of the
    # building, so each resolves to its room and no floor is guessed from the digits
    call = turns(("dispatcher", "What's the suite number?"), ("caller", "I'm in suite 310"),
                 ("dispatcher", "Three ten?"), ("caller", "sorry I meant 402, four oh two"))
    first = interpret_turn(building, call, 1, "rules")
    assert [e.selector.unit for e in first.evidence] == ["310"]
    second = interpret_turn(building, call, 3, "rules", prior_evidence=first.evidence)
    assert _units(second) == [("402", "active")]
    assert set(second.supersedes) == {e.id for e in first.evidence}
    assert [e for e in second.evidence if e.clue is not None and e.clue.type == "floor"] == []
    assert selector_hits(building, second.evidence[0].selector) == ["f4-office-402"]


def test_no_wait_its_number_after_an_open_unit_question(building):
    from core.schema import Question

    call = [TranscriptTurn(turn=0, title="t0", speaker="dispatcher", text="What unit?",
                           question=Question(id="open:unit", question="What unit?", kind="open", partition="unit")),
            *turns(("caller", "unit 307"), ("dispatcher", "Three oh seven?"),
                   ("caller", "no wait, it's 402, I'm at my mom's place"))]
    call = [t.model_copy(update={"turn": i}) for i, t in enumerate(call)]
    first = interpret_turn(building, call, 1, "rules")
    second = interpret_turn(building, call, 3, "rules", prior_evidence=first.evidence)
    assert _units(second) == [("402", "active")]
    assert first.evidence[0].id in second.supersedes


def test_bare_number_correction_needs_a_unit_to_correct(building):
    # nothing earlier named a unit: "402" could be anything, so it is not read as one
    call = turns(("dispatcher", "Where are you?"), ("caller", "I'm on the third floor"),
                 ("caller", "no wait, it's 402"))
    assert correction_unit(call, 2) is None
    assert _units(interpret_turn(building, call, 2, "rules")) == []
    # single digits and durations stay out even with a unit to correct
    call = turns(("caller", "suite 310"), ("caller", "sorry I meant 4"), ("caller", "actually 15 minutes ago"))
    assert correction_unit(call, 1) is None and correction_unit(call, 2) is None


def test_across_the_hall_and_short_units():
    assert [m.unit for m in find_units("we're not in suite 520, we're across the hall in 530")] == ["530"]
    assert [m.unit for m in find_units("across the hall in 530")] == ["530"]
    assert [m.unit for m in find_units("she's across the hall in 530")] == ["530"]
    assert [m.unit for m in find_units("we're in 2B")] == ["2B"]
    # not the caller's location / not a unit
    assert find_units("the defibrillator is across the hall in 530") == []
    assert find_units("I'm in 2") == [] and find_units("we're at 4am") == []


def test_room_names_with_numbers_are_not_units():
    assert find_units("I'm in conference room 2") == []
    assert find_units("we're in break room 3 on the fourth floor") == []
    assert find_units("rest room 2, by the elevator") == []
    assert [m.unit for m in find_units("I'm in room 204")] == ["204"]
    assert [m.unit for m in find_units("office 303")] == ["303"]


def test_negation_guard_still_holds(building):
    assert find_units("we're not in suite 304") == []
    it = interpret_turn(building, turns(("caller", "we're not across the hall in 530")), 0, "rules")
    assert _units(it) == []

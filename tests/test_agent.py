"""Interpreter agent (parser/agent.py) with a stub client: scripted tool calls, no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import parser.agent as agent
import parser.extract as extract
from core.fusion import names_rooms, posterior_probs
from core.schema import Clue, Evidence, RoomSelector, TranscriptTurn
from ingest.matterport import generate_synthetic_office
from parser.building_tools import TOOLS, BuildingIndex, floor_summary, rooms_near, search_rooms, select_rooms


@pytest.fixture(scope="module")
def building():
    return generate_synthetic_office()


@pytest.fixture(scope="module")
def patel_room(building):
    return next(r for r in building.rooms if r.floor_label == "3" and r.room_type == "office")


@pytest.fixture(scope="module")
def meta(building, patel_room):
    other = next(r for r in building.rooms if r.floor_label == "2" and r.room_type == "office")
    return {
        patel_room.id: {"suite": "304", "tenant": "Dr. Anika Patel, Family Medicine"},
        other.id: {"suite": "210", "tenant": "Harbor Insurance Group"},
    }


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep the disk cache, usage file and request budget away from the real ones."""
    monkeypatch.setattr(agent, "AGENT_CACHE_PATH", tmp_path / "agent_cache.json")
    monkeypatch.setattr(extract, "LLM_USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(extract, "_llm_calls_made", 0)
    monkeypatch.setenv("LLM_MAX_CALLS", "200")


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


def tool_use(name: str, **args):
    tool_use.n += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_{tool_use.n}", name=name, input=args)


tool_use.n = 0


def reply(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    usage = SimpleNamespace(input_tokens=50, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


def done():
    return reply(SimpleNamespace(type="text", text="done"))


class StubClient:
    def __init__(self, script=None, forever=None, error=None):
        self.script = list(script or [])
        self.forever = forever
        self.error = error
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        # snapshot the messages: the agent keeps appending to the same list
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        if self.error is not None:
            raise self.error
        if self.forever is not None:
            return self.forever()
        return self.script.pop(0) if self.script else done()


def turns(*pairs):
    return [TranscriptTurn(turn=i, title=f"t{i}", speaker=s, text=t) for i, (s, t) in enumerate(pairs)]


CALL = turns(("dispatcher", "911, what is the address of your emergency?"),
             ("caller", "I'm in Dr. Patel's office, my coworker collapsed and he's not breathing"))


# ---------------------------------------------------------------------------
# Building tools
# ---------------------------------------------------------------------------


def test_search_rooms_finds_tenant_and_suite(building, meta, patel_room):
    idx = BuildingIndex(building, meta)
    by_name = search_rooms(idx, "Dr. Patel's office")
    assert by_name["rooms"][0]["id"] == patel_room.id
    assert by_name["rooms"][0]["suite"] == "304"
    assert search_rooms(idx, "suite 304")["rooms"][0]["id"] == patel_room.id
    assert search_rooms(idx, "Patel", floor_label="1")["total"] == 0


def test_rooms_near_and_floor_summary(building, meta, patel_room):
    idx = BuildingIndex(building, meta)
    near = rooms_near(idx, patel_room.id, 30.0)
    assert near["total"] > 0
    assert all(r["floor"] == "3" and r["walk_m"] <= 30.0 for r in near["rooms"])
    assert "error" in rooms_near(idx, "no-such-room", 10.0)
    summary = floor_summary(idx, "3")
    assert summary["suites"]["304"] == ["Dr. Anika Patel, Family Medicine"]
    assert "error" in floor_summary(idx, "99")


def test_select_rooms_conditions(building, meta, patel_room):
    idx = BuildingIndex(building, meta)
    assert select_rooms(idx, RoomSelector(tenant_contains="patel")) == [patel_room.id]
    assert select_rooms(idx, RoomSelector(unit="Suite 304")) == [patel_room.id]
    assert select_rooms(idx, RoomSelector(tenant_contains="patel", floor_labels=["2"])) == []
    floor3 = select_rooms(idx, RoomSelector(floor_labels=["3"]))
    assert floor3 and all(idx.rooms[r].floor_label == "3" for r in floor3)


def test_tool_schemas_are_strict():
    for tool in TOOLS:
        assert tool["strict"] is True
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])


# ---------------------------------------------------------------------------
# Agent loop and validation
# ---------------------------------------------------------------------------


def test_patel_office_grounds_to_tenant_room(building, meta, patel_room):
    client = StubClient([
        reply(tool_use("search_rooms", query="Patel")),
        reply(
            tool_use("record_location", selector={"tenant_contains": "Patel"}, negate=False,
                     confidence=0.95, certainty=1.0, specificity=1.0, span="Dr. Patel's office",
                     rationale="tenant named by caller"),
            tool_use("record_fact", category="patient", text="Caller reports coworker collapsed, not breathing",
                     span="he's not breathing"),
            tool_use("suggest_incident_type", type="cardiac_arrest", span="he's not breathing"),
        ),
        done(),
    ])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert out.source == "agent"
    # the look-up needed an answer; the records that followed were usable, so the loop stopped
    # there: no third request just to hear "done"
    assert len(client.calls) == 2
    [ev] = out.evidence
    assert ev.kind == "grounded" and ev.source == "agent" and ev.id == "t1-e0"
    assert ev.matched_room_ids == [patel_room.id]
    # the caller named the tenant ("Patel" is in the span) plainly and exactly: the top stated share
    assert ev.confidence == 0.95
    assert ev.default_match == 0.0 and names_rooms(ev)  # fused as a statement that names rooms
    [fact] = out.facts
    assert fact.verified is False and fact.id == "t1-f0" and fact.category == "patient"
    assert out.suggested_type == "cardiac_arrest" and out.suggested_span == "he's not breathing"
    assert [t.tool for t in out.trace] == ["search_rooms", "record_location", "record_fact", "suggest_incident_type"]
    # the look-up result went back as one user message of tool results
    last_user = client.calls[1]["messages"][-1]
    assert last_user["role"] == "user" and [b["type"] for b in last_user["content"]] == ["tool_result"]
    # system block and tools are identical on every request (so they cache)
    assert all(c["system"] == client.calls[0]["system"] and c["tools"] == client.calls[0]["tools"] for c in client.calls)
    assert client.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert client.calls[0]["model"] == "claude-haiku-5-5"


def test_fabricated_span_is_dropped(building, meta):
    client = StubClient([reply(
        tool_use("record_vocab_clue", type="floor", value="3", polarity=True, confidence=0.9, span="third floor"),
        tool_use("record_fact", category="threat", text="Caller reports a man with a gun", span="he has a gun"),
    ), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert out.evidence == [] and out.facts == []
    assert all(not t.kept and "verbatim" in t.reason for t in out.trace)


def test_unknown_room_id_is_dropped(building, meta):
    client = StubClient([reply(
        tool_use("record_location", selector={"room_ids": ["f9-suite-999"]}, negate=False, default_match=0.1,
                 confidence=0.9, span="Dr. Patel's office", rationale="made up"),
        tool_use("record_location", selector={"tenant_contains": "Nobody Inc"}, negate=False, default_match=0.1,
                 confidence=0.9, span="Dr. Patel's office", rationale="no match"),
    ), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert out.evidence == []
    assert "unknown room id" in out.trace[0].reason
    assert out.trace[1].reason == "selector matches no room"
    # the acknowledgement already warned the model
    ack = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "will be dropped" in ack


def test_low_confidence_unknown_vocab_and_verified_fact_dropped(building, meta):
    client = StubClient([reply(
        tool_use("record_vocab_clue", type="room_type", value="office", polarity=True, confidence=0.3,
                 span="Dr. Patel's office"),
        tool_use("record_vocab_clue", type="room_type", value="spaceship", polarity=True, confidence=0.9,
                 span="Dr. Patel's office"),
        tool_use("record_fact", category="patient", text="Confirmed cardiac arrest", span="he's not breathing"),
        tool_use("record_vocab_clue", type="room_type", value="office", polarity=True, confidence=0.8,
                 span="dr. patel's OFFICE"),
    ), done()])
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    kept = [t.kept for t in out.trace]
    assert kept == [False, False, False, True]
    [ev] = out.evidence
    assert ev.kind == "vocab" and ev.clue.value == "office"
    assert ev.span == "Dr. Patel's office"  # caller's own casing recovered


def test_supersede_and_denial(building, meta):
    prior = Evidence(id="t1-e0", kind="vocab", confidence=0.8, span="third floor", turn=1,
                     clue=Clue(type="floor", value="3", polarity=True, confidence=0.8, span="third floor"))
    call = turns(("dispatcher", "911"), ("caller", "we're on the third floor"),
                 ("caller", "sorry, I meant the second floor, we're not in suite 304"))
    client = StubClient([reply(
        tool_use("supersede", evidence_id="t1-e0", span="sorry, I meant the second floor"),
        tool_use("supersede", evidence_id="t0-e7", span="sorry, I meant the second floor"),
        tool_use("record_location", selector={"unit": "304"}, negate=True, default_match=0.4, confidence=0.8,
                 span="we're not in suite 304", rationale="denial"),
    ), done()])
    out = agent.interpret(building, call, 2, prior_evidence=[prior], client=client, meta=meta)
    assert "unknown evidence id" in out.trace[1].reason
    user_msg = client.calls[0]["messages"][0]["content"]
    assert "t1-e0" in user_msg  # earlier ids are shown to the model
    # the unit rule (parser.dialog) leaves a negated mention alone ("not in suite 304"),
    # so the only unit evidence is the agent's denial
    assert not any(e.source == "rules" and e.selector is not None and e.selector.unit == "304"
                   for e in out.evidence)
    [denial] = [e for e in out.evidence if e.source == "agent" and e.selector is not None]
    assert denial.negate is True and denial.default_match == 1.0
    assert denial.status == "active"
    assert out.supersedes[0] == "t1-e0"


def test_dialog_answer_comes_first_and_is_not_duplicated(building, meta):
    call = turns(("dispatcher", "Is there a window in the room you're in?"),
                 ("caller", "yeah, and I can hear sirens outside"))
    client = StubClient([reply(
        tool_use("record_vocab_clue", type="landmark", value="window", polarity=True, confidence=0.8, span="yeah"),
    ), done()])
    out = agent.interpret(building, call, 1, client=client, meta=meta)
    [ev] = out.evidence
    assert ev.source == "rules" and ev.id == "t1-e0" and ev.clue.value == "window" and ev.span == "yeah"
    assert not out.trace[0].kept and "dialog rules" in out.trace[0].reason
    assert "landmark=window" in client.calls[0]["messages"][0]["content"]


def test_fourth_request_is_never_followed_by_a_fifth(building, meta):
    client = StubClient(forever=lambda: reply(tool_use("search_rooms", query="Patel")))
    out = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert len(client.calls) == agent.AGENT_MAX_REQUESTS_PER_TURN == 4
    assert out.source == "agent"
    assert len(out.trace) == 4


def test_api_error_falls_back_to_rules(building, meta):
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    out = agent.interpret(building, CALL, 1, client=StubClient(error=error), meta=meta)
    assert out.source.startswith("rules (agent unavailable:")
    assert "unreachable" in out.source
    from parser.interpret import interpret_turn

    rules = interpret_turn(building, CALL, 1, "rules")
    assert [e.model_dump() for e in out.evidence] == [e.model_dump() for e in rules.evidence]


def test_refusal_and_budget_fall_back(building, meta, monkeypatch):
    out = agent.interpret(building, CALL, 1, client=StubClient([reply(stop="refusal")]), meta=meta)
    assert out.source == "rules (agent unavailable: model refused)"
    monkeypatch.setenv("LLM_MAX_CALLS", "0")
    client = StubClient()
    out = agent.interpret(building, turns(("caller", "help me please")), 0, client=client, meta=meta)
    assert out.source.startswith("rules (agent unavailable: LLM request budget")
    assert client.calls == []


def test_same_transcript_twice_makes_no_second_request(building, meta):
    script = [reply(tool_use("record_vocab_clue", type="room_type", value="office", polarity=True, confidence=0.8,
                             span="Dr. Patel's office")), done()]
    client = StubClient(script)
    first = agent.interpret(building, CALL, 1, client=client, meta=meta)
    n = len(client.calls)
    second = agent.interpret(building, CALL, 1, client=client, meta=meta)
    assert len(client.calls) == n == 1  # one request for the turn, none for the repeat
    assert [e.model_dump() for e in first.evidence] == [e.model_dump() for e in second.evidence]
    assert extract.llm_usage()["cache_hits"] == 1


def test_non_caller_turn_and_missing_key(building, meta, monkeypatch):
    assert agent.interpret(building, CALL, 0, meta=meta).evidence == []
    monkeypatch.setenv("LLM_PROVIDER", "")
    out = agent.interpret(building, CALL, 1, meta=meta)  # no client injected, no provider: never reaches the network
    assert out.source.startswith("rules (agent unavailable: LLM_PROVIDER")


def test_span_from_an_earlier_turn_is_dropped_as_already_read(building, meta):
    call = CALL + turns(("dispatcher", "Is anyone doing CPR?"), ("caller", "yes my friend is"))[1:]
    call = [t.model_copy(update={"turn": i}) for i, t in enumerate(call)]
    client = StubClient([reply(
        tool_use("suggest_incident_type", type="cardiac_arrest", span="he's not breathing"),
        tool_use("record_fact", category="patient", text="Caller reports a friend is doing CPR", span="yes my friend is"),
    ), done()])
    out = agent.interpret(building, call, 2, client=client, meta=meta)
    assert out.suggested_type is None
    assert "earlier caller turn" in out.trace[0].reason
    assert [f.span for f in out.facts] == ["yes my friend is"]


def test_search_uses_vocabulary_synonyms_for_room_types(building, meta):
    idx = BuildingIndex(building, meta, vocab=extract.load_vocab())
    types = {r["type"] for r in search_rooms(idx, "front desk")["rooms"][:2]}
    assert types & {"lobby", "reception"}

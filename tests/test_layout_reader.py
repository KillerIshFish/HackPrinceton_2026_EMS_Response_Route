"""The layout-reasoning reader (parser/agent.py, parser/layout.py, parser/building_tools.py).

``record_clue`` and ``record_floor`` checked in code, the beta fallback path of ``_create``,
and the layout map the model reasons over (hallway door order, size rank, position on the
floor plate, ``find_rooms``). Stub clients only: no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import parser.agent as agent
import parser.extract as extract
from core.fusion import names_rooms
from core.schema import Clue, Evidence, RoomSelector, TranscriptTurn
from ingest.matterport import generate_synthetic_office
from parser.building_tools import BuildingIndex, find_rooms, layout_roster, room_details, run_lookup
from parser.layout import CIRCULATION, build_layout


@pytest.fixture(scope="module")
def building():
    return generate_synthetic_office()  # ptc-100


@pytest.fixture(scope="module")
def idx(building):
    return BuildingIndex(building, None)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep the disk cache, usage file and request counter away from the real ones."""
    monkeypatch.setattr(agent, "AGENT_CACHE_PATH", tmp_path / "agent_cache.json")
    monkeypatch.setattr(extract, "LLM_USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(extract, "_llm_calls_made", 0)
    monkeypatch.delenv("LLM_MAX_CALLS", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


def tool_use(name: str, **args):
    tool_use.n += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_lr{tool_use.n}", name=name, input=args)


tool_use.n = 0


def reply(*blocks):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    usage = SimpleNamespace(input_tokens=50, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


class StubClient:
    """``client.messages.create`` only (no ``beta`` namespace)."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        if self.script:
            return self.script.pop(0)
        return reply(SimpleNamespace(type="text", text="nothing to record"))


class BetaStubClient:
    """Both ``client.messages.create`` and ``client.beta.messages.create``; records which was used."""

    def __init__(self, script=()):
        self.script = list(script)
        self.plain: list[dict] = []
        self.beta_calls: list[dict] = []
        self.messages = SimpleNamespace(create=lambda **kw: self._answer(self.plain, kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._answer(self.beta_calls, kw)))

    def _answer(self, log, kwargs):
        log.append(dict(kwargs, messages=list(kwargs["messages"])))
        if self.script:
            return self.script.pop(0)
        return reply(SimpleNamespace(type="text", text="nothing to record"))


def turns(*pairs):
    return [TranscriptTurn(turn=i, title=f"t{i}", speaker=s, text=t) for i, (s, t) in enumerate(pairs)]


CORNER = turns(("dispatcher", "Where are you?"),
               ("caller", "I'm in the corner office at the end of the south hall, there's no window in here, "
                          "it's on the third floor I think"))
CORNER_ROOMS = ["f3-office-301", "f3-office-303"]


def read(building, call, *blocks, prior_evidence=()):
    client = StubClient([reply(*blocks)])
    out = agent.interpret(building, call, len(call) - 1, client=client, prior_evidence=prior_evidence)
    return out, client


def clue_args(**over):
    args = dict(clue="corner office at the end of the south hall", room_ids=list(CORNER_ROOMS), others_fit=0.2,
                negate=False, confidence=0.8, span="the corner office at the end of the south hall",
                reasoning="301 and 303 are the corner offices at either end of the south corridors")
    args.update(over)
    return args


# ---------------------------------------------------------------------------
# record_clue
# ---------------------------------------------------------------------------


def test_record_clue_is_descriptive_grounded_evidence(building):
    out, client = read(building, CORNER, tool_use("record_clue", **clue_args()))
    assert len(client.calls) == 1
    [ev] = out.evidence
    assert ev.kind == "grounded" and ev.source == "agent" and ev.id == "t1-e0"
    assert ev.selector == RoomSelector(room_ids=CORNER_ROOMS) and ev.matched_room_ids == CORNER_ROOMS
    assert ev.default_match == pytest.approx(0.2) and ev.negate is False
    assert not names_rooms(ev)  # a description, not a statement that names rooms
    assert ev.confidence == pytest.approx(0.8)
    assert ev.rationale.startswith("corner office at the end of the south hall: 301 and 303")
    assert out.trace[0].kept


@pytest.mark.parametrize("given, kept", [(0.0, agent.CLUE_OTHERS_FIT_MIN), (-3, agent.CLUE_OTHERS_FIT_MIN),
                                         (0.3, 0.3), (0.9, agent.CLUE_OTHERS_FIT_MAX), ("loose", 0.2)])
def test_record_clue_others_fit_is_clamped(building, given, kept):
    call = [*CORNER[:1], CORNER[1].model_copy(update={"text": CORNER[1].text + f" ({given})"})]  # distinct cache key
    many = [r.id for r in building.rooms if r.floor_label == "3" and r.room_type not in CIRCULATION][:8]
    assert len(many) > agent.PINPOINT_MAX_ROOMS  # more rooms than a pinpointing clue may leave: stays descriptive
    out, _ = read(building, call, tool_use("record_clue", **clue_args(others_fit=given, room_ids=many)))
    [ev] = out.evidence
    assert ev.default_match == pytest.approx(kept) and ev.default_match > 0.0 and not names_rooms(ev)


def test_a_clue_that_rules_the_others_out_places_the_caller(building):
    # others_fit <= 0.1 and at most five rooms: a room worked out from a description (share <= 0.7)
    out, _ = read(building, CORNER, tool_use("record_clue", **clue_args(others_fit=0.05, confidence=0.85)))
    [ev] = out.evidence
    assert ev.default_match == 0.0 and names_rooms(ev) and ev.matched_room_ids == CORNER_ROOMS
    assert ev.confidence == agent.DESCRIBED_CONFIDENCE_CAP == 0.7 and "pinpointing clue" in out.trace[0].reason
    # a loose clue over the same rooms stays a soft factor
    call = [*CORNER[:1], CORNER[1].model_copy(update={"text": CORNER[1].text + " ok"})]
    out, _ = read(building, call, tool_use("record_clue", **clue_args(others_fit=0.11)))
    assert not names_rooms(out.evidence[0]) and out.evidence[0].default_match == pytest.approx(0.11)


def test_record_clue_denial_leaves_the_other_rooms_alone(building):
    windowless = [r.id for r in building.rooms if r.floor_label == "3" and not r.has_windows
                  and r.room_type not in CIRCULATION]
    windowed = [r.id for r in building.rooms if r.floor_label == "3" and r.has_windows]
    out, _ = read(building, CORNER, tool_use("record_clue", **clue_args(
        clue="no window", room_ids=windowed, others_fit=0.3, negate=True, span="there's no window in here",
        reasoning="rules out the rooms with windows")))
    [ev] = out.evidence
    assert ev.negate is True and ev.default_match == 1.0 and not names_rooms(ev)
    assert ev.matched_room_ids == windowed and not set(windowless) & set(ev.matched_room_ids)


def test_record_clue_with_an_unknown_or_no_room_is_dropped(building):
    out, client = read(
        building, CORNER,
        tool_use("record_clue", **clue_args(room_ids=["f3-office-301", "f3-office-399"])),
        tool_use("record_clue", **clue_args(room_ids=[])),
        tool_use("record_clue", **clue_args(room_ids=["", "  "])),
    )
    assert out.evidence == []
    assert [t.kept for t in out.trace] == [False, False, False]
    assert "unknown room id" in out.trace[0].reason and "f3-office-399" in out.trace[0].reason
    assert out.trace[1].reason == out.trace[2].reason == "clue lists no room"


def test_record_clue_duplicates_are_dropped(building):
    prior = Evidence(id="t0-e0", kind="grounded", selector=RoomSelector(room_ids=list(reversed(CORNER_ROOMS))),
                     default_match=0.2, confidence=0.7, span="corner", turn=0, source="agent")
    out, _ = read(building, CORNER, tool_use("record_clue", **clue_args()), prior_evidence=[prior])
    assert out.evidence == [] and "duplicate of earlier evidence t0-e0" in out.trace[0].reason
    # the same rooms twice in one turn: the second is dropped; a denial of them is a different statement
    call = [*CORNER[:1], CORNER[1].model_copy(update={"text": CORNER[1].text + "."})]
    out, _ = read(building, call, tool_use("record_clue", **clue_args()),
                  tool_use("record_clue", **clue_args(span="at the end of the south hall", room_ids=["f3-office-303",
                                                                                                      "f3-office-301"])),
                  tool_use("record_clue", **clue_args(negate=True)))
    assert [t.kept for t in out.trace] == [True, False, True]
    assert "duplicate within this turn" in out.trace[1].reason


def test_record_clue_confidence_is_capped(building):
    out, _ = read(building, CORNER, tool_use("record_clue", **clue_args(confidence=1.0)))
    [ev] = out.evidence
    assert ev.confidence == agent.CLUE_CONFIDENCE_CAP == 0.9
    assert "capped" in out.trace[0].reason
    call = [*CORNER[:1], CORNER[1].model_copy(update={"text": CORNER[1].text + "!"})]
    out, _ = read(building, call, tool_use("record_clue", **clue_args(confidence=0.39)))
    assert out.evidence == [] and "below" in out.trace[0].reason


# ---------------------------------------------------------------------------
# record_floor
# ---------------------------------------------------------------------------


def floor_args(**over):
    args = dict(floor="3", stated_by_caller=True, polarity=True, confidence=0.9, certainty=1.0, specificity=1.0,
                span="it's on the third floor I think")
    args.update(over)
    return args


def test_record_floor_stated_carries_the_stated_share(building):
    out, _ = read(building, CORNER, tool_use("record_floor", **floor_args(certainty=0.5)))
    [ev] = out.evidence
    assert ev.kind == "vocab" and ev.clue.type == "floor" and ev.clue.value == "3" and ev.clue.polarity
    assert ev.confidence == pytest.approx(0.675) == pytest.approx(ev.clue.confidence)  # stated_share(0.5, 1.0)
    assert names_rooms(ev) and "rated by Claude Sonnet 5.5" in ev.rationale


def test_record_floor_worked_out_is_capped(building):
    out, _ = read(building, CORNER, tool_use("record_floor", **floor_args(stated_by_caller=False, confidence=0.99)))
    [ev] = out.evidence
    assert ev.clue.type == "floor" and ev.confidence == agent.DESCRIBED_CONFIDENCE_CAP == 0.7
    assert "not stated by the caller" in ev.rationale and "capped" in out.trace[0].reason


def test_record_floor_denial_is_a_negative_clue(building):
    call = turns(("caller", "we're not on four, I'm sure of that"))
    out, _ = read(building, call, tool_use("record_floor", **floor_args(
        floor="4", polarity=False, confidence=0.99, span="we're not on four")))
    [ev] = out.evidence
    assert ev.clue.type == "negative" and ev.clue.value == "floors:4" and ev.clue.polarity is False
    assert ev.confidence == agent.GROUNDED_CONFIDENCE_CAP and not names_rooms(ev)
    # the same denial already on the console is not recorded again
    prior = Evidence(id="t0-e0", kind="vocab", confidence=0.85, span="not on four", turn=0, source="agent",
                     clue=Clue(type="negative", value="floors:4", polarity=False, confidence=0.85, span="not on four"))
    call = turns(("caller", "not on four"), ("caller", "we're not on four, I'm sure of that"))
    out, _ = read(building, call, tool_use("record_floor", **floor_args(
        floor="4", polarity=False, span="we're not on four")), prior_evidence=[prior])
    assert out.evidence == [] and "duplicate of earlier evidence t0-e0" in out.trace[0].reason


def test_record_floor_must_be_a_floor_of_the_building(building):
    out, client = read(building, CORNER, tool_use("record_floor", **floor_args(floor="9")),
                       tool_use("record_floor", **floor_args(floor="Top", span="third floor")))
    assert [t.kept for t in out.trace] == [False, True]
    assert "'9' is not a floor of this building" in out.trace[0].reason
    [ev] = out.evidence
    assert ev.clue.type == "floor" and ev.clue.value == "top"  # relative floors are accepted, lower-cased
    assert names_rooms(ev)


# ---------------------------------------------------------------------------
# _create: server-side fallbacks (beta) where the model and the client have them
# ---------------------------------------------------------------------------


def test_create_uses_the_beta_fallback_path_on_opus(monkeypatch):
    client = BetaStubClient()
    agent._create(client, {"model": "claude-opus-5-5"}, max_tokens=5, messages=[{"role": "user", "content": "x"}])
    [call] = client.beta_calls
    assert client.plain == []
    assert call["betas"] == [agent.FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default" and call["max_tokens"] == 5
    # with no model in the settings, the process's model decides
    agent._create(client, {}, max_tokens=5, messages=[])
    assert len(client.beta_calls) == 2


def test_create_uses_the_plain_path_on_haiku_and_on_stubs_without_beta(monkeypatch):
    client = BetaStubClient()
    agent._create(client, {"model": "claude-haiku-5-5"}, max_tokens=5, messages=[])
    [call] = client.plain
    assert client.beta_calls == [] and "betas" not in call and "fallbacks" not in call
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    agent._create(client, {}, max_tokens=5, messages=[])
    assert len(client.plain) == 2 and client.beta_calls == []
    plain_only = StubClient()
    agent._create(plain_only, {"model": "claude-opus-5-5"}, max_tokens=5, messages=[])
    assert len(plain_only.calls) == 1 and "betas" not in plain_only.calls[0]


def test_a_live_turn_on_the_default_model_goes_through_the_beta_path(building):
    client = BetaStubClient([reply(tool_use("record_clue", **clue_args()))])
    out = agent.interpret(building, CORNER, 1, client=client)
    assert client.plain == [] and len(client.beta_calls) == 1
    request = client.beta_calls[0]
    assert request["model"] == agent.DEFAULT_AGENT_MODEL and request["fallbacks"] == "default"
    assert request["thinking"] == {"type": "adaptive"} and request["max_tokens"] == 16000
    assert [e.matched_room_ids for e in out.evidence] == [CORNER_ROOMS]


# ---------------------------------------------------------------------------
# The layout map (parser.layout) on ptc-100
# ---------------------------------------------------------------------------


def ids_of(result: dict) -> list[str]:
    return [line.split(" | ")[0] for line in result["rooms"]]


def test_hallway_door_order_follows_the_corridor_axis(idx):
    lay = build_layout(idx)
    assert lay.halls
    for h in lay.halls.values():
        hall = idx.rooms[h.id]
        along_x = h.axis == "plan W-E"
        coords = [idx.rooms[rid].centroid[0 if along_x else 1] for rid, _ in h.order]
        assert coords == sorted(coords), h.id
        mid = hall.centroid[1 if along_x else 0]
        for rid, side in h.order:
            c = idx.rooms[rid].centroid[1 if along_x else 0]
            assert side in (("N", "S") if along_x else ("E", "W"))
            assert (c > mid) == (side in ("N", "E")), (h.id, rid)
        # door numbers on each side run 1..n in the order listed
        for side in {s for _, s in h.order}:
            on_side = [rid for rid, s in h.order if s == side]
            doors = [next(slot.door for slot in lay.rooms[rid].halls if slot.hall_id == h.id
                          and slot.side == f"plan {side} side") for rid in on_side]
            assert doors == list(range(1, len(on_side) + 1)), (h.id, side)
    south_west = lay.halls["f3-corridor-s-west"]
    assert [rid for rid, side in south_west.order if side == "S"] == ["f3-office-301", "f3-office-302"]


def test_size_rank_position_and_distances(idx, building):
    lay = idx.layout
    for f in lay.floors:
        rooms = [idx.rooms[rid] for rid in f.rooms if idx.rooms[rid].room_type not in CIRCULATION]
        largest = min(rooms, key=lambda r: (-r.area_m2, r.id))
        assert lay.rooms[largest.id].size_rank == 1 and "largest on floor" in lay.rooms[largest.id].size_word
        ranks = sorted(lay.rooms[r.id].size_rank for r in rooms)
        assert ranks == list(range(1, len(rooms) + 1))
        for rid in f.rooms:
            if idx.rooms[rid].room_type in CIRCULATION:
                assert lay.rooms[rid].size_rank == 0
    assert lay.rooms["f3-suite-vacant"].size_rank == 1
    assert lay.rooms["f3-office-301"].position == "plan SW corner"
    assert lay.rooms["f3-office-303"].position == "plan SE corner"
    assert "corner" not in lay.rooms["f3-office-302"].position
    assert lay.rooms["f3-corridor-s-mid"].position.startswith("interior")
    assert lay.rooms["f3-stair-west"].stairs_m == 0 and lay.rooms["f3-elevator"].elevator_m == 0
    assert lay.rooms["f3-office-302"].stairs_m < lay.rooms["f3-office-302"].elevator_m


def test_find_rooms_filters(idx):
    corners = ids_of(find_rooms(idx, {"floor_labels": ["3"], "position": "corner"}))
    assert set(corners) == {"f3-suite-vacant", "f3-conference-n", "f3-office-301", "f3-office-303"}
    big = find_rooms(idx, {"floor_labels": ["3"], "min_area_m2": 90})
    assert set(ids_of(big)) == {"f3-suite-vacant", "f3-open-office-n1", "f3-open-office-s1", "f3-open-office-s2"}
    assert all(idx.rooms[r].area_m2 >= 90 for r in ids_of(big))
    on_hall = ids_of(find_rooms(idx, {"on_hall": "f3-corridor-s-west"}))
    assert {"f3-office-301", "f3-office-302", "f3-stair-west"} <= set(on_hall)
    assert all(idx.rooms[r].floor_label == "3" for r in on_hall)
    assert ids_of(find_rooms(idx, {"text": "302"}))[0] == "f3-office-302"
    both = ids_of(find_rooms(idx, {"floor_labels": ["3"], "position": "corner", "room_types": ["office"],
                                   "max_walk_to_stairs_m": 13}))
    assert set(both) <= {"f3-suite-vacant", "f3-office-301", "f3-office-303"} and both
    ends = find_rooms(idx, {"floor_labels": ["3"], "position": "hall_end", "limit": 3})
    assert ends["total"] > 3 and len(ends["rooms"]) == 3
    assert find_rooms(idx, {"floor_labels": ["3"], "limit": "lots"})["total"] == len(
        [r for r in idx.building.rooms if r.floor_label == "3"])  # a bad limit falls back to the default
    assert find_rooms(idx, {"floor_labels": ["99"]}) == {"total": 0, "rooms": []}


def test_room_details_and_run_lookup(idx):
    out = room_details(idx, ["f3-office-302", "nope"])
    assert out["rooms"][0].startswith("f3-office-302 | Office 302") and out["unknown_ids"] == ["nope"]
    assert run_lookup(idx, "room_details", {"room_ids": "f3-office-302"})["rooms"][0].startswith("f3-office-302")
    assert run_lookup(idx, "find_rooms", {"position": "corner", "floor_labels": ["3"]})["total"] == 4
    assert "error" in run_lookup(idx, "rooms_near", {"room_id": "f3-office-302", "radius_m": "far"})
    assert "error" in run_lookup(idx, "teleport", {})


def test_layout_roster_lists_every_room_and_hallway(idx, building):
    roster = layout_roster(idx)
    lines = roster.splitlines()
    starts = {line.split(" | ")[0] for line in lines if " | " in line}
    assert {r.id for r in building.rooms} <= starts
    halls = [r.id for r in building.rooms if r.room_type == "corridor"]
    assert halls and all(any(line.startswith(f"- {h} ") for line in lines) for h in halls)
    assert "=== Floor 3:" in roster and "plan SW corner" in roster
    system = agent.build_system_prompt(idx)
    assert roster in system and "{{" not in system

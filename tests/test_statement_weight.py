"""Specific caller statements weigh heavily (configs/statements.yaml, core/fusion.py).

A specific statement of where the caller is (a room, unit or suite number, a room or tenant
name, a floor) gives the place it names 40 % to 95 % of the probability on its own, whatever
the prior says. Covered here, all offline (the model is a stub client):

1. the stated share on its own, in rules mode and in model mode (parser/agent.py), and the rating behind it
2. agreement reinforces, a correction replaces, a conflict without a correction: the later
   statement wins and the console shows a notice
3. caller against phone: the caller's statement leads, a notice says so, the phone keeps a
   share, and with two teams the second team starts on the phone's floor
4. still true: nothing reaches zero, the dispatcher's reject undoes a statement, the weights
   are named settings labelled judgement, descriptions keep the soft factor
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import parser.agent as agent
import parser.extract as extract
from api.session import CallSession, SituationPatch
from core.evidence import basement_floors, evidence_match
from core.fusion import (
    EPS,
    REINFORCED_MAX,
    RHO_MAX,
    STATEMENTS_PATH,
    Statement,
    clue_factor,
    fuse,
    fuse_detail,
    names_rooms,
    phone_height_likelihood,
    posterior_probs,
    stated_share,
    statement_settings,
)
from core.phone_likelihood import phone_likelihood
from core.prior import compute_prior
from core.schema import Clue, Evidence, PhoneFix, RoomSelector, TranscriptTurn
from core.uncertainty import floor_marginals
from ingest.matterport import generate_synthetic_office, load_building
from parser.interpret import interpret_turn
from parser.statement_rating import Rating, clean_rating, rate_span
from planning.search_order import plan_search

ROOM_302 = "f3-office-302"
START = [{"title": "connected", "speaker": "system", "text": "Call connected."}]


@pytest.fixture(scope="module")
def building():
    return generate_synthetic_office()  # ptc-100: four floors, 100 rooms


@pytest.fixture(scope="module")
def tall():
    return load_building("bundy-olympic")  # eight floors


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep the disk cache, usage file and request counter away from the real ones."""
    monkeypatch.setattr(agent, "AGENT_CACHE_PATH", tmp_path / "agent_cache.json")
    monkeypatch.setattr(extract, "LLM_USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(extract, "LLM_CACHE_PATH", tmp_path / "clue_cache.json")
    monkeypatch.setattr(extract, "_llm_calls_made", 0)
    monkeypatch.delenv("LLM_MAX_CALLS", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def turns(*pairs):
    return [TranscriptTurn(turn=i, title=f"t{i}", speaker=s, text=t) for i, (s, t) in enumerate(pairs)]


def session(building, *said, fix_z=None, n_teams=1) -> CallSession:
    """A rules-mode call: optional phone height fix first, then the caller's turns."""
    script = list(START)
    if fix_z is not None:
        script.append({"title": "fix", "speaker": "system",
                       "text": f"Device barometric z-axis fix received: {fix_z} m above ground (±3 m).",
                       "phone_fix": PhoneFix(z_m=fix_z)})
    s = CallSession(building, parser_mode="rules", script=script)
    if fix_z is not None:
        s.step()
    for text in said:
        s.say(text)
    if n_teams != 1:
        s.patch_situation(SituationPatch(n_teams=n_teams))
    return s


def floor_share(state, label: str) -> float:
    return state.posterior.p_floor[label]


def notices(state, about: str):
    return [b for b in state.policy.banners if about in b.source]


def floor_ev(value: str, share: float, turn: int = 1, k: int = 0) -> Evidence:
    return Evidence(id=f"t{turn}-e{k}", kind="vocab", confidence=share, turn=turn, span=f"floor {value}",
                    clue=Clue(type="floor", value=value, polarity=True, confidence=share, span=f"floor {value}"))


def room_ev(unit: str, share: float, turn: int = 1, k: int = 0) -> Evidence:
    return Evidence(id=f"t{turn}-e{k}", kind="grounded", selector=RoomSelector(unit=unit), confidence=share,
                    turn=turn, span=f"room {unit}")


def floors(building, probs) -> dict[str, float]:
    return floor_marginals(building, probs)


def tool_use(name: str, **args):
    tool_use.n += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_sw_{tool_use.n}", name=name, input=args)


tool_use.n = 0


def reply(*blocks):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    usage = SimpleNamespace(input_tokens=50, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


class StubClient:
    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        if self.script:
            return self.script.pop(0)
        return reply(SimpleNamespace(type="text", text="nothing to record"))


def haiku(building, call, *blocks, prior_evidence=()):
    """Read the last turn of ``call`` in model mode ("llm") with a scripted reply."""
    client = StubClient([reply(*blocks)])
    out = agent.interpret(building, call, len(call) - 1, client=client, source="llm", prior_evidence=prior_evidence)
    return out, client


# ---------------------------------------------------------------------------
# 0. Settings: named, in configs/, labelled judgement
# ---------------------------------------------------------------------------


def test_the_weights_are_named_settings_labelled_judgement():
    data = yaml.safe_load(Path(STATEMENTS_PATH).read_text(encoding="utf-8"))
    assert Path(STATEMENTS_PATH).parent.name == "configs"
    low, high = data["stated_share"]["low"], data["stated_share"]["high"]
    assert (low["value"], high["value"]) == (0.40, 0.95)
    for entry in (low, high, data["stated_share"]["missing_rating"], data["reinforced_share_max"],
                  data["phone_share_when_caller_disagrees"]):
        assert entry["label"] == "judgement" and entry["source"]
    cfg = statement_settings()
    assert (cfg.share_low, cfg.share_high) == (0.40, 0.95) and RHO_MAX == cfg.share_high
    assert cfg.phone_share == data["phone_share_when_caller_disagrees"]["value"] == 0.25
    assert cfg.reinforced_max == REINFORCED_MAX < 1.0
    assert set(cfg.labels.values()) == {"judgement"}


def test_stated_share_maps_the_two_ratings_onto_40_to_95_percent():
    assert stated_share(1.0, 1.0) == 0.95  # plain and exact
    assert stated_share(0.0, 1.0) == stated_share(1.0, 0.0) == 0.40  # either weakness pulls it to the low end
    assert stated_share(0.5, 1.0) == pytest.approx(0.675)
    assert 0.40 < stated_share(0.5, 0.5) < stated_share(0.5, 1.0) < stated_share(0.8, 1.0) < 0.95
    # clamped, and a missing or unusable rating counts as 0.5, never as certain
    assert stated_share(7, 1.0) == 0.95 and stated_share(-3, 1.0) == 0.40
    assert stated_share(None, 1.0) == stated_share("loud", 1.0) == stated_share(float("nan"), 1.0) == pytest.approx(0.675)
    assert stated_share(None, None) == pytest.approx(0.40 + 0.55 * 0.25, abs=1e-3)


# ---------------------------------------------------------------------------
# 1. The stated share on its own, whatever the prior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.40, 0.675, 0.95])
def test_a_stated_floor_gets_exactly_its_share_whatever_the_building_height(building, tall, share):
    for b, label in ((building, "3"), (tall, "6")):
        probs, nim = posterior_probs(b, clues=[floor_ev(label, share)])
        assert floors(b, probs)[label] == pytest.approx(share, abs=1e-9)
        assert sum(probs.values()) + nim == pytest.approx(1.0)
        assert min(probs.values()) > 0.0 and nim > 0.0


def test_the_stated_share_does_not_depend_on_the_prior(building):
    low_prior = "f3-suite-vacant"  # "Suite 310 (vacant)"
    for hour in (3, 14, 20):  # night, business hours, evening: three different priors
        prior, _ = compute_prior(building, hour=hour)
        assert prior[low_prior] < prior[ROOM_302]
        for share in (0.40, 0.95):
            for room_id, unit in ((ROOM_302, "302"), (low_prior, "310")):
                probs, _ = posterior_probs(building, hour=hour, clues=[room_ev(unit, share)])
                assert probs[room_id] == pytest.approx(share, abs=1e-9)
            probs, _ = posterior_probs(building, hour=hour, clues=[floor_ev("3", share)])
            assert floors(building, probs)["3"] == pytest.approx(share, abs=1e-9)


def test_the_rest_goes_by_the_prior_and_the_floors_next_door_keep_more(building):
    probs, nim = posterior_probs(building, clues=[floor_ev("1", 0.95)])
    prior, p0 = compute_prior(building)
    by_floor, prior_floor = floors(building, probs), floors(building, prior)
    # floor 2 is next to the stated floor: callers miscount, so it keeps more than its prior share of the rest
    assert by_floor["2"] / prior_floor["2"] > 2 * by_floor["4"] / prior_floor["4"]
    assert by_floor["3"] / prior_floor["3"] == pytest.approx(by_floor["4"] / prior_floor["4"])
    # a room statement leaves the other rooms in their prior proportions
    probs, _ = posterior_probs(building, clues=[room_ev("302", 0.95)])
    a, b = "f1-lobby", "f4-boardroom"
    assert probs[a] / probs[b] == pytest.approx(prior[a] / prior[b])


def test_a_statement_never_lowers_the_place_it_names(building):
    prior, p0 = compute_prior(building)
    total = sum(prior.values()) + p0
    upper = {r.id for r in building.rooms if r.floor >= 2}
    before = sum(prior[r] for r in upper) / total
    assert before > 0.40
    match = {r.id: (1.0 if r.id in upper else 0.0) for r in building.rooms}
    probs, _ = fuse(prior, p0, None, [Statement(match, 0.40, True)])
    assert sum(probs[r] for r in upper) == pytest.approx(before)


@pytest.mark.parametrize("text, label", [
    ("I'm on the third floor", "3"),
    ("we're on floor 2", "2"),
    ("I'm on the top floor", "4"),
    ("I'm on the ground floor", "1"),
])
def test_rules_mode_a_floor_said_plainly_gets_95_percent(building, text, label):
    state = session(building, text).state()
    assert floor_share(state, label) == pytest.approx(0.95, abs=1e-6)
    [ev] = [e for e in state.evidence if e.clue is not None and e.clue.type == "floor"]
    assert ev.confidence == 0.95 and names_rooms(ev) and "rated by rules" in ev.rationale


def test_rules_mode_the_three_reference_statements(building, tall):
    """Before this change: 48 % on floor 3 (ptc-100), 42 % on floor 6 (bundy-olympic), 81 % on Office 302."""
    assert floor_share(session(building, "I'm on the third floor").state(), "3") == pytest.approx(0.95, abs=1e-6)
    assert floor_share(session(tall, "I'm on the sixth floor").state(), "6") == pytest.approx(0.95, abs=1e-6)
    assert session(building, "I'm in room 302").state().posterior.room_probs[ROOM_302] == pytest.approx(0.95, abs=1e-6)


def test_basement_is_a_floor_statement_where_the_building_has_one(building):
    assert basement_floors(building) == []
    state = session(building, "I'm in the basement").state()
    [ev] = state.evidence
    assert ev.clue.value == "basement" and ev.unmatched  # shown as not placed
    assert state.posterior.room_probs == session(building).state().posterior.room_probs
    below = building.model_copy(deep=True)
    for r in below.rooms:
        if r.floor == 1:
            r.floor_label = "B1"
    assert basement_floors(below) == [1]
    state = session(below, "I'm in the basement").state()
    assert floor_share(state, "B1") == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("text, span, certainty, specificity", [
    ("I'm in room 302", "room 302", 1.0, 1.0),
    ("I think I'm in 302", "in 302", 0.5, 1.0),
    ("302, I think.", "302", 0.5, 1.0),
    ("I'm in 302, I think the fire is in 301", "in 302", 1.0, 1.0),  # the hedge is about the fire
    ("third floor?", "third floor", 0.5, 1.0),  # questioning tone
    ("maybe 302", "302", 0.25, 1.0),
    ("I think maybe the third floor?", "third floor", 0.0625, 1.0),
    ("somewhere on the third floor", "third floor", 1.0, 0.5),
    ("the third or fourth floor", "third", 1.0, 0.5),
    ("room 302, please hurry, I don't know what to do", "room 302", 1.0, 1.0),  # fear is not doubt
])
def test_the_rule_rating_hears_hedges_questions_and_vagueness(text, span, certainty, specificity):
    assert rate_span(text, span) == Rating(certainty, specificity)


def test_rules_mode_a_hesitant_statement_sits_near_the_low_end(building):
    state = session(building, "I think maybe the third floor?").state()
    assert 0.40 <= floor_share(state, "3") < 0.45
    hedged = session(building, "I think I'm in 302").state()
    assert hedged.posterior.room_probs[ROOM_302] == pytest.approx(0.675, abs=1e-6)
    plain = session(building, "I'm in 302").state()
    assert plain.posterior.room_probs[ROOM_302] == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("text", [
    "he went up to the third floor",
    "the shooter is on the third floor",
    "I think the fire is on the third floor",
    "my coworker ran down to the second floor to get help",
])
def test_a_floor_that_is_about_someone_else_is_not_the_callers_location(building, text):
    it = interpret_turn(building, turns(("caller", text)), 0, "rules")
    assert not [e for e in it.evidence if e.clue is not None and e.clue.type == "floor"]


def test_the_callers_own_move_is_still_their_floor(building):
    it = interpret_turn(building, turns(("caller", "we ran up to the third floor and locked the door")), 0, "rules")
    assert [(e.clue.value, e.confidence) for e in it.evidence if e.clue is not None and e.clue.type == "floor"] == [("3", 0.95)]


@pytest.mark.parametrize("text", ["I'm upstairs somewhere", "please hurry, he's not breathing", "hello? hello?"])
def test_words_that_name_no_place_produce_no_location_evidence(building, text):
    assert session(building, text).state().evidence == []


def test_descriptions_keep_the_soft_factor(building):
    """A room type, a view, a denial: at most 10 to 1 between two rooms, as before."""
    prior, _ = compute_prior(building)
    for clue in (Clue(type="room_type", value="restroom", polarity=True, confidence=1.0, span="bathroom"),
                 Clue(type="view", value="parking_lot", polarity=True, confidence=0.9, span="parking lot"),
                 Clue(type="negative", value="floors:3", polarity=False, confidence=0.7, span="not on three")):
        ev = Evidence(id="d", kind="vocab", clue=clue, confidence=clue.confidence)
        assert not names_rooms(ev)
        match = evidence_match(building, ev).match
        probs, _ = posterior_probs(building, clues=[ev])
        hit = max(match, key=match.get)
        miss = min(match, key=match.get)
        ratio = (probs[hit] / prior[hit]) / (probs[miss] / prior[miss])
        assert ratio == pytest.approx(clue_factor(match[hit], clue.confidence, EPS) / clue_factor(match[miss], clue.confidence, EPS))
        assert ratio <= 10.0 + 1e-9


# ---- Model mode (stub client) -------------------------------------------


def test_the_model_rates_a_hedged_named_room(building, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    call = turns(("dispatcher", "Where are you?"), ("caller", "um I think we're in 302? I'm not sure"))
    out, client = haiku(building, call,
                        tool_use("record_location", selector={"room_ids": [ROOM_302]}, negate=False, confidence=0.9,
                                 certainty=0.2, specificity=1.0, span="we're in 302", rationale="Office 302"))
    [ev] = out.evidence  # no rule reading first: the one statement is the model's
    assert ev.source == "llm" and ev.matched_room_ids == [ROOM_302] and names_rooms(ev)
    assert ev.confidence == pytest.approx(stated_share(0.2, 1.0)) == pytest.approx(0.51)
    assert "rated by Claude Sonnet 5.5" in ev.rationale
    assert len(client.calls) == 1
    message = client.calls[0]["messages"][0]["content"]
    assert "rate_statement" not in message and "Already recorded for turn" not in message
    names = {t["name"] for t in client.calls[0]["tools"]}
    assert "rate_statement" not in names and "record_vocab_clue" not in names
    assert {"record_clue", "record_floor", "find_rooms"} <= names
    probs, _ = posterior_probs(building, clues=out.evidence)
    assert probs[ROOM_302] == pytest.approx(0.51, abs=1e-6)
    assert "stated share 0.51" in out.trace[0].reason


def test_model_mode_runs_no_word_rules_and_rates_like_the_rules(building):
    call = turns(("caller", "I'm in room 302"))
    by_rules = interpret_turn(building, call, 0, "rules")
    silent, _ = haiku(building, call)  # the model recorded nothing: nothing is on the console
    rated, _ = haiku(building, turns(("caller", "I'm in room 302 ")),  # (a different message, so not the cached reply)
                     tool_use("record_location", selector={"room_ids": [ROOM_302]}, negate=False, confidence=0.9,
                              certainty=1.0, specificity=1.0, span="room 302", rationale="Office 302"))
    assert [e.confidence for e in by_rules.evidence if e.kind == "grounded"] == [0.95]
    assert silent.evidence == [] and silent.source == "llm"
    assert [e.confidence for e in rated.evidence] == [0.95]
    assert "rated by rules" in by_rules.evidence[0].rationale and "rated by Claude" in rated.evidence[0].rationale


def test_haiku_rating_sets_the_share_of_a_floor_and_of_a_named_room(building):
    call = turns(("dispatcher", "Where are you?"),
                 ("caller", "I don't know the building, there's a little plate by the door, it has 302 on it, third floor I think"))
    out, _ = haiku(
        building, call,
        tool_use("record_location", selector={"room_ids": [ROOM_302]}, negate=False, confidence=0.9, certainty=1.0,
                 specificity=1.0, span="it has 302 on it", rationale="Office 302"),
        tool_use("record_floor", floor="3", stated_by_caller=True, polarity=True, confidence=0.7, certainty=0.5,
                 specificity=1.0, span="third floor I think"),
    )
    room, floor = out.evidence
    assert room.confidence == 0.95 and room.source == "llm" and names_rooms(room)
    assert floor.confidence == pytest.approx(0.675) and floor.clue.confidence == pytest.approx(0.675)
    assert "stated share 0.68" in floor.rationale and floor.clue.type == "floor" and names_rooms(floor)


def test_a_floor_the_model_worked_out_is_capped_and_a_denial_stays_descriptive(building):
    call = turns(("dispatcher", "Where are you?"), ("caller", "it has 302 on it, and we're not on four"))
    out, _ = haiku(
        building, call,
        tool_use("record_floor", floor="3", stated_by_caller=False, polarity=True, confidence=0.95, certainty=1.0,
                 specificity=1.0, span="it has 302 on it"),
        tool_use("record_floor", floor="4", stated_by_caller=True, polarity=False, confidence=0.99, certainty=1.0,
                 specificity=1.0, span="we're not on four"),
    )
    worked, denial = out.evidence
    assert worked.clue.type == "floor" and worked.confidence == agent.DESCRIBED_CONFIDENCE_CAP == 0.7
    assert "not stated by the caller" in worked.rationale
    assert denial.clue.type == "negative" and denial.clue.value == "floors:4" and denial.clue.polarity is False
    assert denial.confidence == agent.GROUNDED_CONFIDENCE_CAP and not names_rooms(denial)


def test_a_missing_or_unusable_rating_defaults_conservatively(building):
    plate = "there's a little plate by the door, it has 302 on it"
    out, _ = haiku(building, turns(("caller", plate)),
                   tool_use("record_location", selector={"room_ids": [ROOM_302]}, negate=False, confidence=0.99,
                            span="it has 302 on it", rationale="Office 302"))
    [ev] = out.evidence
    assert ev.confidence == stated_share(0.5, 0.5) == pytest.approx(0.5375, abs=1e-3)  # not 0.95
    assert "certainty missing" in out.trace[0].reason and "specificity missing" in out.trace[0].reason
    # out of range is clamped; nonsense counts as missing; a hedge the rules heard lowers the default further
    assert clean_rating(3, -1)[0] == Rating(1.0, 0.0)
    assert clean_rating("very", None)[0] == Rating(0.5, 0.5)
    assert clean_rating(None, None, Rating(0.25, 1.0))[0] == Rating(0.25, 0.5)
    out, _ = haiku(building, turns(("caller", plate + " ")),
                   tool_use("record_location", selector={"room_ids": [ROOM_302]}, negate=False, confidence=0.9,
                            certainty=40, specificity="exact", span="it has 302 on it", rationale="Office 302"))
    assert out.evidence[0].confidence == pytest.approx(stated_share(1.0, 0.5))
    # rate_statement is no longer offered; one in an older cached run has nothing to rate and is dropped
    out, _ = haiku(building, turns(("caller", "I'm in room 302")),
                   tool_use("rate_statement", statement="s1", certainty=0.0, specificity=0.0))
    assert out.evidence == []
    assert [t.kept for t in out.trace if t.tool == "rate_statement"] == [False]


def test_a_location_worked_out_from_a_description_keeps_its_cap(building):
    call = turns(("caller", "I'm in the office right next to the copy room on three"))
    out, _ = haiku(building, call,
                   tool_use("record_location", selector={"room_ids": ["f3-office-301"]}, negate=False, confidence=0.9,
                            certainty=1.0, specificity=1.0, span="office right next to the copy room on three",
                            rationale="office beside the copy room"))
    described = [e for e in out.evidence if e.source == "llm"]
    assert [e.confidence for e in described] == [agent.DESCRIBED_CONFIDENCE_CAP]


def test_haiku_mode_keeps_a_repeated_specific_statement(building):
    """Said again in a later turn, a room number or a floor is kept: agreement reinforces."""
    first = floor_ev("3", 0.95, turn=1)
    call = turns(("dispatcher", "Where are you?"), ("caller", "third floor"), ("dispatcher", "Say again?"),
                 ("caller", "the third floor, please hurry"))
    windowed = [r.id for r in building.rooms if r.floor_label == "3" and r.has_windows]
    out, _ = haiku(building, call,
                   tool_use("record_floor", floor="3", stated_by_caller=True, polarity=True, confidence=0.9,
                            certainty=1.0, specificity=1.0, span="the third floor"),
                   tool_use("record_clue", clue="room with a window", room_ids=windowed, others_fit=0.3, negate=False,
                            confidence=0.8, span="please hurry", reasoning="test clue"),
                   prior_evidence=[first])
    assert [(e.kind, e.confidence) for e in out.evidence] == [("vocab", 0.95), ("grounded", 0.8)]
    assert out.evidence[0].clue.type == "floor"
    probs, _ = posterior_probs(building, clues=[first, out.evidence[0]])
    assert 0.95 < floors(building, probs)["3"] < 1.0


# ---------------------------------------------------------------------------
# 2. Agreement, correction, conflict
# ---------------------------------------------------------------------------


def test_the_same_place_said_twice_ends_above_either_statement_and_below_one(building):
    once = session(building, "I'm in room 302").state().posterior.room_probs[ROOM_302]
    twice = session(building, "I'm in room 302", "yes, room 302, hurry").state().posterior.room_probs[ROOM_302]
    assert once == pytest.approx(0.95, abs=1e-6)
    assert once < twice < 1.0 and twice == pytest.approx(1 - 0.05 * 0.05, abs=1e-3)
    # two hesitant statements stay modest, however small the prior of the room
    probs, _ = posterior_probs(building, clues=[room_ev("302", 0.40, turn=1), room_ev("302", 0.40, turn=3)])
    assert probs[ROOM_302] == pytest.approx(1 - 0.6 * 0.6, abs=0.01)
    probs, _ = posterior_probs(building, clues=[floor_ev("3", 0.40, turn=1), floor_ev("3", 0.40, turn=3)])
    assert 0.40 < floors(building, probs)["3"] < 0.64


def test_a_room_on_the_stated_floor_and_the_floor_reinforce_each_other(building):
    state = session(building, "I'm on the third floor", "room 302").state()
    assert state.posterior.room_probs[ROOM_302] > 0.95 and floor_share(state, "3") > 0.99
    assert max(state.posterior.room_probs.values()) < 1.0 and floor_share(state, "3") < 1.0
    assert notices(state, "caller statement") == []
    # the order they were said in does not matter
    room, floor = room_ev("302", 0.95, turn=1), floor_ev("3", 0.95, turn=3)
    a, _ = posterior_probs(building, clues=[room, floor])
    b, _ = posterior_probs(building, clues=[floor.model_copy(update={"turn": 1, "id": "t1-e0"}),
                                             room.model_copy(update={"turn": 3, "id": "t3-e0"})])
    assert a[ROOM_302] == pytest.approx(b[ROOM_302]) and a == pytest.approx(b)
    # and many agreeing statements never pass the cap
    many = [room_ev("302", 0.95, turn=t) for t in range(1, 8)] + [floor_ev("3", 0.95, turn=9)]
    probs, nim = posterior_probs(building, clues=many)
    assert probs[ROOM_302] <= REINFORCED_MAX + 1e-12 and min(probs.values()) > 0.0 and nim > 0.0


def test_a_correction_replaces_what_it_corrects(building):
    corrected = session(building, "I'm on the third floor", "sorry, I meant the fourth floor").state()
    only_fourth = session(building, "I'm on the fourth floor").state()
    assert [e.status for e in corrected.evidence] == ["superseded", "active"]
    assert floor_share(corrected, "4") == pytest.approx(0.95, abs=1e-6)
    assert corrected.posterior.room_probs == pytest.approx(only_fourth.posterior.room_probs)
    assert notices(corrected, "caller statement") == []  # a correction is not a conflict


def test_a_conflict_without_a_correction_the_later_statement_wins_and_a_notice_shows(building):
    s = session(building, "I'm on the third floor", "we're on the fourth floor")
    state = s.state()
    first, second = state.evidence
    assert first.status == second.status == "active"  # nobody corrected anything
    assert floor_share(state, "4") == pytest.approx(0.95, abs=1e-6) and floor_share(state, "3") < 0.05
    [notice] = notices(state, "caller statements")
    assert notice.level == "warning"
    assert first.id in notice.source and second.id in notice.source and "no correction" in notice.source
    assert '"third floor"' in notice.text and '"fourth floor"' in notice.text
    assert "floor 3" in notice.text and "floor 4" in notice.text and "later statement is used" in notice.text
    # the dispatcher's reject of the later statement brings the earlier one back, and the notice goes
    s.set_evidence_status(second.id, "rejected")
    state = s.state()
    assert floor_share(state, "3") == pytest.approx(0.95, abs=1e-6)
    assert notices(state, "caller statements") == []


def test_conflict_between_a_room_and_a_floor_and_a_later_agreeing_statement(building):
    # a room on floor 3, then "fourth floor": the later one wins
    state = session(building, "I'm in room 302", "we're on the fourth floor").state()
    assert floor_share(state, "4") == pytest.approx(0.95, abs=1e-6)
    assert len(notices(state, "caller statements")) == 1
    # then the room again: it is the latest, the floor-4 statement is set aside, and the first
    # statement (which agrees with the latest) counts again
    state = session(building, "I'm in room 302", "we're on the fourth floor", "no it's room 302").state()
    assert state.posterior.room_probs[ROOM_302] > 0.95
    [notice] = notices(state, "caller statements")
    assert "floor 4" in notice.text


def test_two_places_in_one_turn_the_more_certain_then_the_more_specific_is_used(building):
    room, floor = room_ev("302", 0.95, turn=1, k=0), floor_ev("4", 0.95, turn=1, k=1)
    out = fuse_detail(*compute_prior(building), None, [
        Statement(evidence_match(building, e).match, e.confidence, True, e.turn, e.id) for e in (room, floor)])
    assert out.set_aside == [(1, 0)] and out.room_probs[ROOM_302] == pytest.approx(0.95)
    hedged_room = room.model_copy(update={"confidence": 0.45})
    out = fuse_detail(*compute_prior(building), None, [
        Statement(evidence_match(building, e).match, e.confidence, True, e.turn, e.id) for e in (hedged_room, floor)])
    assert out.set_aside == [(0, 1)]


# ---------------------------------------------------------------------------
# 3. Caller against phone
# ---------------------------------------------------------------------------

PHONE_ON_ONE = 1.4  # metres: floor 1 of ptc-100 (floors are 4 m apart; the fix is quoted +/- 3 m)


def first_stop_floors(state) -> list[str]:
    return [team[0].floor_label for team in state.plan.teams]


def test_the_callers_floor_leads_over_the_phone_and_a_notice_says_so(building):
    state = session(building, "I'm on the third floor", fix_z=PHONE_ON_ONE).state()
    by_floor = state.posterior.p_floor
    assert max(by_floor, key=by_floor.get) == "3"
    assert by_floor["3"] == pytest.approx(0.75 * 0.95, abs=0.01)  # the caller's reading keeps 75 %
    assert by_floor["1"] > 0.15  # most of the phone's 25 % sits on its floor
    [notice] = notices(state, "phone's height fix")
    assert notice.level == "warning" and state.evidence[0].id in notice.source
    assert '"third floor"' in notice.text and "floor 3" in notice.text and "floor 1 (1.4 m)" in notice.text
    assert "caller's statement leads" in notice.text and "about 25%" in notice.text
    # without the statement the phone has floor 3 near zero; the statement alone has it at 95 %
    phone_only = session(building, fix_z=PHONE_ON_ONE).state().posterior.p_floor
    assert phone_only["3"] < 0.01 and max(phone_only, key=phone_only.get) == "1"


def test_the_phone_keeps_the_share_the_setting_names(building):
    prior, p0 = compute_prior(building)
    fix = PhoneFix(z_m=PHONE_ON_ONE)
    phone, height = phone_likelihood(building, fix), phone_height_likelihood(building, fix)
    ev = floor_ev("3", 0.95)
    term = Statement(evidence_match(building, ev).match, ev.confidence, True, 1, ev.id)
    out = fuse_detail(prior, p0, phone, [term], phone_height=height)
    assert out.phone_share == statement_settings().phone_share == 0.25 and out.phone_conflicts == [0]
    assert {r.floor for r in building.rooms if r.id in out.phone_band} == {1}
    caller_only, nim_c = fuse(prior, p0, None, [term])
    phone_only, nim_p = fuse(prior, p0, phone, [])
    for rid in prior:
        assert out.room_probs[rid] == pytest.approx(0.75 * caller_only[rid] + 0.25 * phone_only[rid])
    assert out.p_not_in_model == pytest.approx(0.75 * nim_c + 0.25 * nim_p)
    assert min(out.room_probs.values()) > 0.0
    # without the height part given, the old plain product: the phone's fix outweighs the caller
    plain, _ = fuse(prior, p0, phone, [term])
    by_floor = floors(building, plain)
    assert max(by_floor, key=by_floor.get) != "3" and by_floor["3"] < 0.3


def test_a_weak_statement_lowers_the_phones_share_so_the_stated_floor_still_leads(building):
    prior, p0 = compute_prior(building)
    fix = PhoneFix(z_m=5.2)  # floor 2, next to the floor the caller names
    ev = floor_ev("3", 0.40)
    term = Statement(evidence_match(building, ev).match, ev.confidence, True, 1, ev.id)
    out = fuse_detail(prior, p0, phone_likelihood(building, fix), [term],
                      phone_height=phone_height_likelihood(building, fix))
    by_floor = floors(building, out.room_probs)
    assert out.phone_conflicts == [0] and 0.0 < out.phone_share < 0.25
    assert max(by_floor, key=by_floor.get) == "3"
    # at the full 25 % the phone's floor would have led
    full = {rid: 0.75 * fuse(prior, p0, None, [term])[0][rid]
            + 0.25 * fuse(prior, p0, phone_likelihood(building, fix), [])[0][rid] for rid in prior}
    assert max(floors(building, full), key=floors(building, full).get) == "2"


def test_a_phone_that_agrees_with_the_caller_changes_nothing_about_the_rule(building):
    state = session(building, "I'm on the third floor", fix_z=9.4).state()  # 9.4 m: floor 3
    assert floor_share(state, "3") > 0.95
    assert notices(state, "phone's height fix") == []
    assert all("phone's height fix" not in n for n in state.plan.notes)


@pytest.mark.parametrize("text, first_room", [("I'm on the third floor", None), ("I'm in room 302", ROOM_302)])
def test_two_teams_team_one_follows_the_caller_team_two_starts_on_the_phones_floor(building, text, first_room):
    state = session(building, text, fix_z=PHONE_ON_ONE, n_teams=2).state()
    assert first_stop_floors(state) == ["3", "1"]
    if first_room is not None:
        assert state.plan.teams[0][0].room_id == first_room
    one = session(building, text, fix_z=PHONE_ON_ONE, n_teams=1).state()
    assert first_stop_floors(one) == ["3"]  # with one team the plan goes to the stated place first
    if first_room is not None:
        assert one.plan.teams[0][0].room_id == first_room


def test_two_teams_on_a_tall_building_and_under_the_survival_objective(tall, building):
    # bundy-olympic: 5.0 m is floor 2; the caller says the sixth floor
    state = session(tall, "I'm on the sixth floor", fix_z=5.0, n_teams=2).state()
    assert first_stop_floors(state) == ["6", "2"]
    assert any("Team 2 starts" in n and "phone's height fix" in n for n in state.plan.notes)
    # with a confirmed cardiac arrest the plan maximises survival: the rule still holds
    s = session(building, "I'm on the third floor", fix_z=PHONE_ON_ONE, n_teams=2)
    s.patch_situation(SituationPatch(incident_type="cardiac_arrest"))
    state = s.state()
    assert state.plan.objective == "survival" and first_stop_floors(state) == ["3", "1"]


def test_rejecting_the_statement_gives_the_phone_its_say_back(building):
    s = session(building, "I'm on the third floor", fix_z=PHONE_ON_ONE, n_teams=2)
    [ev] = s.state().evidence
    s.set_evidence_status(ev.id, "rejected")
    state = s.state()
    phone_only = session(building, fix_z=PHONE_ON_ONE, n_teams=2).state()
    assert state.posterior.room_probs == pytest.approx(phone_only.posterior.room_probs)
    assert notices(state, "phone's height fix") == []
    assert first_stop_floors(state) == first_stop_floors(phone_only)


def test_second_team_rooms_only_moves_the_second_teams_start(building):
    """planning.search_order.plan_search(second_team_rooms=...): the one planning change."""
    prior, p0 = compute_prior(building)
    ev = floor_ev("3", 0.95)
    fix = PhoneFix(z_m=PHONE_ON_ONE)
    probs, nim = fuse(prior, p0, phone_likelihood(building, fix),
                      [Statement(evidence_match(building, ev).match, 0.95, True)],
                      phone_height=phone_height_likelihood(building, fix))
    floor_one = {r.id for r in building.rooms if r.floor == 1}
    plain = plan_search(building, probs, nim, n_teams=2)
    assert [t[0].floor_label for t in plain.teams] == ["3", "3"]
    sent = plan_search(building, probs, nim, n_teams=2, second_team_rooms=floor_one)
    assert [t[0].floor_label for t in sent.teams] == ["3", "1"]
    assert sent.teams[0][0].room_id == plain.teams[0][0].room_id  # team 1's first stop is untouched
    stops = lambda plan: sorted(s.room_id for t in plan.teams for s in t if s.kind == "first")  # noqa: E731
    assert stops(sent) == stops(plain)  # the same rooms are searched, only the order differs
    # one team: nothing changes; a team already starting there: nothing changes
    solo = plan_search(building, probs, nim, n_teams=1)
    assert [s.room_id for s in plan_search(building, probs, nim, n_teams=1, second_team_rooms=floor_one).teams[0]] \
        == [s.room_id for s in solo.teams[0]]
    floor_three = {r.id for r in building.rooms if r.floor == 3}
    same = plan_search(building, probs, nim, n_teams=2, second_team_rooms=floor_three)
    assert all("Team 2 starts" not in n for n in same.notes)


# ---------------------------------------------------------------------------
# 4. Still true
# ---------------------------------------------------------------------------


def test_no_single_statement_can_take_a_room_to_zero(building):
    for ev in (room_ev("302", 1.0), floor_ev("3", 1.0)):
        probs, nim = posterior_probs(building, clues=[ev])
        assert min(probs.values()) > 0.0 and nim > 0.0
        named = sum(p for rid, p in probs.items() if evidence_match(building, ev).match[rid] >= 1.0)
        assert named == pytest.approx(RHO_MAX)  # a confidence of 1.0 still counts as 0.95


def test_the_dispatchers_reject_undoes_a_specific_statement(building):
    before = session(building).state().posterior
    for text in ("I'm on the third floor", "I'm in room 302"):
        s = session(building, text)
        [ev] = [e for e in s.state().evidence if names_rooms(e)]
        s.set_evidence_status(ev.id, "rejected")
        after = s.state().posterior
        assert after.room_probs == pytest.approx(before.room_probs)
        assert after.p_not_in_model == pytest.approx(before.p_not_in_model)
        s.set_evidence_status(ev.id, "active")  # and restore brings it back
        assert s.state().posterior.room_probs != pytest.approx(before.room_probs)


def test_the_phones_reading_keeps_an_earlier_statement_that_agrees_with_it(building):
    """Phone on floor 3; the caller says room 302, then "first floor". The later statement leads
    (two notices: it conflicts with the earlier statement and with the phone); in the phone's
    25 % the "first floor" statement is left out, so room 302 counts again and team 2 starts there."""
    state = session(building, "I'm in room 302", "I'm on the first floor", fix_z=9.4, n_teams=2).state()
    by_floor = state.posterior.p_floor
    assert max(by_floor, key=by_floor.get) == "1" and by_floor["1"] == pytest.approx(0.75 * 0.95, abs=0.01)
    assert state.posterior.room_probs[ROOM_302] > 0.2  # nearly all of the phone's share
    assert len(notices(state, "caller statements")) == 1 and len(notices(state, "phone's height fix")) == 1
    assert state.plan.teams[0][0].floor_label == "1" and state.plan.teams[1][0].room_id == ROOM_302

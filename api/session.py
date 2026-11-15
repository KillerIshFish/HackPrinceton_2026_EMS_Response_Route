"""One in-memory call: transcript, situation, evidence, searched rooms and the undo stack.

Moved out of ``api/server.py`` in Step 0 (``DemoSession`` -> ``CallSession``; the old name
stays as an alias there). Step I1 wired every lane in.

Pipeline per state::

    caller turn -> parser.interpret.interpret_turn            -> Evidence, Facts, suggested type
                   (+ parser.rule_facts.read_facts when the agent did not read the turn)
    Evidence    -> core.evidence.evidence_match (cached)
    prior x phone fix x matches -> core.fusion.fuse_evidence  -> evidence-only belief + notices
                   (same maths as core.fusion.posterior_probs: a specific statement gives the
                   place it names its stated share; of two that conflict the later one is used;
                   a statement that contradicts the phone's height fix leads and the phone keeps
                   a share. The notices about the last two are added to the policy banners.)
    belief + every look (room, mode) -> planning.search_order.residual_probs -> displayed posterior
    situation + facts -> situation.policy.policy_for          -> Policy (travel profile, stake, banners)
    policy.route_enabled false -> empty plan, approach with no entrance / staging, no question prompts
    otherwise   -> plan_search(belief, searched=looks, gates=officer gates, profile=policy.travel,
                               survival=survival_from(situation, elapsed_s))
                   recommend_approach(residual, profile=policy.travel, situation=, gates=)
                   top_questions / next_question on the residual posterior

Call events (searched, found, timestamp, gate, undo) are appended to the call log
(``data/processed/call_log.jsonl``; ``CALL_LOG_PATH`` or the ``call_log_path`` argument
override it), one JSON line each with seconds since call start, room id, room area, search
mode and incident type.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.evidence import EvidenceMatch, evidence_match, normalize_unit
from core.fusion import Fused, fuse_evidence
from core.prior import compute_prior
from core.schema import (
    AgentTraceEntry,
    Answer,
    Building,
    Clue,
    CrewLoad,
    Evidence,
    Fact,
    Gates,
    IncidentType,
    Interpretation,
    ParserMode,
    PhoneFix,
    Policy,
    Posterior,
    Question,
    QuestionKind,
    SearchedRoom,
    SearchMode,
    SearchPlan,
    Situation,
    Speaker,
    TimestampName,
    TranscriptTurn,
)
from core.uncertainty import summarize
from ingest.matterport import list_buildings
from ingest.real_building import load_meta
from parser.dialog import asked_keys as dialog_asked_keys
from parser.agent import agent_model
from parser.extract import SECTION_FOR_TYPE, llm_usage
from parser.interpret import interpret_turn
from parser.next_question import asked_from_clues, next_question, question_key, top_questions
from parser.rule_facts import read_facts
from planning.approach import recommend_approach
from planning.search_order import looks_by_room, miss_factor, plan_search, residual_probs, search_modes
from sim.generate_calls import DEMO_BUILDING_ID, demo_script_for, scenario_definitions, scripted_calls
from situation.policy import active_threat_pending, policy_for
from situation.survival import survival_from

DEMO_HOUR = 14  # 2 pm on a weekday: business-hours prior for scripted demo calls
MAX_TEAMS = 3
MAX_UTTERANCE_CHARS = 600
_NOT_NULLABLE = {"incident_confirmed", "smoke_reported", "load", "call_started_at"}

REPO_ROOT = Path(__file__).resolve().parents[1]
CALL_LOG_ENV = "CALL_LOG_PATH"
DEFAULT_CALL_LOG = REPO_ROOT / "data" / "processed" / "call_log.jsonl"

# Hazard-class reading of facts (fire_smoke only, display only; the stop order never changes).
_FIRE_WORDS = re.compile(r"(?<!\w)(?:fire|flames?|burning|smoke|smoky)(?!\w)", re.IGNORECASE)
_CLOSED_DOOR = re.compile(
    # also the model reader's wording of the caller's report: "they shut the office door", "their door is shut"
    r"(?<!\w)(?:door(?:'s|\s+is|\s+was)?\s+(?:closed|shut|locked)"
    r"|(?:closed|shut|locked)\s+(?:the|my|our|their|his|her)\s+(?:\w+\s+)?door)(?!\w)",
    re.IGNORECASE,
)
_SMOKE_WORDS = re.compile(r"(?<!\w)(?:smoke|smoky)(?!\w)", re.IGNORECASE)


def llm_available() -> bool:
    return (os.getenv("LLM_PROVIDER") or "").strip().lower() == "anthropic" and bool((os.getenv("LLM_API_KEY") or "").strip())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def call_log_path() -> Path:
    """Where call events go: ``$CALL_LOG_PATH`` if set, else data/processed/call_log.jsonl."""
    env = (os.getenv(CALL_LOG_ENV) or "").strip()
    return Path(env) if env else DEFAULT_CALL_LOG


# ---------------------------------------------------------------------------
# Response and request models
# ---------------------------------------------------------------------------


class HistoryPoint(BaseModel):
    turn: int
    entropy_bits: float
    expected_time_to_find_s: float
    top_room_id: str
    top_room_prob: float


class BuildingSummary(BaseModel):
    id: str
    name: str
    real: bool
    floors: int
    rooms: int


class LLMStatus(BaseModel):
    available: bool
    model: str
    usage: dict[str, Any]


class ScenarioSummary(BaseModel):
    """A scripted call the scenario picker can start (POST /calls/demo/scenario?id=)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    building_id: str
    incident_type: IncidentType = "unknown"  # what the call turns out to be (the call itself starts unset)
    description: str = ""
    # beyond section 4 (step I1)
    true_room_id: Optional[str] = None  # where the scripted caller actually is (scoring only)
    turns: int = 0


class DemoState(BaseModel):
    """Everything the console shows. Every mutating endpoint returns the whole thing."""

    model_config = ConfigDict(extra="forbid")

    building_id: str
    turn: int  # index of the latest revealed *scripted* turn
    total_turns: int
    done: bool
    transcript: list[TranscriptTurn]
    clues: list[Clue]  # active vocabulary clues (older clients)
    phone_fix: Optional[PhoneFix] = None
    cleared_room_ids: list[str]  # distinct searched rooms, first-look order (older clients)
    n_teams: int
    posterior: Posterior  # after every unsuccessful look (residual masses)
    plan: SearchPlan
    approach: dict[str, Any]
    next_question: Optional[dict[str, Any]] = None
    history: list[HistoryPoint]
    parser_mode: str = "rules"
    llm: Optional[LLMStatus] = None
    buildings: list[BuildingSummary] = []
    # ---- Step 0 contract (section 4) ----
    situation: Situation = Field(default_factory=Situation)
    policy: Policy = Field(default_factory=Policy)
    evidence: list[Evidence] = Field(default_factory=list)  # every status; only "active" counts
    facts: list[Fact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)  # top three
    searched: list[SearchedRoom] = Field(default_factory=list)  # every look, oldest first
    found_room_id: Optional[str] = None
    found_at: Optional[datetime] = None  # server time the dispatcher recorded found_room_id
    scenarios: list[ScenarioSummary] = Field(default_factory=list)
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list)
    # ---- beyond section 4 ----
    scenario_id: Optional[str] = None  # the scripted scenario running, if any
    prior_hour: int = DEMO_HOUR  # hour of day the prior was computed for
    can_undo: bool = False  # an undoable searched / found / gate action exists


class Utterance(BaseModel):
    """Body of POST /calls/demo/utterance."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_UTTERANCE_CHARS)
    speaker: Speaker = "caller"  # "caller" | "dispatcher" ("system" is refused by the endpoint)
    answers: Optional[Answer] = None  # set by the Yes / No / Not sure buttons


class AskRequest(BaseModel):
    """Body of POST /calls/demo/ask. A ``Question`` from ``state.questions`` can be posted as is."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=MAX_UTTERANCE_CHARS)
    id: Optional[str] = None
    kind: Optional[QuestionKind] = None  # default: "yes_no" when clue_type is set, else "open"
    clue_type: Optional[str] = None
    value: Optional[str] = None
    partition: Optional[str] = None


class SituationPatch(BaseModel):
    """Body of POST /calls/demo/situation. Only the fields sent are changed.

    Choosing ``incident_type`` without ``incident_confirmed`` confirms it (the dispatcher chose it).
    Choosing ``fire_smoke`` while a caller report mentions smoke also sets ``smoke_reported``
    (the dispatcher has confirmed the smoke), unless the patch sets ``smoke_reported`` itself.
    ``dismiss_suggestion`` drops the interpreter's pending suggestion and stops it coming back.
    Gates and timestamps have their own endpoints (gates are undoable).
    """

    model_config = ConfigDict(extra="forbid")

    incident_type: Optional[IncidentType] = None
    incident_confirmed: Optional[bool] = None
    dismiss_suggestion: bool = False
    witnessed: Optional[bool] = None
    cpr_in_progress: Optional[bool] = None
    smoke_reported: Optional[bool] = None
    load: Optional[CrewLoad] = None
    call_started_at: Optional[datetime] = None
    meet_person: Optional[str] = None
    n_teams: Optional[int] = Field(default=None, ge=1, le=MAX_TEAMS)


# ---------------------------------------------------------------------------
# Scenarios (sim/generate_calls.py)
# ---------------------------------------------------------------------------


def _scenario_table() -> dict[str, tuple[ScenarioSummary, list[dict], dict]]:
    """id -> (summary, script, starting situation fields). Step I1 scenarios first, then the
    older scripted calls (all on the synthetic building, incident type unset)."""
    out: dict[str, tuple[ScenarioSummary, list[dict], dict]] = {}
    for d in scenario_definitions():
        summary = ScenarioSummary(
            id=d["id"], title=d["title"], building_id=d["building_id"], incident_type=d["incident_type"],
            description=d.get("description", ""), true_room_id=d.get("true_room_id"), turns=len(d["script"]),
        )
        out[d["id"]] = (summary, d["script"], dict(d.get("situation") or {}))
    for name, script in scripted_calls().items():
        if name in out:
            continue
        summary = ScenarioSummary(id=name, title=name.replace("_", " ").capitalize(), building_id=DEMO_BUILDING_ID,
                                  turns=len(script))
        out[name] = (summary, script, {})
    return out


def available_scenarios() -> list[ScenarioSummary]:
    return [summary for summary, _, _ in _scenario_table().values()]


def scenario_script(scenario_id: str) -> tuple[ScenarioSummary, list[dict]]:
    """``(summary, script)`` for a scenario id; KeyError when unknown."""
    summary, script, _ = _scenario_table()[scenario_id]
    return summary, script


def scenario_situation(scenario_id: str) -> Situation:
    """The situation a scenario starts with (KeyError when unknown). Scripted calls start
    with the incident type unset: the dispatcher confirms what the interpreter suggests."""
    _, _, fields = _scenario_table()[scenario_id]
    return Situation.model_validate(fields)


def building_summaries() -> list[BuildingSummary]:
    return [
        BuildingSummary(id=b.id, name=b.name, real=bool(load_meta(b.id).get("real")),
                        floors=len(b.floors), rooms=len(b.rooms))
        for b in list_buildings()
    ]


def _empty_approach() -> dict[str, Any]:
    """The approach when route planning is off: no entrance, staging spot, stairs or target.

    Same keys as ``planning.approach.recommend_approach``.
    """
    return {
        "entrance_id": "", "entrance_name": "", "entrance_latlng": (0.0, 0.0),
        "staging_id": "", "staging_name": "", "staging_latlng": (0.0, 0.0),
        "stairwell_id": None, "stairwell_name": None,
        "target_room_id": "", "target_floor_label": "",
        "entrance_expected_travel_s": {},
        "stairs_s_to_target_floor": None, "elevator_s_to_target_floor": None,
        "entrance_access_s": {}, "entrance_warnings": {},
        "meet_person_credit_s": 0.0, "predicted_curb_to_target_s": None,
        "warnings": [],
    }


def _given_looks(plan: SearchPlan, belief: dict[str, float], p0: float,
                 looks: list[tuple[str, str]]) -> SearchPlan:
    """Put the plan's chances on the scale the console shows: given the searches so far.

    ``plan_search`` reports ``p_find`` / ``cumulative_prob`` / ``cumulative_success`` on the
    scale of the belief *before* any look (its documented contract). The console shows the
    posterior after the looks (``residual_probs``, which renormalises), so the plan's numbers
    are divided by the mass still unaccounted for, ``p0 + sum_r p_r * prod(1 - q)``. With no
    looks that mass is 1 and nothing changes.
    """
    if not looks:
        return plan
    table = search_modes()
    by_room = looks_by_room(looks)
    left = p0 + sum(p * miss_factor(by_room.get(rid, ()), table) for rid, p in belief.items())
    if left <= 0.0 or abs(left - 1.0) < 1e-12:
        return plan
    k = 1.0 / left
    teams = [[st.model_copy(update={
        "p_find": None if st.p_find is None else round(st.p_find * k, 8),
        "cumulative_prob": round(min(1.0, st.cumulative_prob * k), 6),
    }) for st in team] for team in plan.teams]
    success = None if plan.cumulative_success is None else round(min(1.0, plan.cumulative_success * k), 6)
    return plan.model_copy(update={"teams": teams, "cumulative_success": success})


def _empty_plan(n_teams: int, note: str) -> SearchPlan:
    return SearchPlan(
        teams=[[] for _ in range(n_teams)], expected_time_to_find_s=0.0, baseline_nearest_s=0.0,
        baseline_highest_prob_s=0.0, entrance_id="", staging_latlng=(0.0, 0.0), notes=[note],
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CallSession:
    """One call against one building: scripted turns plus anything the dispatcher enters.

    * ``script``        scripted turns (default: the building's demo script).
    * ``fixed_hour``    hour of day for the prior. Scripted demo calls keep 14 so results are
                        reproducible; ``None`` uses the local hour of ``situation.call_started_at``.
    * ``situation``     starting situation (a scenario may set fields).
    * ``call_log_path`` where call events go (default: ``call_log_path()`` at write time).
    """

    def __init__(
        self,
        building: Building,
        parser_mode: str = "rules",
        script: Optional[list[dict]] = None,
        scenario_id: Optional[str] = None,
        situation: Optional[Situation] = None,
        fixed_hour: Optional[int] = DEMO_HOUR,
        call_log_path: Optional[Path | str] = None,
    ) -> None:
        self.building = building
        self.script = script if script is not None else demo_script_for(building.id, building)
        self.scenario_id = scenario_id
        self.parser_mode = parser_mode
        self.fixed_hour = fixed_hour
        self.call_log_path = Path(call_log_path) if call_log_path is not None else None
        self._initial_situation = situation or Situation()
        self.lock = threading.Lock()
        self._history_cache: dict[tuple, HistoryPoint] = {}
        self._match_cache: dict[str, EvidenceMatch] = {}
        self._rooms = {r.id: r for r in building.rooms}
        self._room_ids = set(self._rooms)
        self._entrance_ids = {e.get("id") for e in building.entrances}
        self.reset()

    # -- transcript --------------------------------------------------------

    def _append_turn(self, raw: dict, scripted: bool, question: Optional[Question] = None,
                     answer: Optional[Answer] = None) -> TranscriptTurn:
        fix = raw.get("phone_fix")
        if fix is not None and not isinstance(fix, PhoneFix):
            fix = PhoneFix.model_validate(fix)
        index = len(self.revealed)
        turn = TranscriptTurn(
            turn=index, title=raw.get("title", f"Turn {index}"), speaker=raw.get("speaker", "system"),
            text=raw["text"], phone_fix=fix, scripted=scripted, at=_now(), question=question, answer=answer,
        )
        self.revealed.append(turn)
        if turn.speaker == "caller":
            self._interpret(index)
        return turn

    def _interpret(self, index: int) -> None:
        prior = [e for i in sorted(self.interps) if i < index for e in self.interps[i].evidence]
        sit0 = self.situation
        known_type = sit0.incident_type if sit0.incident_type != "unknown" else sit0.suggested_type
        it = interpret_turn(
            self.building, self.revealed, index, self.parser_mode, prior_evidence=prior,
            prior_facts=[f for i in sorted(self.interps) if i < index for f in self.interps[i].facts],
            incident_type=known_type,
        )
        if it.source not in ("agent", "llm"):
            # The model did not read this turn (rules mode, or a fallback): caller reports come
            # from fixed patterns.
            facts, suggested, span = read_facts(self.building, self.revealed, index, start_k=len(it.facts))
            it = it.model_copy(update={
                "facts": [*it.facts, *facts],
                "suggested_type": it.suggested_type or suggested,
                "suggested_span": it.suggested_span if it.suggested_type else span,
            })
        self.interps[index] = it
        turn = self.revealed[index]
        turn.clues = [e.clue for e in it.evidence if e.kind == "vocab" and e.clue is not None]
        turn.parser = it.source
        turn.evidence_ids = [e.id for e in it.evidence]
        turn.fact_ids = [f.id for f in it.facts]
        sit = self.situation
        if (it.suggested_type and it.suggested_type != sit.incident_type
                and it.suggested_type not in sit.dismissed_types
                and not (active_threat_pending(sit) and it.suggested_type != "active_threat")):
            # A pending active-threat suggestion is never replaced by a milder one: it keeps the
            # route withheld until the dispatcher confirms or dismisses it.
            sit.suggested_type = it.suggested_type
            sit.suggested_span = it.suggested_span
        for f in it.facts:
            if f.premise_key == "meet_person" and not sit.meet_person:
                sit.meet_person = f.text

    # -- mutations ---------------------------------------------------------

    def reset(self) -> None:
        self.revealed: list[TranscriptTurn] = []
        self.interps: dict[int, Interpretation] = {}
        self.status_overrides: dict[str, str] = {}  # evidence id -> "active" | "rejected" (dispatcher)
        self.script_pos = 0  # index of the latest revealed scripted turn
        self.searched: list[SearchedRoom] = []
        self.found_room_id: Optional[str] = None
        self.found_at: Optional[datetime] = None
        self.n_teams = 1
        # (event, searched, (found room, found at), closed rooms, closed entrances) before each undoable action
        self.undo_stack: list[tuple[str, list[SearchedRoom], tuple[Optional[str], Optional[datetime]],
                                    list[str], list[str]]] = []
        self.situation = self._initial_situation.model_copy(deep=True)
        if self.situation.call_started_at is None:
            self.situation.call_started_at = _now()
        self.call_id = uuid.uuid4().hex[:12]
        self._history_cache.clear()
        self._match_cache.clear()
        self._append_turn(self.script[0], scripted=True)

    def step(self) -> None:
        if self.script_pos < len(self.script) - 1:
            self.script_pos += 1
            self._append_turn(self.script[self.script_pos], scripted=True)

    def say(self, text: str, speaker: str = "caller", answer: Optional[Answer] = None) -> None:
        """Something the caller said (typed, pasted or dictated), or a dispatcher turn."""
        title = "Caller (live)" if speaker == "caller" else "Dispatcher (live)"
        self._append_turn({"title": title, "speaker": speaker, "text": text}, scripted=False, answer=answer)

    def ask(self, req: AskRequest) -> None:
        """Log a question the dispatcher asked as a dispatcher turn carrying the question."""
        kind = req.kind or ("yes_no" if req.clue_type else "open")
        qid = req.id or (question_key(req.clue_type, req.value or "") if req.clue_type else f"open:{req.question}")
        question = Question(
            id=qid, question=req.question, kind=kind, clue_type=req.clue_type,  # type: ignore[arg-type]
            value=req.value, partition=req.partition,
        )
        self._append_turn({"title": "Dispatcher: question", "speaker": "dispatcher", "text": req.question},
                          scripted=False, question=question)

    def set_parser(self, mode: str) -> None:
        """Switch parser and re-read every revealed caller turn with it."""
        if mode == self.parser_mode:
            return
        self.parser_mode = mode
        self.interps.clear()
        for t in self.revealed:
            if t.speaker == "caller":
                self._interpret(t.turn)
        self._history_cache.clear()
        self._match_cache.clear()

    def _all_evidence_ids(self) -> set[str]:
        return {e.id for it in self.interps.values() for e in it.evidence}

    def set_evidence_status(self, evidence_id: str, status: str) -> None:
        """Dispatcher override: "rejected" (reject) or "active" (restore). KeyError for an unknown id."""
        if evidence_id not in self._all_evidence_ids():
            raise KeyError(evidence_id)
        self.status_overrides[evidence_id] = status
        self._history_cache.clear()

    def _check_room(self, room_id: str) -> None:
        if room_id not in self._room_ids:
            raise KeyError(room_id)

    def _push_undo(self, event: str) -> None:
        sit = self.situation
        self.undo_stack.append((event, list(self.searched), (self.found_room_id, self.found_at),
                                list(sit.closed_room_ids), list(sit.closed_entrance_ids)))

    def search_room(self, room_id: str, mode: str = "rapid", only_if_new: bool = False) -> None:
        """Record one look at a room. ``only_if_new`` keeps the old /clear endpoint idempotent."""
        self._check_room(room_id)
        if only_if_new and any(s.room_id == room_id for s in self.searched):
            return
        self._push_undo("searched")
        look = SearchedRoom(room_id=room_id, mode=mode, at=_now())  # type: ignore[arg-type]
        self.searched.append(look)
        prior_looks = sum(1 for s in self.searched[:-1] if s.room_id == room_id)
        self._log("searched", room_id=room_id, mode=mode, look=prior_looks + 1)

    def clear_room(self, room_id: str) -> None:
        """Pre-Step-0 name: a rapid search, recorded once per room."""
        self.search_room(room_id, "rapid", only_if_new=True)

    def mark_found(self, room_id: str) -> None:
        self._check_room(room_id)
        self._push_undo("found")
        self.found_room_id = room_id
        self.found_at = _now()
        self._log("found", room_id=room_id, looks_before=len(self.searched))

    def set_gates(self, gates: Gates) -> None:
        """Rooms and entrances the officer closed (replaces both lists). KeyError for an unknown id."""
        for rid in gates.closed_room_ids:
            self._check_room(rid)
        for eid in gates.closed_entrance_ids:
            if eid not in self._entrance_ids:
                raise KeyError(eid)
        self._push_undo("gate")
        self.situation.closed_room_ids = list(dict.fromkeys(gates.closed_room_ids))
        self.situation.closed_entrance_ids = list(dict.fromkeys(gates.closed_entrance_ids))
        self._log("gate", closed_room_ids=list(self.situation.closed_room_ids),
                  closed_entrance_ids=list(self.situation.closed_entrance_ids))

    def undo(self) -> bool:
        """Undo the last searched, found or gate action. False when there is nothing to undo."""
        if not self.undo_stack:
            return False
        event, searched, found, rooms, entrances = self.undo_stack.pop()
        undone_room = None
        if event == "searched" and self.searched:
            undone_room = self.searched[-1].room_id
        elif event == "found":
            undone_room = self.found_room_id
        self.searched, (self.found_room_id, self.found_at) = searched, found
        self.situation.closed_room_ids, self.situation.closed_entrance_ids = rooms, entrances
        self._log("undo", undone=event, room_id=undone_room)
        return True

    def log_timestamp(self, name: TimestampName) -> None:
        self.situation.timestamps[name] = _now()
        self._log("timestamp", name=name)

    def patch_situation(self, patch: SituationPatch) -> None:
        data = patch.model_dump(exclude_unset=True)
        sit = self.situation
        if data.pop("dismiss_suggestion", False) and sit.suggested_type:
            if sit.suggested_type not in sit.dismissed_types:
                sit.dismissed_types.append(sit.suggested_type)
            sit.suggested_type, sit.suggested_span = None, None
        if "n_teams" in data:
            n = data.pop("n_teams")
            if n is not None:
                self.n_teams = n
        if "incident_type" in data:
            new_type = data.pop("incident_type") or "unknown"
            sit.incident_type = new_type
            sit.incident_confirmed = bool(data.pop("incident_confirmed", True))
            if sit.suggested_type == new_type:
                sit.suggested_type, sit.suggested_span = None, None
            if new_type == "fire_smoke" and "smoke_reported" not in data and self._caller_reported_smoke():
                sit.smoke_reported = True  # the dispatcher confirmed the caller's smoke report
        for key, value in data.items():
            if value is None and key in _NOT_NULLABLE:
                continue  # an explicit null only clears the tri-state fields (witnessed, cpr, meet_person)
            setattr(sit, key, value)
        self._history_cache.clear()

    # -- call log ----------------------------------------------------------

    def _log(self, event: str, **fields: Any) -> None:
        """Append one call event to the call log. A write failure never breaks the call."""
        room_id = fields.get("room_id")
        room = self._rooms.get(room_id) if room_id else None
        record = {
            "event": event,
            "call_id": self.call_id,
            "building_id": self.building.id,
            "scenario_id": self.scenario_id,
            "at": _now().isoformat(),
            "t_s": round(self.elapsed_s(), 3),
            "incident_type": self.situation.incident_type,
            "incident_confirmed": self.situation.incident_confirmed,
            "smoke_reported": self.situation.smoke_reported,
            "room_id": room_id,
            "room_area_m2": round(float(room.area_m2), 2) if room is not None else None,
            "room_type": room.room_type if room is not None else None,
            "floor_label": room.floor_label if room is not None else None,
            "mode": fields.pop("mode", None),
            **{k: v for k, v in fields.items() if k != "room_id"},
        }
        path = self.call_log_path or call_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    # -- evidence ----------------------------------------------------------

    def _evidence(self, upto: int) -> tuple[Optional[PhoneFix], list[Evidence]]:
        """Latest phone fix and every evidence item up to turn ``upto``, with statuses resolved."""
        fix: Optional[PhoneFix] = None
        for t in self.revealed[: upto + 1]:
            if t.phone_fix is not None:
                fix = t.phone_fix  # latest fix wins
        items: list[Evidence] = []
        superseded: set[str] = set()
        for i in sorted(self.interps):
            if i > upto:
                break
            superseded.update(self.interps[i].supersedes)
            items.extend(self.interps[i].evidence)
        out = []
        for e in items:
            status = e.status
            if e.id in superseded:
                status = "superseded"
            status = self.status_overrides.get(e.id, status)
            m = self._match(e)
            out.append(e.model_copy(update={"status": status, "matched_room_ids": m.matched_room_ids,
                                            "unmatched": m.unmatched}))
        return fix, out

    def _match(self, e: Evidence) -> EvidenceMatch:
        key = e.model_dump_json(exclude={"status", "matched_room_ids", "unmatched"})
        if key not in self._match_cache:
            self._match_cache[key] = evidence_match(self.building, e)
        return self._match_cache[key]

    def _facts(self, upto: Optional[int] = None) -> list[Fact]:
        upto = self._last if upto is None else upto
        return [f for i in sorted(self.interps) if i <= upto for f in self.interps[i].facts]

    def _caller_reported_smoke(self) -> bool:
        return any(f.category == "hazard" and _SMOKE_WORDS.search(f.text) for f in self._facts())

    @property
    def prior_hour(self) -> int:
        if self.fixed_hour is not None:
            return self.fixed_hour
        started = self.situation.call_started_at or _now()
        return started.astimezone().hour

    def _searched_ids(self) -> list[str]:
        """Distinct searched rooms in first-look order."""
        return list(dict.fromkeys(s.room_id for s in self.searched))

    def _looks(self) -> list[tuple[str, SearchMode]]:
        """Every look so far as ``(room_id, mode)``, oldest first, repeats included."""
        return [(s.room_id, s.mode) for s in self.searched]

    def _gates(self) -> Gates:
        sit = self.situation
        return Gates(closed_room_ids=list(sit.closed_room_ids), closed_entrance_ids=list(sit.closed_entrance_ids))

    def _fused(self, fix: Optional[PhoneFix], items: list[Evidence]) -> Fused:
        """Evidence-only belief with the notices about it: prior x phone fix x active evidence,
        before any search.

        Same maths as ``core.fusion.posterior_probs`` (``core.fusion.fuse_evidence``); match
        vectors come from the session cache. ``Fused.notices`` are the console notices for
        caller statements that conflict with each other or with the phone's height fix.
        """
        prior, p0 = compute_prior(self.building, hour=self.prior_hour)
        return fuse_evidence(self.building, prior, p0, fix, items, match_of=self._match)

    def _belief(self, fix: Optional[PhoneFix], items: list[Evidence]) -> tuple[dict[str, float], float]:
        """Evidence-only belief (``_fused`` without the notices)."""
        fused = self._fused(fix, items)
        return fused.room_probs, fused.p_not_in_model

    def _probs(self, upto: int) -> tuple[dict[str, float], float]:
        return self._belief(*self._evidence(upto))

    def _history_point(self, index: int, policy: Policy, gates: Gates) -> HistoryPoint:
        """Evidence-only snapshot after a revealed turn (single team, nothing searched,
        expected-time objective). Cached per turn, travel profile and gates."""
        key = (index, policy.route_enabled, policy.travel.model_dump_json(),
               tuple(gates.closed_room_ids), tuple(gates.closed_entrance_ids))
        if key not in self._history_cache:
            room_probs, p_nim = self._probs(index)
            post = summarize(self.building, room_probs, p_nim)
            et = (
                plan_search(self.building, room_probs, p_nim, n_teams=1, profile=policy.travel,
                            gates=gates).expected_time_to_find_s
                if policy.route_enabled else 0.0
            )
            top_id = max(room_probs, key=room_probs.get)
            self._history_cache[key] = HistoryPoint(
                turn=index, entropy_bits=post.entropy_bits, expected_time_to_find_s=et,
                top_room_id=top_id, top_room_prob=room_probs[top_id],
            )
        return self._history_cache[key]

    # -- views -------------------------------------------------------------

    @property
    def _last(self) -> int:
        return len(self.revealed) - 1

    def elapsed_s(self) -> float:
        started = self.situation.call_started_at or _now()
        return max(0.0, (_now() - started).total_seconds())

    def policy(self) -> Policy:
        return policy_for(self.building, self.situation, self._facts(), elapsed_s=self.elapsed_s())

    def statement_notices(self) -> list:
        """Notices (``core.schema.Banner``) about caller statements that conflict with each
        other or with the phone's height fix (``core.fusion.statement_notices``)."""
        return list(self._fused(*self._evidence(self._last)).notices)

    def posterior(self) -> Posterior:
        belief, p0 = self._probs(self._last)
        room_probs, p_nim = residual_probs(belief, p0, self._looks())
        return summarize(self.building, room_probs, p_nim)

    def plan(self, n_teams: Optional[int] = None) -> SearchPlan:
        fused = self._fused(*self._evidence(self._last))
        policy = self.policy()
        return self._plan(fused.room_probs, fused.p_not_in_model, n_teams or self.n_teams, policy, self._gates(),
                          self._phone_floor_rooms(fused))

    @staticmethod
    def _phone_floor_rooms(fused: Fused) -> Optional[frozenset]:
        """Rooms on the floors the phone's height fix allows, when a caller statement
        contradicts the fix (the second team starts there); None otherwise."""
        return fused.phone_band if fused.phone_conflicts else None

    def _plan(self, belief: dict[str, float], p0: float, n_teams: int, policy: Policy, gates: Gates,
              second_team_rooms: Optional[frozenset] = None) -> SearchPlan:
        """The search plan, or an empty one when the policy turns routes off (never computed then).

        ``belief`` is evidence only; ``plan_search`` applies every look itself, so searched
        rooms keep their residual mass and can return as a second look (``kind="re_search"``).
        Only officer gates take rooms out. The crew replans from the room it searched last.
        """
        if not policy.route_enabled:
            return _empty_plan(n_teams, "Route planning is off for this incident type.")
        if self.found_room_id is not None:
            return _empty_plan(n_teams, "Search ended: the dispatcher recorded the caller as found.")
        start = self.searched[-1].room_id if self.searched else None
        looks = self._looks()
        plan = plan_search(
            self.building, belief, p0, n_teams=n_teams, start_node=start,
            profile=policy.travel, survival=survival_from(self.situation, self.elapsed_s()),
            searched=looks, gates=gates, second_team_rooms=second_team_rooms,
        )
        return self._with_hazard_classes(_given_looks(plan, belief, p0, looks), policy)

    def hazard_classes(self, facts: Optional[list[Fact]] = None) -> dict[str, str]:
        """room id -> hazard class from caller facts that name a room (display only).

        A hazard fact about fire or smoke in a named room -> ``fire_room``; a room whose door
        the caller said is closed -> ``closed_door``; every other room on those floors ->
        ``open_path``. Nothing when no fact names a room.
        """
        facts = self._facts() if facts is None else facts
        fire = {f.room_id for f in facts
                if f.room_id in self._rooms and f.category == "hazard" and _FIRE_WORDS.search(f.text)}
        closed = {f.room_id for f in facts
                  if f.room_id in self._rooms and f.category in ("access", "hazard") and _CLOSED_DOOR.search(f.text)}
        closed -= fire
        out: dict[str, str] = {rid: "fire_room" for rid in fire}
        out.update({rid: "closed_door" for rid in closed if rid})
        floors = {self._rooms[rid].floor for rid in out}
        for r in self.building.rooms:
            if r.floor in floors and r.id not in out:
                out[r.id] = "open_path"
        return out

    def _with_hazard_classes(self, plan: SearchPlan, policy: Policy) -> SearchPlan:
        if not policy.show_hazard_class:
            return plan
        classes = self.hazard_classes()
        if not classes:
            return plan
        teams = [[st.model_copy(update={"hazard_class": classes.get(st.room_id)}) for st in team] for team in plan.teams]
        return plan.model_copy(update={"teams": teams})

    def _approach(self, room_probs: dict[str, float], policy: Policy, gates: Gates) -> dict[str, Any]:
        if not policy.route_enabled:
            return _empty_approach()
        return recommend_approach(self.building, room_probs, profile=policy.travel, situation=self.situation,
                                  gates=gates)

    def _asked_keys(self, evidence: list[Evidence]) -> set[str]:
        """Every key that counts as asked or answered, for next_question / top_questions.

        * clues the caller gave (``asked_from_clues``: asserted ``type:value``, denied
          ``negative:<section>:<value>``), active vocabulary evidence only;
        * logged questions and button answers (``parser.dialog.asked_keys``);
        * ``unit:<x>`` / ``tenant:<x>`` for a unit or tenant the caller named, so the open
          question retires.

        A floor yes/no question that was asked but not answered "yes" is kept as
        ``negative:floors:<x>``: it is not offered again, while the open floor question
        ("What floor are you on?") still is.
        """
        clues = [e.clue for e in evidence if e.kind == "vocab" and e.clue is not None and e.status == "active"]
        asked = asked_from_clues(clues) | {question_key(c.type, c.value) for c in clues if c.polarity}
        asked |= dialog_asked_keys(self.revealed)
        for e in evidence:
            sel = e.selector
            if e.kind != "grounded" or sel is None or e.status == "rejected" or e.negate:
                continue
            if sel.unit:
                asked.add(f"unit:{normalize_unit(sel.unit)}")
            if sel.tenant_contains:
                asked.add(f"tenant:{' '.join(sel.tenant_contains.lower().split())}")
        asserted_floors = {c.value for c in clues if c.type == "floor" and c.polarity}
        for key in list(asked):
            head, _, value = key.partition(":")
            if head == "floor" and value and value not in asserted_floors:
                asked.discard(key)
                asked.add(f"negative:{SECTION_FOR_TYPE['floor']}:{value}")
        return asked

    def state(self) -> DemoState:
        policy = self.policy()
        gates = self._gates()
        fix, evidence = self._evidence(self._last)
        fused = self._fused(fix, evidence)
        belief, p0 = fused.room_probs, fused.p_not_in_model
        if fused.notices:  # shown with the policy banners, after the policy's own
            policy = policy.model_copy(update={"banners": [*policy.banners, *fused.notices]})
        room_probs, p_nim = residual_probs(belief, p0, self._looks())
        clues = [e.clue for e in evidence if e.kind == "vocab" and e.clue is not None and e.status == "active"]
        asked = self._asked_keys(evidence)
        return DemoState(
            building_id=self.building.id,
            turn=self.script_pos,
            total_turns=len(self.script),
            done=self.script_pos >= len(self.script) - 1,
            transcript=list(self.revealed),
            clues=clues,
            phone_fix=fix,
            cleared_room_ids=self._searched_ids(),
            n_teams=self.n_teams,
            posterior=summarize(self.building, room_probs, p_nim),
            plan=self._plan(belief, p0, self.n_teams, policy, gates, self._phone_floor_rooms(fused)),
            approach=self._approach(room_probs, policy, gates),
            next_question=next_question(self.building, room_probs, p_nim, asked=asked, policy=policy),
            history=[self._history_point(i, policy, gates) for i in range(len(self.revealed))],
            parser_mode=self.parser_mode,
            llm=LLMStatus(available=llm_available(), model=agent_model(), usage=llm_usage()),
            buildings=building_summaries(),
            situation=self.situation.model_copy(deep=True),
            policy=policy,
            evidence=evidence,
            facts=self._facts(),
            questions=top_questions(self.building, room_probs, p_nim, asked=asked, policy=policy),
            searched=list(self.searched),
            found_room_id=self.found_room_id,
            found_at=self.found_at,
            scenarios=available_scenarios(),
            agent_trace=[e for i in sorted(self.interps) for e in self.interps[i].trace],
            scenario_id=self.scenario_id,
            prior_hour=self.prior_hour,
            can_undo=bool(self.undo_stack),
        )


def start_scenario(scenario_id: str, building: Building, parser_mode: str = "rules",
                   call_log_path: Optional[Path | str] = None) -> CallSession:
    """A new session running a scripted scenario (KeyError when unknown)."""
    summary, script = scenario_script(scenario_id)
    return CallSession(building, parser_mode=parser_mode, script=script, scenario_id=summary.id,
                       situation=scenario_situation(scenario_id), call_log_path=call_log_path)


__all__ = [
    "AskRequest", "BuildingSummary", "CALL_LOG_ENV", "CallSession", "DEMO_HOUR", "DemoState", "HistoryPoint",
    "LLMStatus", "MAX_TEAMS", "MAX_UTTERANCE_CHARS", "ParserMode", "ScenarioSummary", "SearchMode",
    "SituationPatch", "Utterance", "available_scenarios", "building_summaries", "call_log_path", "llm_available",
    "scenario_script", "scenario_situation", "start_scenario",
]

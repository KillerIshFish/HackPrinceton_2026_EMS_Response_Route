"""FastAPI backend for the 911 Caller Location Predictor (v0, synthetic data).

Run from the repo root::

    uv run uvicorn api.server:app --reload --port 8000

v0 holds ONE in-memory demo call. Every mutating endpoint returns the full
``DemoState`` so the web client can simply replace its state.

Pipeline per request (all real math, no network calls)::

    script turns -> parser.extract (rule-based) -> clues
    core.fusion.posterior_probs(prior, phone fix, clues)
    planning.search_order.koopman_update   for every room marked clear
    core.uncertainty.summarize             -> Posterior
    planning.search_order.plan_search      -> SearchPlan (+ baselines)
    planning.approach.recommend_approach   -> entrance / staging / stairwell
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.fusion import posterior_probs
from core.schema import Building, Clue, PhoneFix, Posterior, SearchPlan
from core.uncertainty import summarize
from ingest.matterport import list_buildings, load_building
from parser.extract import extract_clues
from parser.next_question import next_question
from planning.approach import recommend_approach
from planning.search_order import koopman_update, plan_search
from sim.generate_calls import demo_call_script

load_dotenv()  # v0 reads nothing secret; keeps the seam ready for LLM_PROVIDER / LLM_API_KEY

DEMO_BUILDING_ID = "ptc-100"
DEMO_HOUR = 14  # 2 pm on a weekday: business-hours prior
MAX_TEAMS = 3


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TranscriptTurn(BaseModel):
    turn: int
    title: str
    speaker: str
    text: str
    phone_fix: Optional[PhoneFix] = None
    clues: list[Clue] = []


class HistoryPoint(BaseModel):
    turn: int
    entropy_bits: float
    expected_time_to_find_s: float
    top_room_id: str
    top_room_prob: float


class DemoState(BaseModel):
    building_id: str
    turn: int
    total_turns: int
    done: bool
    transcript: list[TranscriptTurn]
    clues: list[Clue]
    phone_fix: Optional[PhoneFix] = None
    cleared_room_ids: list[str]
    n_teams: int
    posterior: Posterior
    plan: SearchPlan
    approach: dict[str, Any]
    next_question: Optional[dict[str, Any]] = None
    history: list[HistoryPoint]


# ---------------------------------------------------------------------------
# In-memory demo session
# ---------------------------------------------------------------------------


class DemoSession:
    """One scripted call against the synthetic building."""

    def __init__(self, building: Building) -> None:
        self.building = building
        self.script = [self._normalise_turn(i, t) for i, t in enumerate(demo_call_script())]
        self.lock = threading.Lock()
        self._history_cache: dict[int, HistoryPoint] = {}
        self.reset()

    @staticmethod
    def _normalise_turn(i: int, raw: dict) -> TranscriptTurn:
        fix = raw.get("phone_fix")
        if fix is not None and not isinstance(fix, PhoneFix):
            fix = PhoneFix.model_validate(fix)
        text = raw["text"]
        speaker = raw.get("speaker", "system")
        clues = extract_clues(text) if speaker == "caller" else []
        return TranscriptTurn(
            turn=raw.get("turn", i),
            title=raw.get("title", f"Turn {i}"),
            speaker=speaker,
            text=text,
            phone_fix=fix,
            clues=clues,
        )

    # -- mutations ---------------------------------------------------------

    def reset(self) -> None:
        self.turn = 0
        self.cleared: list[str] = []
        self.n_teams = 1

    def step(self) -> None:
        if self.turn < len(self.script) - 1:
            self.turn += 1

    def clear_room(self, room_id: str) -> None:
        if room_id not in {r.id for r in self.building.rooms}:
            raise KeyError(room_id)
        if room_id not in self.cleared:
            self.cleared.append(room_id)

    # -- evidence ----------------------------------------------------------

    def _evidence(self, upto_turn: int) -> tuple[Optional[PhoneFix], list[Clue]]:
        fix: Optional[PhoneFix] = None
        clues: list[Clue] = []
        for t in self.script[: upto_turn + 1]:
            if t.phone_fix is not None:
                fix = t.phone_fix  # latest fix wins
            clues.extend(t.clues)
        return fix, clues

    def _probs(self, upto_turn: int, cleared: list[str]) -> tuple[dict[str, float], float]:
        fix, clues = self._evidence(upto_turn)
        room_probs, p_nim = posterior_probs(self.building, hour=DEMO_HOUR, fix=fix, clues=clues)
        for room_id in cleared:
            room_probs, p_nim = koopman_update(room_probs, p_nim, room_id)
        return room_probs, p_nim

    def _history_point(self, turn: int) -> HistoryPoint:
        """Evidence-only snapshot for a turn (single team, nothing cleared). Cached."""
        if turn not in self._history_cache:
            room_probs, p_nim = self._probs(turn, cleared=[])
            post = summarize(self.building, room_probs, p_nim)
            plan = plan_search(self.building, room_probs, p_nim, n_teams=1)
            top_id = max(room_probs, key=room_probs.get)
            self._history_cache[turn] = HistoryPoint(
                turn=turn,
                entropy_bits=post.entropy_bits,
                expected_time_to_find_s=plan.expected_time_to_find_s,
                top_room_id=top_id,
                top_room_prob=room_probs[top_id],
            )
        return self._history_cache[turn]

    # -- views -------------------------------------------------------------

    def posterior(self) -> Posterior:
        room_probs, p_nim = self._probs(self.turn, self.cleared)
        return summarize(self.building, room_probs, p_nim)

    def plan(self, n_teams: Optional[int] = None) -> SearchPlan:
        room_probs, p_nim = self._probs(self.turn, self.cleared)
        return self._plan(room_probs, p_nim, n_teams or self.n_teams)

    def _plan(self, room_probs: dict[str, float], p_nim: float, n_teams: int) -> SearchPlan:
        # After a room is cleared the crew replans from where it is standing, and
        # cleared rooms are not offered again (their residual mass stays in the posterior).
        start = self.cleared[-1] if self.cleared else None
        return plan_search(
            self.building,
            room_probs,
            p_nim,
            n_teams=n_teams,
            start_node=start,
            exclude=tuple(self.cleared),
        )

    def state(self) -> DemoState:
        fix, clues = self._evidence(self.turn)
        room_probs, p_nim = self._probs(self.turn, self.cleared)
        asked = {f"{c.type}:{c.value}" for c in clues} | {c.value for c in clues}
        return DemoState(
            building_id=self.building.id,
            turn=self.turn,
            total_turns=len(self.script),
            done=self.turn >= len(self.script) - 1,
            transcript=self.script[: self.turn + 1],
            clues=clues,
            phone_fix=fix,
            cleared_room_ids=list(self.cleared),
            n_teams=self.n_teams,
            posterior=summarize(self.building, room_probs, p_nim),
            plan=self._plan(room_probs, p_nim, self.n_teams),
            approach=recommend_approach(self.building, room_probs),
            next_question=next_question(self.building, room_probs, p_nim, asked=asked),
            history=[self._history_point(t) for t in range(self.turn + 1)],
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="911 Caller Location Predictor",
    version="0.1.0",
    description="v0: synthetic building, scripted demo call, rule-based clue parser.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: Optional[DemoSession] = None
_session_lock = threading.Lock()


def get_session() -> DemoSession:
    global _session
    with _session_lock:
        if _session is None:
            _session = DemoSession(load_building(DEMO_BUILDING_ID))
        return _session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/buildings", response_model=list[Building])
def get_buildings() -> list[Building]:
    return list_buildings()


@app.get("/buildings/{building_id}", response_model=Building)
def get_building(building_id: str) -> Building:
    try:
        return load_building(building_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown building '{building_id}'")


@app.get("/calls/demo/state", response_model=DemoState)
def demo_state() -> DemoState:
    s = get_session()
    with s.lock:
        return s.state()


@app.post("/calls/demo/step", response_model=DemoState)
def demo_step() -> DemoState:
    """Reveal the next scripted call turn (no-op once the script is finished)."""
    s = get_session()
    with s.lock:
        s.step()
        return s.state()


@app.post("/calls/demo/reset", response_model=DemoState)
def demo_reset() -> DemoState:
    s = get_session()
    with s.lock:
        s.reset()
        return s.state()


@app.post("/calls/demo/teams", response_model=DemoState)
def demo_teams(n: int = Query(1, ge=1, le=MAX_TEAMS)) -> DemoState:
    s = get_session()
    with s.lock:
        s.n_teams = n
        return s.state()


@app.get("/posterior", response_model=Posterior)
def get_posterior() -> Posterior:
    s = get_session()
    with s.lock:
        return s.posterior()


@app.get("/plan", response_model=SearchPlan)
def get_plan(teams: Optional[int] = Query(None, ge=1, le=MAX_TEAMS)) -> SearchPlan:
    s = get_session()
    with s.lock:
        return s.plan(teams)


@app.post("/rooms/{room_id}/clear", response_model=DemoState)
def clear_room(room_id: str) -> DemoState:
    """Dispatcher marks a room searched-and-empty: Koopman update p_r <- p_r (1 - 0.9), then replan."""
    s = get_session()
    with s.lock:
        try:
            s.clear_room(room_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown room '{room_id}'")
        return s.state()

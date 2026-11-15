"""FROZEN CONTRACT - shared data models for every module.

Do not rename or remove fields without telling the whole team: ingest/, core/,
parser/, planning/, api/ and web/ all depend on these exact shapes.

Conventions
-----------
* Coordinates are building-local metres: x = east, y = north, z = up. z = 0 is
  the finished floor of the lowest floor in ``Building.floors``.
* ``Room.floor`` is the integer floor index; ``Room.floor_label`` is what a
  human says ("1", "2", "M", "B1").
* ``Building.floor_heights_m[floor]`` is the finished-floor elevation of that
  floor above the z = 0 datum.
* Probabilities: ``sum(Posterior.room_probs.values()) + p_not_in_model == 1``.

Free-form dict shapes (kept as dicts in the frozen contract, keys by convention)
--------------------------------------------------------------------------------
Building.edges[i]::

    {"u": room_id, "v": room_id,
     "kind": "door" | "corridor" | "stairs" | "elevator",
     "length_m": float,          # walking distance between the two centroids
     "floors": int}              # abs floor delta (0 unless stairs/elevator)

    Optional keys (read by core/route_clues.py; absent = fall back to room outlines):
     "xy": [x, y]                          # door / corridor edge: where the opening is
     "top_xy": [x, y], "heading": [dx, dy]            # stairs: arrival point and travel direction going up
     "bottom_xy": [x, y], "bottom_heading": [dx, dy]  # stairs: the same going down

Building.entrances[i]::

    {"id": str, "name": str, "room_id": str,     # room the door opens into
     "xy": [x, y], "latlng": [lat, lng],
     "facing": "street" | "parking_lot"}

Building.staging_spots[i]::

    {"id": str, "name": str, "xy": [x, y], "latlng": [lat, lng],
     "kind": "street" | "parking_lot", "near_entrance_id": str}
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# "route" was added after the freeze (additive): how the caller reached the room, e.g.
# "turned right after coming up the stairs". Values are keys of `routes` in clue_vocab.yaml.
# "layout" (additive): an answer to a layout yes / no question ("is it a corner room?"); its
# value is a key of parser.layout.LAYOUT_QUESTIONS.
ClueType = Literal["room_type", "floor", "landmark", "view", "sound", "smell", "negative", "route", "layout"]

# Enumerations of the Step 0 contract (docs/BUILD_PLAN.md section 4). Spelling is the contract.
IncidentType = Literal[
    "unknown", "cardiac_arrest", "stroke", "internal_bleeding", "external_bleeding",
    "overdose", "fire_smoke", "active_threat", "other_medical",
]
INCIDENT_TYPES: tuple[str, ...] = (
    "unknown", "cardiac_arrest", "stroke", "internal_bleeding", "external_bleeding",
    "overdose", "fire_smoke", "active_threat", "other_medical",
)
CrewLoad = Literal["light", "heavy"]
SearchMode = Literal["rapid", "thorough"]
StopKind = Literal["first", "re_search"]
HazardClass = Literal["fire_room", "open_path", "closed_door"]
EvidenceKind = Literal["vocab", "grounded"]
EvidenceSource = Literal["rules", "llm", "agent", "dispatcher"]
EvidenceStatus = Literal["active", "rejected", "superseded"]
FactCategory = Literal["hazard", "access", "patient", "people", "threat", "resource"]
# Facts can also come from the premise record (premise facts panel), hence "record".
FactSource = Literal["rules", "llm", "agent", "dispatcher", "record"]
PremiseKey = Literal["means_of_entry", "meet_person", "elevator", "defibrillator"]
TimestampName = Literal["wheels_stopped", "building_entry", "floor_arrival", "patient_contact"]
TIMESTAMP_NAMES: tuple[str, ...] = ("wheels_stopped", "building_entry", "floor_arrival", "patient_contact")
BannerLevel = Literal["info", "warning", "critical"]
QuestionKind = Literal["yes_no", "open"]
AnswerValue = Literal["yes", "no", "unsure"]
PlanObjective = Literal["expected_time", "survival"]
Solver = Literal["exact", "heuristic"]
Speaker = Literal["system", "caller", "dispatcher"]
ParserMode = Literal["rules", "llm", "agent"]


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    floor: int
    floor_label: str
    polygon: list[tuple[float, float]]
    centroid: tuple[float, float, float]
    area_m2: float
    room_type: str
    has_windows: bool
    window_facing: Optional[str] = None
    near_elevator: bool
    near_stairs: bool
    occupancy_factor: float


class Building(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    use_type: str
    floors: list[int]
    floor_heights_m: dict[int, float]
    rooms: list[Room]
    edges: list[dict]
    entrances: list[dict]
    staging_spots: list[dict]


class Clue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ClueType
    value: str
    polarity: bool
    confidence: float = Field(ge=0.0, le=1.0)
    span: str


class Posterior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_probs: dict[str, float]
    p_not_in_model: float
    p_floor: dict[str, float]
    credible_set_90: list[str]
    entropy_bits: float
    peak_count: int


class Stop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str
    room_name: str
    floor_label: str
    eta_s: float
    cumulative_prob: float
    # Additive (Step 0): filled by the planner from A1 on.
    kind: StopKind = "first"
    mode: SearchMode = "rapid"
    p_find: Optional[float] = None
    travel_s: Optional[float] = None
    search_s: Optional[float] = None
    hazard_class: Optional[HazardClass] = None  # fire_smoke only, display only


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teams: list[list[Stop]]
    expected_time_to_find_s: float
    baseline_nearest_s: float
    baseline_highest_prob_s: float
    entrance_id: str
    staging_latlng: tuple[float, float]
    # Additive (Step 0): filled by the planner from A1 on.
    objective: PlanObjective = "expected_time"
    solver: Solver = "heuristic"
    p_out: Optional[float] = None
    cumulative_success: Optional[float] = None
    time_to_cover_all_s: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Supplementary model (not one of the six frozen contract models above, but
# shared by sim/, core/phone_likelihood.py and api/).
# ---------------------------------------------------------------------------

# FCC z-axis benchmark (as cited in eval/benchmark/common.py): +/-3 m for 80 % of calls. 1.28 is the
# standard normal's two-sided 80 % quantile, so a Gaussian with this sigma puts 80 % of fixes within 3 m.
Z_SIGMA_M = 3.0 / 1.28  # +/-3 m is the 80% z-axis bound -> sigma ~= 2.34 m


class PhoneFix(BaseModel):
    """Device location fix in building-local metres. Any component may be missing."""

    model_config = ConfigDict(extra="forbid")

    x_m: Optional[float] = None
    y_m: Optional[float] = None
    horizontal_sigma_m: Optional[float] = None
    z_m: Optional[float] = None  # barometric height above the z = 0 datum
    z_sigma_m: float = Z_SIGMA_M


# ---------------------------------------------------------------------------
# Situation, evidence and policy (Step 0 contract, docs/BUILD_PLAN.md section 4).
# Additive only: every field has a default except identifiers and the text of a
# record. Datetimes serialise as ISO 8601 strings. Fields marked "beyond section 4"
# were added in Step 0 because a later step needs them; they are optional too.
# ---------------------------------------------------------------------------


class Situation(BaseModel):
    """What kind of emergency this is and the conditions the crew faces.

    Set by the dispatcher. The interpreter may only *suggest* an incident type
    (``suggested_type`` + the caller's verbatim ``suggested_span``); it never switches
    ``incident_type`` itself.
    """

    model_config = ConfigDict(extra="forbid")

    incident_type: IncidentType = "unknown"
    incident_confirmed: bool = False
    suggested_type: Optional[IncidentType] = None
    suggested_span: Optional[str] = None
    # beyond section 4: types the dispatcher dismissed, so they are not suggested again
    dismissed_types: list[IncidentType] = Field(default_factory=list)
    witnessed: Optional[bool] = None  # None = not known
    cpr_in_progress: Optional[bool] = None  # None = not known
    smoke_reported: bool = False
    load: CrewLoad = "light"
    call_started_at: Optional[datetime] = None
    closed_room_ids: list[str] = Field(default_factory=list)  # officer gates only
    closed_entrance_ids: list[str] = Field(default_factory=list)  # officer gates only
    meet_person: Optional[str] = None  # who will meet the crew, in words; None = nobody recorded
    timestamps: dict[TimestampName, datetime] = Field(default_factory=dict)


class WithinM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str
    radius_m: float = Field(gt=0.0)


class RoomSelector(BaseModel):
    """Rooms of this building a statement points at.

    Every condition is optional and the set ones combine with AND. ``None`` means
    "no condition". ``unit`` / ``tenant_contains`` read the sidecar metadata
    (``ingest.real_building.load_meta(id)["rooms"][room_id]["suite" | "tenant"]``);
    a room without that metadata does not match them. ``within_m_of`` is walking
    distance on the building graph, same floor.
    """

    model_config = ConfigDict(extra="forbid")

    room_ids: Optional[list[str]] = None
    floor_labels: Optional[list[str]] = None
    room_types: Optional[list[str]] = None
    name_contains: Optional[str] = None  # case-insensitive on Room.name
    unit: Optional[str] = None
    tenant_contains: Optional[str] = None  # case-insensitive
    has_windows: Optional[bool] = None
    window_facing: Optional[str] = None
    near_elevator: Optional[bool] = None
    near_stairs: Optional[bool] = None
    within_m_of: Optional[WithinM] = None


class Evidence(BaseModel):
    """One statement the posterior uses.

    * ``kind="vocab"``: wraps ``clue`` (scored by ``core.clue_likelihood.clue_match``).
    * ``kind="grounded"``: rooms satisfying ``selector`` get 1.0, every other room
      ``default_match``. With ``negate`` the satisfying rooms get 0.0 instead (a denial:
      "I'm not in 304" is ``negate=True, default_match=1.0``).

    Fused as ``[eps + (1 - eps) * m]^confidence`` (eps = 0.1), so no single statement
    can zero a room. Only ``status == "active"`` items count.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    clue: Optional[Clue] = None
    selector: Optional[RoomSelector] = None
    default_match: float = Field(default=0.0, ge=0.0, le=1.0)
    negate: bool = False  # beyond section 4: C2's record_location negate flag
    confidence: float = Field(ge=0.0, le=1.0)
    span: str = ""  # verbatim words from a caller turn
    turn: int = 0  # transcript index of the turn it came from
    source: EvidenceSource = "rules"
    rationale: str = ""
    status: EvidenceStatus = "active"
    # beyond section 4: filled by the session from core.evidence.evidence_match, for the UI
    matched_room_ids: list[str] = Field(default_factory=list)
    unmatched: bool = False  # True: the selector matched no room, the statement could not be placed


class Fact(BaseModel):
    """Something said that matters to the response but is not the caller's location.

    Shown with source and time, never as verified.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: FactCategory
    text: str
    span: str = ""
    turn: Optional[int] = None  # None for facts from the premise record
    source: FactSource = "rules"
    observed_at: Optional[datetime] = None
    room_id: Optional[str] = None
    verified: bool = False
    # beyond section 4: which premise-facts row this fills, if any
    premise_key: Optional[PremiseKey] = None


class TravelProfile(BaseModel):
    """Movement rates for this crew in this situation. Defaults are today's constants
    (``core.building_graph``: 1.2 m/s walking, 15 s per floor of stairs, no elevator)."""

    model_config = ConfigDict(extra="forbid")

    walk_mps: float = 1.2
    stair_s_per_floor: float = 15.0
    stair_s_per_floor_high: float = 15.0
    high_floor_from: Optional[int] = None  # floor above which the high rate applies; None = never
    search_advance_mps: Optional[float] = None  # None = area-based search time (clear air)
    elevator_allowed: bool = False
    elevator_fixed_s: Optional[float] = None  # None when no elevator option
    elevator_s_per_floor: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class Stake(BaseModel):
    """What a minute costs for this incident type, as a range. Never shown as zero."""

    model_config = ConfigDict(extra="forbid")

    label: str = "Time-critical"
    quantified: bool = False
    per_min_low: Optional[float] = None
    per_min_high: Optional[float] = None
    unit: Optional[str] = None  # e.g. "percentage points of survival per minute"
    origin: Optional[str] = None  # the time origin, in words
    note: Optional[str] = None


class Banner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: BannerLevel = "info"
    text: str
    source: str = ""


class Policy(BaseModel):
    """What the console may show for this situation. The server computes it; the UI only renders it."""

    model_config = ConfigDict(extra="forbid")

    route_enabled: bool = True
    elevator_allowed: bool = False
    show_hazard_class: bool = False
    show_incident_board: bool = False
    stake: Optional[Stake] = None
    banners: list[Banner] = Field(default_factory=list)
    travel: TravelProfile = Field(default_factory=TravelProfile)
    protocol_prompts: list[str] = Field(default_factory=list)


class Gates(BaseModel):
    """Rooms and entrances the officer has closed (body of POST /calls/demo/gates; plan_search ``gates``)."""

    model_config = ConfigDict(extra="forbid")

    closed_room_ids: list[str] = Field(default_factory=list)
    closed_entrance_ids: list[str] = Field(default_factory=list)


class SearchedRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str
    mode: SearchMode = "rapid"
    at: Optional[datetime] = None


class Question(BaseModel):
    """A question the dispatcher could ask the caller (DemoState.questions, top three).

    For ``kind="yes_no"``, ``clue_type`` + ``value`` are what the Yes / No / Not sure
    buttons send back in ``Answer``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str  # "<clue_type>:<value>" for yes/no, "open:<partition>" for open questions
    question: str  # the text to read out (same key as the older next_question dict)
    kind: QuestionKind = "yes_no"
    expected_info_gain_bits: float = 0.0
    clue_type: Optional[ClueType] = None
    value: Optional[str] = None
    p_yes: Optional[float] = None
    partition: Optional[str] = None  # open questions: "unit" | "tenant" | "floor"


class Answer(BaseModel):
    """The caller's reply to a yes/no question, sent by the Yes / No / Not sure buttons."""

    model_config = ConfigDict(extra="forbid")

    clue_type: ClueType
    value: str
    answer: AnswerValue


class AgentTraceEntry(BaseModel):
    """One step of the interpreter: a tool call, or a record it kept or dropped (and why)."""

    model_config = ConfigDict(extra="forbid")

    turn: int
    tool: str
    args: dict = Field(default_factory=dict)
    result_size: int = 0
    kept: bool = True
    reason: Optional[str] = None


class TranscriptTurn(BaseModel):
    """One turn of the call. Moved here from api/server.py so parser/ can read the transcript."""

    model_config = ConfigDict(extra="forbid")

    turn: int
    title: str
    speaker: Speaker
    text: str
    phone_fix: Optional[PhoneFix] = None
    clues: list[Clue] = Field(default_factory=list)  # vocab clues read from this turn (older clients)
    parser: Optional[str] = None  # caller turns: "rules" | "llm" | "agent" | "rules (<mode> unavailable: ...)"
    scripted: bool = True
    at: Optional[datetime] = None
    question: Optional[Question] = None  # dispatcher turn logged by POST /calls/demo/ask
    answer: Optional[Answer] = None  # caller turn sent with the Yes / No / Not sure buttons
    evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    """What ``parser.interpret.interpret_turn`` read from one transcript turn."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[Evidence] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    suggested_type: Optional[IncidentType] = None
    suggested_span: Optional[str] = None
    source: str = "rules"  # becomes TranscriptTurn.parser
    trace: list[AgentTraceEntry] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)  # ids of earlier evidence this turn corrects

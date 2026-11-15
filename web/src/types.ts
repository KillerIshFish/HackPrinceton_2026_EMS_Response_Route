// Mirrors core/schema.py (pydantic v2) and the DemoState JSON served by api/server.py.
// Tuples serialise as arrays; Building.floor_heights_m keys arrive as strings.

export type ClueType =
  | "room_type"
  | "floor"
  | "landmark"
  | "view"
  | "sound"
  | "smell"
  | "route"
  | "negative";

export type XY = [number, number];
export type LatLng = [number, number];

export interface Room {
  id: string;
  name: string;
  floor: number;
  floor_label: string;
  polygon: XY[];
  centroid: [number, number, number];
  area_m2: number;
  room_type: string;
  has_windows: boolean;
  window_facing: string | null;
  near_elevator: boolean;
  near_stairs: boolean;
  occupancy_factor: number;
}

export interface Edge {
  u: string;
  v: string;
  kind: "door" | "corridor" | "stairs" | "elevator";
  length_m: number;
  floors: number;
  /** where the door or opening is, when the source knows (scans and generated layouts) */
  xy?: XY;
  /** a measured stair: where it meets the lower and the upper floor */
  bottom_xy?: XY;
  top_xy?: XY;
}

export interface Entrance {
  id: string;
  name: string;
  room_id: string;
  xy: XY;
  latlng: LatLng;
  facing: "street" | "parking_lot" | string;
}

export interface StagingSpot {
  id: string;
  name: string;
  xy: XY;
  latlng: LatLng;
  kind: "street" | "parking_lot" | string;
  near_entrance_id: string;
}

export interface Building {
  id: string;
  name: string;
  use_type: string;
  floors: number[];
  floor_heights_m: Record<string, number>;
  rooms: Room[];
  edges: Edge[];
  entrances: Entrance[];
  staging_spots: StagingSpot[];
}

export interface Clue {
  type: ClueType;
  value: string;
  polarity: boolean;
  confidence: number;
  span: string;
}

export interface Posterior {
  /** keyed by room id; sums to 1 */
  room_probs: Record<string, number>;
  /** always 0: no mass is held back for outside the model (not displayed) */
  p_not_in_model: number;
  /** keyed by floor_label; sums to 1 */
  p_floor: Record<string, number>;
  credible_set_90: string[];
  entropy_bits: number;
  peak_count: number;
}

export interface Stop {
  room_id: string;
  room_name: string;
  floor_label: string;
  /** arrival time at the room, seconds from now */
  eta_s: number;
  cumulative_prob: number;
  // ---- Step 0 contract; filled by the planner from step A1 on (null until then) ----
  /** "re_search": a second (or later) look at a room already searched */
  kind: StopKind;
  /** search mode planned for this look */
  mode: SearchMode;
  /** probability this stop is the one that finds the caller */
  p_find: number | null;
  travel_s: number | null;
  search_s: number | null;
  /** fire_smoke only, display only: never changes the order */
  hazard_class: HazardClass | null;
}

export interface SearchPlan {
  teams: Stop[][];
  expected_time_to_find_s: number;
  baseline_nearest_s: number;
  baseline_highest_prob_s: number;
  entrance_id: string;
  staging_latlng: LatLng;
  // ---- Step 0 contract; filled by the planner from step A1 on (null until then) ----
  objective: PlanObjective;
  solver: Solver;
  /** always 0: no mass is held back for outside the model (not displayed) */
  p_out: number | null;
  /** probability the caller has been found by the end of this plan */
  cumulative_success: number | null;
  /** when the last first look in the plan finishes, seconds from now */
  time_to_cover_all_s: number | null;
  notes: string[];
}

export interface PhoneFix {
  x_m: number | null;
  y_m: number | null;
  horizontal_sigma_m: number | null;
  /** barometric height above the z = 0 datum */
  z_m: number | null;
  z_sigma_m: number;
}

export type Speaker = "system" | "caller" | "dispatcher";

export interface TranscriptTurn {
  turn: number;
  title: string;
  speaker: Speaker;
  text: string;
  phone_fix: PhoneFix | null;
  /** vocabulary clues read from this turn (DemoState.evidence has everything) */
  clues: Clue[];
  /** which interpreter read a caller turn: "rules", "llm", "agent" or "rules (<mode> unavailable: ...)" */
  parser?: string | null;
  /** false for utterances typed in by the dispatcher */
  scripted?: boolean;
  // ---- Step 0 contract ----
  /** ISO 8601 time the turn was logged */
  at: string | null;
  /** dispatcher turn logged by POST /calls/demo/ask: the question asked */
  question: Question | null;
  /** caller turn sent with the Yes / No / Not sure buttons */
  answer: Answer | null;
  evidence_ids: string[];
  fact_ids: string[];
}

export type ParserMode = "rules" | "llm" | "agent";

export interface BuildingSummary {
  id: string;
  name: string;
  real: boolean;
  floors: number;
  rooms: number;
}

export interface LLMStatus {
  available: boolean;
  model: string;
  usage: { calls?: number; input_tokens?: number; output_tokens?: number; cache_hits?: number; [k: string]: number | undefined };
}

/** GET /config */
export interface ClientConfig {
  matterport_sdk_key: string | null;
  llm_available: boolean;
  llm_model: string;
  vexcel_available: boolean;
}

/** one thing on the ground near the building, from OpenStreetMap */
export interface SiteFeature {
  class: "street" | "parking_lot" | "residential" | "adjacent_building";
  shape: "line" | "area";
  name: string;
  /** true for the map's own tracing of the modelled building */
  own: boolean;
  latlng: LatLng[];
}

/** roads, car parks, neighbouring buildings and named places around the building */
export interface SiteSurroundings {
  source: string;
  features: SiteFeature[];
  places?: { name: string; kind: string; latlng: LatLng }[];
  counts: Record<string, number>;
}

export interface SiteTiles {
  layer: string;
  zoom: number;
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  tile_px: number;
  m_per_px: number;
}

export interface RoomMeta {
  source: "matterport" | "costar" | "inferred" | "floor_plan";
  suite?: string;
  status?: string;
  tenant?: string | null;
  rentable_sf?: number;
  model_id?: string;
  sweep_id?: string | null;
  windows_source?: string;
  note?: string;
}

/** Matterport's plan image of one floor, placed in the same coordinates as the rooms */
export interface FloorPlanImage {
  width: number;
  height: number;
  /** pixels per metre */
  resolution: number;
  /** building-local (x, y) of the image's lower-left corner */
  origin: XY;
}

export interface MatterportModel {
  model_id: string;
  /** suite the scan covers; null when the scan covers the whole building */
  suite: string | null;
  name: string;
  floor: number | null;
  created?: string;
  sweeps?: number;
  has_room_boundaries?: boolean;
  /** false when the scan's owner switched that overview mode off in Matterport; null/undefined = unknown */
  dollhouse_enabled?: boolean | null;
  floorplan_enabled?: boolean | null;
}

/** GET /lookup - one property matching an address, a name or an id */
export interface PropertyHit {
  property_id: number;
  name: string;
  address: string;
  city: string | null;
  state: string | null;
  stories: number | null;
  units: number | null;
}

/** POST /buildings/generate - what the pipeline built and how long each stage took */
export interface GenerateReport {
  building_id: string;
  name: string;
  tier: string;
  archetype: string;
  total_ms: number;
  stages: { stage: string; ms: number; detail: string }[];
  notes: string[];
}

/** GET /buildings/{id}/model3d - the unified building model, ready to draw */
export interface Model3DPayload {
  id: string;
  name: string;
  tier: string;
  archetype: string;
  /** where the geometry came from, in words */
  source?: string;
  notes: string[];
  levels: { i: number; label: string; z0: number; h: number }[];
  /** l level, k kind, n label, f fidelity, p polygon, par parent space, r the app room it counts toward */
  spaces: { id: string; l: number; k: string; n: string; f: string; p: [number, number][]; par: string | null; r: string }[];
  /** p solid panels and g window glass as [t0, t1, z0, z1]; t is a fraction along a-b */
  walls: { l: number; a: [number, number]; b: [number, number]; th: number; x: boolean; f: string; p: [number, number, number, number][]; g: [number, number, number, number][] }[];
  /** doors to the outside. xy on level l; o: unit vector pointing out of the building at the door; r: the app room it opens into */
  entrances?: { id: string; n: string; xy: [number, number]; l: number; o?: [number, number]; r?: string }[];
  bounds: { x0: number; y0: number; x1: number; y1: number; z1: number };
  counts: { spaces: number; walls: number; fidelity: Record<string, number> };
}

/** GET /buildings/{id}/meta - provenance, scans and the aerial site annotation */
export interface BuildingMeta {
  building_id: string;
  real: boolean;
  name?: string;
  address?: string;
  center_latlng?: LatLng;
  site: {
    /** "generated" | "scan" for buildings made at call time; absent for hand-annotated sites */
    source?: string;
    footprint_latlng: LatLng[];
    /** what to outline on the aerial: the real building (OpenStreetMap) and the model's own extent */
    outlines?: { kind: string; source?: string; latlng: LatLng[] }[];
    /** what fixed the model's position and orientation on the map, in words */
    placement_source?: string | null;
    /** false when the layout's positions on the map are invented (several buildings on one property):
     *  only the address point and the real outline there are drawn */
    layout_placed?: boolean;
    address_latlng?: LatLng | null;
    /** number of buildings on the property */
    buildings?: number;
    /** true while the OpenStreetMap lookup that refines the placement is still running */
    pending?: boolean;
    /** what stands around the building; drawn as the site map and used for window views */
    surroundings?: SiteSurroundings | null;
    tiles?: SiteTiles;
  };
  matterport_models: MatterportModel[];
  /** "full_scan" when one Matterport scan covers the whole building; "floor_plan" when it was traced from published plans */
  kind?: string;
  /** rotation_deg: counter-clockwise angle of the floor plan's +x axis from east */
  georeference?: { rotation_deg: number };
  /** keyed by floor number; only for buildings built from a whole-building scan */
  floorplans?: Record<string, FloorPlanImage>;
  room_sources?: Record<string, number>;
  rooms: Record<string, RoomMeta>;
  caveats: string[];
}

export interface Approach {
  entrance_id: string;
  entrance_name: string;
  entrance_latlng: LatLng;
  staging_id: string;
  staging_name: string;
  staging_latlng: LatLng;
  stairwell_id: string | null;
  stairwell_name: string | null;
  target_room_id: string | null;
  target_floor_label: string | null;
  entrance_expected_travel_s: Record<string, number>;
  // ---- Step 0 contract: keys step B1 adds to recommend_approach (present in the fixtures) ----
  /** stair time from the entrance floor to the target floor */
  stairs_s_to_target_floor?: number | null;
  /** elevator time to the target floor; null/absent when no elevator option (fire, smoke) */
  elevator_s_to_target_floor?: number | null;
  /** walk from the staging spot to each entrance, by entrance id */
  entrance_access_s?: Record<string, number>;
  /** barrier warnings per entrance id ("No key holder recorded for this entrance") */
  entrance_warnings?: Record<string, string[]>;
  /** whole-approach warnings (e.g. faster than measured curb-to-patient intervals) */
  warnings?: string[];
  /** seconds credited when someone meets the crew (Situation.meet_person set); 0 otherwise */
  meet_person_credit_s?: number;
  /** predicted curb (staging spot) to the most likely room, seconds, less the meet-person credit; null when unknown or the route is off */
  predicted_curb_to_target_s?: number | null;
}

export interface NextQuestion {
  question: string;
  clue_type: string;
  value: string;
  expected_info_gain_bits: number;
  /** prior-predictive probability the caller answers yes (optional extra from the backend) */
  p_yes?: number;
}

export interface HistoryEntry {
  turn: number;
  entropy_bits: number;
  expected_time_to_find_s: number;
  top_room_id: string;
  top_room_prob: number;
}

// ---------------------------------------------------------------------------
// Step 0 contract (docs/BUILD_PLAN.md section 4). Mirrors core/schema.py and api/session.py.
// Datetimes are ISO 8601 strings. Optional Python fields arrive as null.
// ---------------------------------------------------------------------------

export type IncidentType =
  | "unknown"
  | "cardiac_arrest"
  | "stroke"
  | "internal_bleeding"
  | "external_bleeding"
  | "overdose"
  | "fire_smoke"
  | "active_threat"
  | "other_medical";

export const INCIDENT_TYPES: IncidentType[] = [
  "unknown",
  "cardiac_arrest",
  "stroke",
  "internal_bleeding",
  "external_bleeding",
  "overdose",
  "fire_smoke",
  "active_threat",
  "other_medical",
];

export type CrewLoad = "light" | "heavy";
export type SearchMode = "rapid" | "thorough";
export type StopKind = "first" | "re_search";
export type HazardClass = "fire_room" | "open_path" | "closed_door";
export type EvidenceKind = "vocab" | "grounded";
export type EvidenceSource = "rules" | "llm" | "agent" | "dispatcher";
export type EvidenceStatus = "active" | "rejected" | "superseded";
export type FactCategory = "hazard" | "access" | "patient" | "people" | "threat" | "resource";
/** facts can also come from the premise record */
export type FactSource = EvidenceSource | "record";
export type PremiseKey = "means_of_entry" | "meet_person" | "elevator" | "defibrillator";
export type TimestampName = "wheels_stopped" | "building_entry" | "floor_arrival" | "patient_contact";
export const TIMESTAMP_NAMES: TimestampName[] = ["wheels_stopped", "building_entry", "floor_arrival", "patient_contact"];
export type BannerLevel = "info" | "warning" | "critical";
export type QuestionKind = "yes_no" | "open";
export type AnswerValue = "yes" | "no" | "unsure";
export type PlanObjective = "expected_time" | "survival";
export type Solver = "exact" | "heuristic";

/** Set by the dispatcher. The interpreter only suggests a type (suggested_type + suggested_span). */
export interface Situation {
  incident_type: IncidentType;
  incident_confirmed: boolean;
  suggested_type: IncidentType | null;
  /** the caller's words behind the suggestion (verbatim) */
  suggested_span: string | null;
  /** types the dispatcher dismissed; not suggested again */
  dismissed_types: IncidentType[];
  /** null = not known */
  witnessed: boolean | null;
  /** null = not known */
  cpr_in_progress: boolean | null;
  smoke_reported: boolean;
  load: CrewLoad;
  /** ISO 8601; the call clock counts up from here */
  call_started_at: string | null;
  /** officer gates only */
  closed_room_ids: string[];
  closed_entrance_ids: string[];
  /** who will meet the crew, in words; null = nobody recorded */
  meet_person: string | null;
  timestamps: Partial<Record<TimestampName, string>>;
}

export interface WithinM {
  room_id: string;
  radius_m: number;
}

/** Rooms a statement points at. Set conditions AND together; null = no condition. */
export interface RoomSelector {
  room_ids: string[] | null;
  floor_labels: string[] | null;
  room_types: string[] | null;
  name_contains: string | null;
  unit: string | null;
  tenant_contains: string | null;
  has_windows: boolean | null;
  window_facing: string | null;
  near_elevator: boolean | null;
  near_stairs: boolean | null;
  within_m_of: WithinM | null;
}

/** One statement the posterior uses. Only status "active" counts. */
export interface Evidence {
  id: string;
  kind: EvidenceKind;
  /** kind "vocab" */
  clue: Clue | null;
  /** kind "grounded" */
  selector: RoomSelector | null;
  /** match for rooms outside the selector */
  default_match: number;
  /** a denial: selector rooms score 0 instead of 1 */
  negate: boolean;
  confidence: number;
  /** verbatim words from a caller turn */
  span: string;
  /** transcript index of the turn it came from */
  turn: number;
  source: EvidenceSource;
  rationale: string;
  status: EvidenceStatus;
  /** rooms the statement points at (for outlining on the map) */
  matched_room_ids: string[];
  /** true: the statement could not be placed in this building; it changes nothing */
  unmatched: boolean;
}

/** Something said that matters to the response but is not the caller's location. Never verified. */
export interface Fact {
  id: string;
  category: FactCategory;
  text: string;
  span: string;
  /** null for facts from the premise record */
  turn: number | null;
  source: FactSource;
  /** ISO 8601 */
  observed_at: string | null;
  room_id: string | null;
  verified: boolean;
  /** which premise-facts row this fills, if any */
  premise_key: PremiseKey | null;
}

export interface TravelProfile {
  walk_mps: number;
  stair_s_per_floor: number;
  stair_s_per_floor_high: number;
  high_floor_from: number | null;
  /** null = area-based search time (clear air) */
  search_advance_mps: number | null;
  elevator_allowed: boolean;
  elevator_fixed_s: number | null;
  elevator_s_per_floor: number | null;
  notes: string[];
}

/** What a minute costs. Never shown as zero. When quantified is false show the label alone. */
export interface Stake {
  label: string;
  quantified: boolean;
  per_min_low: number | null;
  per_min_high: number | null;
  unit: string | null;
  /** the time origin, in words */
  origin: string | null;
  note: string | null;
}

export interface Banner {
  level: BannerLevel;
  text: string;
  source: string;
}

/** What the console may show. Computed by the server; the UI only renders it. */
export interface Policy {
  route_enabled: boolean;
  elevator_allowed: boolean;
  show_hazard_class: boolean;
  show_incident_board: boolean;
  stake: Stake | null;
  banners: Banner[];
  travel: TravelProfile;
  protocol_prompts: string[];
}

/** Body of POST /calls/demo/gates (replaces both lists) */
export interface Gates {
  closed_room_ids: string[];
  closed_entrance_ids: string[];
}

export interface SearchedRoom {
  room_id: string;
  mode: SearchMode;
  /** ISO 8601 */
  at: string | null;
}

/** A question the dispatcher could ask (DemoState.questions, top three). */
export interface Question {
  /** "<clue_type>:<value>" for yes/no, "open:<partition>" for open questions */
  id: string;
  /** the text to read out */
  question: string;
  kind: QuestionKind;
  expected_info_gain_bits: number;
  /** yes/no: what the Yes / No / Not sure buttons send back */
  clue_type: ClueType | null;
  value: string | null;
  p_yes: number | null;
  /** open questions: "unit" | "tenant" | "floor" */
  partition: string | null;
}

/** The caller's reply to a yes/no question (UtteranceBody.answers) */
export interface Answer {
  clue_type: ClueType;
  value: string;
  answer: AnswerValue;
}

/** One interpreter step: a tool call, or a record kept or dropped. */
export interface AgentTraceEntry {
  turn: number;
  tool: string;
  args: Record<string, unknown>;
  result_size: number;
  kept: boolean;
  /** why it was dropped */
  reason: string | null;
}

export interface ScenarioSummary {
  id: string;
  title: string;
  building_id: string;
  /** what the call turns out to be; the scripted call itself starts with the type unset */
  incident_type: IncidentType;
  description: string;
  /** where the scripted caller actually is (for scoring; not a console hint) */
  true_room_id?: string | null;
  /** number of scripted turns */
  turns?: number;
}

/** Body of POST /calls/demo/utterance */
export interface UtteranceBody {
  text: string;
  speaker?: "caller" | "dispatcher";
  answers?: Answer | null;
}

/** Body of POST /calls/demo/ask (a Question from state.questions can be posted as is) */
export interface AskBody {
  question: string;
  id?: string | null;
  kind?: QuestionKind | null;
  clue_type?: string | null;
  value?: string | null;
  partition?: string | null;
}

/** Body of POST /calls/demo/situation: only the fields sent change.
 *  Choosing incident_type without incident_confirmed confirms it. */
export interface SituationPatch {
  incident_type?: IncidentType;
  incident_confirmed?: boolean;
  /** drop the interpreter's pending suggestion and stop it coming back */
  dismiss_suggestion?: boolean;
  witnessed?: boolean | null;
  cpr_in_progress?: boolean | null;
  smoke_reported?: boolean;
  load?: CrewLoad;
  call_started_at?: string;
  meet_person?: string | null;
  n_teams?: number;
}

export interface DemoState {
  building_id: string;
  /** index of the latest revealed turn (0-based) */
  turn: number;
  total_turns: number;
  done: boolean;
  transcript: TranscriptTurn[];
  clues: Clue[];
  phone_fix: PhoneFix | null;
  cleared_room_ids: string[];
  n_teams: number;
  posterior: Posterior;
  plan: SearchPlan;
  approach: Approach;
  next_question: NextQuestion | null;
  history: HistoryEntry[];
  parser_mode: ParserMode;
  llm: LLMStatus | null;
  buildings: BuildingSummary[];
  // ---- Step 0 contract (section 4) ----
  situation: Situation;
  policy: Policy;
  /** every evidence item, whatever its status; only "active" ones move the posterior */
  evidence: Evidence[];
  facts: Fact[];
  /** top three questions; none during an active threat */
  questions: Question[];
  /** every look, oldest first (a room can appear more than once) */
  searched: SearchedRoom[];
  found_room_id: string | null;
  /** server time (ISO) the dispatcher recorded found_room_id; absent in older recordings */
  found_at?: string | null;
  scenarios: ScenarioSummary[];
  agent_trace: AgentTraceEntry[];
  // ---- beyond section 4 ----
  scenario_id: string | null;
  /** hour of day the prior was computed for */
  prior_hour: number;
  /** an undoable searched / found / gate action exists */
  can_undo: boolean;
}

// Mirrors core/schema.py (pydantic v2) and the DemoState JSON served by api/server.py.
// Tuples serialise as arrays; Building.floor_heights_m keys arrive as strings.

export type ClueType =
  | "room_type"
  | "floor"
  | "landmark"
  | "view"
  | "sound"
  | "smell"
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
  /** keyed by room id; sum(room_probs) + p_not_in_model == 1 */
  room_probs: Record<string, number>;
  p_not_in_model: number;
  /** keyed by floor_label; sums to 1 - p_not_in_model */
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
}

export interface SearchPlan {
  teams: Stop[][];
  expected_time_to_find_s: number;
  baseline_nearest_s: number;
  baseline_highest_prob_s: number;
  entrance_id: string;
  staging_latlng: LatLng;
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
  clues: Clue[];
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
}

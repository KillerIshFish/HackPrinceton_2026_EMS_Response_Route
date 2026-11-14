// Fixture backend for `?fixture=<name>`: the console runs on a recorded DemoState with no server.
//
// The four states were captured from the real backend on the synthetic building (ptc-100) and then
// given plausible values for the Step 0 fields (see docs/BUILD_PLAN.md section 4):
//   unknown         the headline demo call, a pending (unconfirmed) incident-type suggestion
//   cardiac_arrest  agent mode: correction (superseded), answered yes/no question, rejected clue,
//                   a searched room offered again (re_search), quantified stake, premise facts, trace
//   fire_smoke      smoke rates + heavy load notes, hazard classes, an officer gate, rejected clues
//   active_threat   route_enabled=false: no plan stops, no entrance, incident board facts, no questions
//
// Mutations update the recorded state locally (transcript, situation, evidence status, searched,
// found, gates, undo, timestamps, teams) so controls can be exercised. The posterior and plan do
// NOT change: only the backend computes those.
//
// Timestamps are shifted on load so the call clock reads as it did when the state was captured.

import type {
  AskBody,
  Building,
  BuildingMeta,
  ClientConfig,
  ClueType,
  DemoState,
  Gates,
  IncidentType,
  Model3DPayload,
  ParserMode,
  ScenarioSummary,
  SearchMode,
  SituationPatch,
  TimestampName,
  TranscriptTurn,
  UtteranceBody,
} from "../types";

export const FIXTURE_NAMES = ["unknown", "cardiac_arrest", "fire_smoke", "active_threat"] as const;
export type FixtureName = (typeof FIXTURE_NAMES)[number];

const FIXTURE_TITLES: Record<FixtureName, string> = {
  unknown: "Fixture: incident type not set",
  cardiac_arrest: "Fixture: cardiac arrest",
  fire_smoke: "Fixture: smoke on the third floor",
  active_threat: "Fixture: active threat",
};

const loaders = import.meta.glob<unknown>("./*.json", { import: "default" });

async function loadJson<T>(file: string): Promise<T> {
  const load = loaders[`./${file}.json`];
  if (!load) throw new Error(`No fixture file ${file}.json`);
  return structuredClone((await load()) as T);
}

const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;

/** Shift every ISO timestamp at or after `from` by `deltaMs` (premise-record dates stay as they are). */
function shiftTimes(value: unknown, from: number, deltaMs: number): unknown {
  if (typeof value === "string" && ISO.test(value)) {
    const t = Date.parse(value);
    return t >= from ? new Date(t + deltaMs).toISOString() : value;
  }
  if (Array.isArray(value)) return value.map((v) => shiftTimes(v, from, deltaMs));
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = shiftTimes(v, from, deltaMs);
    return out;
  }
  return value;
}

/** The moment the state was captured: its latest timestamp. */
function capturedAt(s: DemoState): number {
  const times = [
    ...s.transcript.map((t) => t.at),
    ...s.searched.map((r) => r.at),
    ...Object.values(s.situation.timestamps),
    s.situation.call_started_at,
  ]
    .filter((t): t is string => typeof t === "string")
    .map((t) => Date.parse(t));
  return Math.max(...times);
}

function isFixtureName(name: string): name is FixtureName {
  return (FIXTURE_NAMES as readonly string[]).includes(name);
}

const nowIso = () => new Date().toISOString();

type Snapshot = Pick<DemoState, "searched" | "found_room_id" | "cleared_room_ids"> & Gates;

export class FixtureBackend {
  name: FixtureName;
  state!: DemoState;
  private undoStack: Snapshot[] = [];

  constructor(name: FixtureName) {
    this.name = name;
  }

  async load(name: FixtureName = this.name): Promise<DemoState> {
    this.name = name;
    const raw = await loadJson<DemoState>(name);
    const start = Date.parse(raw.situation.call_started_at ?? "") || capturedAt(raw);
    const delta = Date.now() - capturedAt(raw);
    this.state = shiftTimes(raw, start, delta) as DemoState;
    this.state.scenarios = FIXTURE_NAMES.map(
      (id): ScenarioSummary => ({
        id,
        title: FIXTURE_TITLES[id],
        building_id: this.state.building_id,
        incident_type: id as IncidentType,
        description: "Recorded state (fixture mode)",
      }),
    );
    this.state.scenario_id = name;
    this.undoStack = [];
    return this.snapshot();
  }

  snapshot(): DemoState {
    return structuredClone(this.state);
  }

  private pushUndo(): void {
    const s = this.state;
    this.undoStack.push(
      structuredClone({
        searched: s.searched,
        found_room_id: s.found_room_id,
        cleared_room_ids: s.cleared_room_ids,
        closed_room_ids: s.situation.closed_room_ids,
        closed_entrance_ids: s.situation.closed_entrance_ids,
      }),
    );
    s.can_undo = true;
  }

  private addTurn(turn: Omit<TranscriptTurn, "turn">): DemoState {
    this.state.transcript.push({ ...turn, turn: this.state.transcript.length });
    return this.snapshot();
  }

  say(body: UtteranceBody): DemoState {
    const speaker = body.speaker ?? "caller";
    return this.addTurn({
      title: speaker === "caller" ? "Caller (live)" : "Dispatcher (live)",
      speaker,
      text: body.text,
      phone_fix: null,
      clues: [],
      parser: speaker === "caller" ? "fixture (not interpreted)" : null,
      scripted: false,
      at: nowIso(),
      question: null,
      answer: body.answers ?? null,
      evidence_ids: [],
      fact_ids: [],
    });
  }

  ask(body: AskBody): DemoState {
    const kind = body.kind ?? (body.clue_type ? "yes_no" : "open");
    const id = body.id ?? (body.clue_type ? `${body.clue_type}:${body.value ?? ""}` : `open:${body.question}`);
    this.state.questions = this.state.questions.filter((q) => q.id !== id);
    return this.addTurn({
      title: "Dispatcher: question",
      speaker: "dispatcher",
      text: body.question,
      phone_fix: null,
      clues: [],
      parser: null,
      scripted: false,
      at: nowIso(),
      question: {
        id,
        question: body.question,
        kind,
        expected_info_gain_bits: 0,
        clue_type: (body.clue_type ?? null) as ClueType | null,
        value: body.value ?? null,
        p_yes: null,
        partition: body.partition ?? null,
      },
      answer: null,
      evidence_ids: [],
      fact_ids: [],
    });
  }

  setSituation(patch: SituationPatch): DemoState {
    const sit = this.state.situation;
    if (patch.dismiss_suggestion && sit.suggested_type) {
      if (!sit.dismissed_types.includes(sit.suggested_type)) sit.dismissed_types.push(sit.suggested_type);
      sit.suggested_type = null;
      sit.suggested_span = null;
    }
    if (patch.n_teams !== undefined) this.state.n_teams = patch.n_teams;
    if (patch.incident_type !== undefined) {
      sit.incident_type = patch.incident_type;
      sit.incident_confirmed = patch.incident_confirmed ?? true;
      if (sit.suggested_type === patch.incident_type) {
        sit.suggested_type = null;
        sit.suggested_span = null;
      }
    } else if (patch.incident_confirmed !== undefined) {
      sit.incident_confirmed = patch.incident_confirmed;
    }
    if (patch.witnessed !== undefined) sit.witnessed = patch.witnessed;
    if (patch.cpr_in_progress !== undefined) sit.cpr_in_progress = patch.cpr_in_progress;
    if (patch.smoke_reported !== undefined) sit.smoke_reported = patch.smoke_reported;
    if (patch.load !== undefined) sit.load = patch.load;
    if (patch.call_started_at !== undefined) sit.call_started_at = patch.call_started_at;
    if (patch.meet_person !== undefined) sit.meet_person = patch.meet_person;
    return this.snapshot();
  }

  setEvidenceStatus(id: string, status: "active" | "rejected"): DemoState {
    const item = this.state.evidence.find((e) => e.id === id);
    if (!item) throw new Error(`Unknown evidence '${id}'`);
    item.status = status;
    return this.snapshot();
  }

  searched(roomId: string, mode: SearchMode, onlyIfNew = false): DemoState {
    const s = this.state;
    if (onlyIfNew && s.searched.some((r) => r.room_id === roomId)) return this.snapshot();
    this.pushUndo();
    s.searched.push({ room_id: roomId, mode, at: nowIso() });
    if (!s.cleared_room_ids.includes(roomId)) s.cleared_room_ids.push(roomId);
    return this.snapshot();
  }

  found(roomId: string): DemoState {
    this.pushUndo();
    this.state.found_room_id = roomId;
    return this.snapshot();
  }

  setGates(gates: Gates): DemoState {
    this.pushUndo();
    this.state.situation.closed_room_ids = [...new Set(gates.closed_room_ids)];
    this.state.situation.closed_entrance_ids = [...new Set(gates.closed_entrance_ids)];
    return this.snapshot();
  }

  undo(): DemoState {
    const last = this.undoStack.pop();
    if (last) {
      const s = this.state;
      s.searched = last.searched;
      s.found_room_id = last.found_room_id;
      s.cleared_room_ids = last.cleared_room_ids;
      s.situation.closed_room_ids = last.closed_room_ids;
      s.situation.closed_entrance_ids = last.closed_entrance_ids;
    }
    this.state.can_undo = this.undoStack.length > 0;
    return this.snapshot();
  }

  timestamp(name: TimestampName): DemoState {
    this.state.situation.timestamps[name] = nowIso();
    return this.snapshot();
  }

  setTeams(n: number): DemoState {
    this.state.n_teams = n;
    return this.snapshot();
  }

  setParser(mode: ParserMode): DemoState {
    this.state.parser_mode = mode;
    return this.snapshot();
  }

  /** A scenario id that names another fixture switches to it; anything else keeps the current one. */
  async scenario(id: string): Promise<DemoState> {
    return isFixtureName(id) ? this.load(id) : this.snapshot();
  }

  building = () => loadJson<Building>("building");
  meta = () => loadJson<BuildingMeta>("meta");
  model3d = () => loadJson<Model3DPayload>("model3d");
  config = (): ClientConfig => ({
    matterport_sdk_key: null,
    llm_available: false,
    llm_model: "fixture",
    vexcel_available: false,
  });
}

/** Load the named fixture (unknown names fall back to "unknown"). */
export async function createFixtureBackend(name: string): Promise<FixtureBackend> {
  const backend = new FixtureBackend(isFixtureName(name) ? name : "unknown");
  await backend.load();
  return backend;
}

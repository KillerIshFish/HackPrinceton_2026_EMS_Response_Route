import type {
  Answer,
  AskBody,
  Building,
  BuildingMeta,
  ClientConfig,
  DemoState,
  Gates,
  GenerateReport,
  Model3DPayload,
  Posterior,
  PropertyHit,
  SearchMode,
  SearchPlan,
  SituationPatch,
  TimestampName,
  UtteranceBody,
} from "./types";
import type { FixtureBackend } from "./fixtures";

const BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "/api";

/**
 * `?fixture=<name>` in the URL (unknown | cardiac_arrest | fire_smoke | active_threat) runs the
 * console on a recorded DemoState instead of the backend. See src/fixtures/index.ts.
 */
export const FIXTURE: string | null =
  typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("fixture");

let fixtureBackend: Promise<FixtureBackend> | null = null;
function fixture(): Promise<FixtureBackend> {
  fixtureBackend ??= import("./fixtures").then((m) => m.createFixtureBackend(FIXTURE ?? "unknown"));
  return fixtureBackend;
}

export class ApiError extends Error {
  status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError("Backend unreachable (network error).", null);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body, e.g. the Vite proxy's ECONNREFUSED page */
    }
    // The dev proxy answers 500/502/504 when nothing is listening on :8000.
    const unreachable = !detail && [500, 502, 503, 504].includes(res.status);
    throw new ApiError(
      unreachable
        ? `Backend unreachable (HTTP ${res.status} from proxy).`
        : `${init?.method ?? "GET"} ${path} failed: HTTP ${res.status}${detail ? ` - ${detail}` : ""}`,
      res.status,
    );
  }
  return (await res.json()) as T;
}

const post = <T,>(path: string) => request<T>(path, { method: "POST" });

const postJson = <T,>(path: string, body: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/** Live call, or the fixture backend's answer when `?fixture=` is set. */
function live<T>(call: () => Promise<T>, onFixture: (b: FixtureBackend) => T | Promise<T>): Promise<T> {
  if (FIXTURE === null) return call();
  return fixture().then((b) => onFixture(b));
}

const enc = encodeURIComponent;

export const api = {
  config: () => live(() => request<ClientConfig>("/config"), (b) => b.config()),
  buildings: () => live(() => request<Building[]>("/buildings"), async (b) => [await b.building()]),
  building: (id: string) => live(() => request<Building>(`/buildings/${enc(id)}`), (b) => b.building()),
  buildingMeta: (id: string) => live(() => request<BuildingMeta>(`/buildings/${enc(id)}/meta`), (b) => b.meta()),
  /** properties matching an address, a building name or a property id */
  lookup: (q: string) => live(() => request<PropertyHit[]>(`/lookup?q=${enc(q)}`), () => []),
  /** generate a building model now from an address, a property id or "mp:<model id>" */
  generate: (q: string) =>
    live(
      () => post<GenerateReport>(`/buildings/generate?q=${enc(q)}`),
      () => Promise.reject(new ApiError("Generating a building is not available in fixture mode.", null)),
    ),
  model3d: (id: string) => live(() => request<Model3DPayload>(`/buildings/${enc(id)}/model3d`), (b) => b.model3d()),
  /** URL of Matterport's plan image for one floor (whole-building scans only) */
  floorplanUrl: (id: string, floor: number) => `${BASE}/buildings/${enc(id)}/floorplan/${floor}`,
  /** URL of one aerial tile, served (and cached) by the backend so the Vexcel key stays there */
  tileUrl: (layer: string, z: number, x: number, y: number) => `${BASE}/vexcel/tile/${layer}/${z}/${x}/${y}`,

  // ---- the demo call (every call returns the full DemoState) ----
  state: () => live(() => request<DemoState>("/calls/demo/state"), (b) => b.snapshot()),
  step: () => live(() => post<DemoState>("/calls/demo/step"), (b) => b.snapshot()),
  reset: () => live(() => post<DemoState>("/calls/demo/reset"), (b) => b.load()),
  setBuilding: (id: string) => live(() => post<DemoState>(`/calls/demo/building?id=${enc(id)}`), (b) => b.snapshot()),
  setTeams: (n: number) => live(() => post<DemoState>(`/calls/demo/teams?n=${n}`), (b) => b.setTeams(n)),
  /** A caller turn (default) or a dispatcher turn. `answers` is set by the Yes / No / Not sure buttons. */
  say: (text: string, speaker: "caller" | "dispatcher" = "caller", answers: Answer | null = null) => {
    const body: UtteranceBody = { text, speaker, answers };
    return live(() => postJson<DemoState>("/calls/demo/utterance", body), (b) => b.say(body));
  },
  /** Log a question the dispatcher asked (a Question from state.questions can be passed as is). */
  ask: (question: AskBody) => live(() => postJson<DemoState>("/calls/demo/ask", question), (b) => b.ask(question)),
  /** Patch the situation: incident type (confirms it), dismiss_suggestion, smoke, load, teams, ... */
  setSituation: (patch: SituationPatch) =>
    live(() => postJson<DemoState>("/calls/demo/situation", patch), (b) => b.setSituation(patch)),
  rejectEvidence: (id: string) =>
    live(() => post<DemoState>(`/calls/demo/evidence/${enc(id)}/reject`), (b) => b.setEvidenceStatus(id, "rejected")),
  restoreEvidence: (id: string) =>
    live(() => post<DemoState>(`/calls/demo/evidence/${enc(id)}/restore`), (b) => b.setEvidenceStatus(id, "active")),
  /** Rooms and entrances closed on the officer's word (replaces both lists). */
  setGates: (gates: Gates) => live(() => postJson<DemoState>("/calls/demo/gates", gates), (b) => b.setGates(gates)),
  /** Undo the last searched, found or gate action. */
  undo: () => live(() => post<DemoState>("/calls/demo/undo"), (b) => b.undo()),
  timestamp: (name: TimestampName) =>
    live(() => post<DemoState>(`/calls/demo/timestamp?name=${name}`), (b) => b.timestamp(name)),
  /** Start a scripted scenario from state.scenarios. */
  scenario: (id: string) => live(() => post<DemoState>(`/calls/demo/scenario?id=${enc(id)}`), (b) => b.scenario(id)),

  // ---- search outcomes ----
  searched: (roomId: string, mode: SearchMode = "rapid") =>
    live(() => post<DemoState>(`/rooms/${enc(roomId)}/searched?mode=${mode}`), (b) => b.searched(roomId, mode)),
  found: (roomId: string) => live(() => post<DemoState>(`/rooms/${enc(roomId)}/found`), (b) => b.found(roomId)),
  /** pre-Step-0 name: a rapid search recorded once per room */
  clearRoom: (roomId: string) =>
    live(() => post<DemoState>(`/rooms/${enc(roomId)}/clear`), (b) => b.searched(roomId, "rapid", true)),

  posterior: () => live(() => request<Posterior>("/posterior"), (b) => b.snapshot().posterior),
  plan: (teams = 1) => live(() => request<SearchPlan>(`/plan?teams=${teams}`), (b) => b.snapshot().plan),
};

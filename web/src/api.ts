import type {
  Building,
  BuildingMeta,
  ClientConfig,
  DemoState,
  GenerateReport,
  Model3DPayload,
  ParserMode,
  Posterior,
  PropertyHit,
  SearchPlan,
} from "./types";

const BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "/api";

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

export const api = {
  config: () => request<ClientConfig>("/config"),
  buildings: () => request<Building[]>("/buildings"),
  building: (id: string) => request<Building>(`/buildings/${encodeURIComponent(id)}`),
  buildingMeta: (id: string) => request<BuildingMeta>(`/buildings/${encodeURIComponent(id)}/meta`),
  /** properties matching an address, a building name or a property id */
  lookup: (q: string) => request<PropertyHit[]>(`/lookup?q=${encodeURIComponent(q)}`),
  /** generate a building model now from an address, a property id or "mp:<model id>" */
  generate: (q: string) => post<GenerateReport>(`/buildings/generate?q=${encodeURIComponent(q)}`),
  model3d: (id: string) => request<Model3DPayload>(`/buildings/${encodeURIComponent(id)}/model3d`),
  /** URL of Matterport's plan image for one floor (whole-building scans only) */
  floorplanUrl: (id: string, floor: number) => `${BASE}/buildings/${encodeURIComponent(id)}/floorplan/${floor}`,
  /** URL of one aerial tile, served (and cached) by the backend so the Vexcel key stays there */
  tileUrl: (layer: string, z: number, x: number, y: number) => `${BASE}/vexcel/tile/${layer}/${z}/${x}/${y}`,
  setBuilding: (id: string) => post<DemoState>(`/calls/demo/building?id=${encodeURIComponent(id)}`),
  setParser: (mode: ParserMode) => post<DemoState>(`/calls/demo/parser?mode=${mode}`),
  say: (text: string) =>
    request<DemoState>("/calls/demo/utterance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  state: () => request<DemoState>("/calls/demo/state"),
  step: () => post<DemoState>("/calls/demo/step"),
  reset: () => post<DemoState>("/calls/demo/reset"),
  setTeams: (n: number) => post<DemoState>(`/calls/demo/teams?n=${n}`),
  clearRoom: (roomId: string) => post<DemoState>(`/rooms/${encodeURIComponent(roomId)}/clear`),
  posterior: () => request<Posterior>("/posterior"),
  plan: (teams = 1) => request<SearchPlan>(`/plan?teams=${teams}`),
};

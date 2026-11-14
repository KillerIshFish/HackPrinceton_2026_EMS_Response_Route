import type { Building, DemoState, Posterior, SearchPlan } from "./types";

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
  buildings: () => request<Building[]>("/buildings"),
  building: (id: string) => request<Building>(`/buildings/${encodeURIComponent(id)}`),
  state: () => request<DemoState>("/calls/demo/state"),
  step: () => post<DemoState>("/calls/demo/step"),
  reset: () => post<DemoState>("/calls/demo/reset"),
  setTeams: (n: number) => post<DemoState>(`/calls/demo/teams?n=${n}`),
  clearRoom: (roomId: string) => post<DemoState>(`/rooms/${encodeURIComponent(roomId)}/clear`),
  posterior: () => request<Posterior>("/posterior"),
  plan: (teams = 1) => request<SearchPlan>(`/plan?teams=${teams}`),
};

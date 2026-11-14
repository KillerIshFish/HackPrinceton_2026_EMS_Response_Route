import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Building2, ChevronRight, RotateCcw, Sparkles, TriangleAlert, Users } from "lucide-react";
import { api } from "./api";
import type { Building, BuildingMeta, ClientConfig, DemoState, HistoryEntry, ParserMode } from "./types";
import { mmss, mostProbableFloor, pct, roomIndex } from "./lib";
import { FloorPlanHeatmap } from "./components/FloorPlanHeatmap";
import { FloorProbBar } from "./components/FloorProbBar";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { SearchPlan } from "./components/SearchPlan";
import { AerialApproach } from "./components/AerialApproach";
import { ScanViewer } from "./components/ScanViewer";

const BACKEND_CMD = "uv run uvicorn api.server:app --reload --port 8000";
const TEAM_OPTIONS = [1, 2];

/** "step" | "reset" | "teams" | "building" | "parser" | "say" | "clear:<room id>" while a request is in flight */
type Busy = string | null;

export default function App() {
  const [allBuildings, setAllBuildings] = useState<Building[]>([]);
  const [state, setState] = useState<DemoState | null>(null);
  const [meta, setMeta] = useState<BuildingMeta | null>(null);
  const [config, setConfig] = useState<ClientConfig | null>(null);
  /** room whose 3D scan the dispatcher pinned; null = the viewer follows the posterior */
  const [pinnedRoomId, setPinnedRoomId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<Busy>(null);
  const [follow, setFollow] = useState(true);
  const [floor, setFloor] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [s, buildings, cfg] = await Promise.all([api.state(), api.buildings(), api.config()]);
      if (buildings.length === 0) throw new Error("Backend returned no buildings.");
      setAllBuildings(buildings);
      setConfig(cfg);
      setState(s);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  /** Run a mutating call; every response is a full DemoState that replaces local state. */
  const run = useCallback(async (kind: string, call: () => Promise<DemoState>) => {
    setBusy(kind);
    setActionError(null);
    try {
      setState(await call());
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const buildingId = state?.building_id ?? null;
  const building = useMemo(
    () => allBuildings.find((b) => b.id === buildingId) ?? null,
    [allBuildings, buildingId],
  );

  // A building switch starts a new call: reload its metadata and drop per-building UI state.
  useEffect(() => {
    if (!buildingId) return;
    let cancelled = false;
    setMeta(null);
    setPinnedRoomId(null);
    setFollow(true);
    api
      .buildingMeta(buildingId)
      .then((m) => {
        if (!cancelled) setMeta(m);
      })
      .catch(() => {
        /* metadata is optional: the console works without scans or imagery */
      });
    return () => {
      cancelled = true;
    };
  }, [buildingId]);

  const scannedRoomIds = useMemo(
    () => new Set(Object.entries(meta?.rooms ?? {}).filter(([, m]) => m.model_id).map(([id]) => id)),
    [meta],
  );

  // Follow the most probable floor whenever the posterior changes (unless the user pinned a floor).
  const topFloor = useMemo(
    () => (building && state ? mostProbableFloor(building, state.posterior) : null),
    [building, state],
  );
  useEffect(() => {
    if (follow && topFloor !== null) setFloor(topFloor);
  }, [follow, topFloor]);

  if (!building || !state) {
    return <LoadScreen loading={loading} error={loadError} onRetry={() => void load()} />;
  }

  const shownFloor = floor ?? topFloor ?? building.floors[0] ?? 1;
  // Matterport's plan image of the shown floor (buildings built from a whole-building scan)
  const plan = meta?.building_id === building.id ? meta.floorplans?.[String(shownFloor)] : undefined;
  const planImage = plan ? { ...plan, href: api.floorplanUrl(building.id, shownFloor) } : null;
  const clearingRoomId = busy?.startsWith("clear:") ? busy.slice(6) : null;

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-bg/95 px-3 py-2 backdrop-blur">
        <div className="flex min-w-0 items-center gap-2.5">
          <Building2 size={18} className="shrink-0 text-ink-mute" aria-hidden />
          <div className="min-w-0">
            <h1 className="truncate text-[15px] font-semibold leading-tight text-ink">
              {state.buildings.length > 1 ? (
                <select
                  aria-label="Building"
                  value={building.id}
                  disabled={busy !== null}
                  onChange={(e) => void run("building", () => api.setBuilding(e.target.value))}
                  className="-ml-1 max-w-full cursor-pointer truncate rounded border border-transparent bg-bg px-1 py-0.5 text-[15px] font-semibold text-ink hover:border-line-strong focus:border-accent focus:outline-none"
                >
                  {state.buildings.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name}
                      {b.real ? "" : " (synthetic)"}
                    </option>
                  ))}
                </select>
              ) : (
                building.name
              )}
            </h1>
            <p className="text-[11px] leading-tight text-ink-mute">
              {meta?.address ? `${meta.address} - ` : "Caller location - "}
              {building.floors.length} floors, {building.rooms.length} rooms modelled
              {meta?.room_sources ? ` (${sourceSummary(meta.room_sources, building.rooms.length)})` : ""}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded border border-line px-2 py-1">
          <span className="live-dot h-2 w-2 rounded-full bg-alert" aria-hidden />
          <span className="text-xs font-semibold tracking-wide text-ink">911 CALL ACTIVE</span>
          <span className="num border-l border-line pl-2 text-xs text-ink-dim">
            Turn {Math.min(state.turn + 1, state.total_turns)} of {state.total_turns}
          </span>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ParserToggle
            mode={state.parser_mode}
            available={state.llm?.available ?? false}
            calls={state.llm?.usage.calls ?? 0}
            tokens={(state.llm?.usage.input_tokens ?? 0) + (state.llm?.usage.output_tokens ?? 0)}
            disabled={busy !== null}
            onChange={(mode) => void run("parser", () => api.setParser(mode))}
          />
          <div className="flex items-center gap-1.5" role="group" aria-label="Search teams">
            <Users size={14} className="text-ink-mute" aria-hidden />
            <span className="text-xs text-ink-mute">Teams</span>
            <div className="flex overflow-hidden rounded border border-line">
              {TEAM_OPTIONS.map((n) => {
                const active = state.n_teams === n;
                return (
                  <button
                    key={n}
                    aria-pressed={active}
                    disabled={busy !== null}
                    onClick={() => !active && void run("teams", () => api.setTeams(n))}
                    className={`num min-w-8 border-l border-line px-2 py-1 text-xs first:border-l-0 disabled:cursor-not-allowed ${
                      active ? "bg-ink font-semibold text-bg" : "text-ink-dim hover:bg-raised hover:text-ink"
                    }`}
                  >
                    {n}
                  </button>
                );
              })}
            </div>
          </div>
          <button
            onClick={() => void run("reset", api.reset)}
            disabled={busy !== null}
            className="flex items-center gap-1.5 rounded border border-line px-2.5 py-1.5 text-xs text-ink-dim hover:bg-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RotateCcw size={13} aria-hidden />
            Reset
          </button>
          <button
            onClick={() => void run("step", api.step)}
            disabled={state.done || busy !== null}
            className="flex items-center gap-1 rounded bg-accent px-3 py-1.5 text-[13px] font-semibold text-accent-ink hover:bg-accent-strong disabled:cursor-not-allowed disabled:bg-raised disabled:text-ink-mute"
          >
            {state.done ? "Call script complete" : busy === "step" ? "Updating…" : "Next Call Turn"}
            {state.done ? null : <ChevronRight size={15} aria-hidden />}
          </button>
        </div>
      </header>

      {actionError ? (
        <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-alert/40 bg-alert/10 px-3 py-2 text-xs text-ink">
          <TriangleAlert size={14} className="shrink-0 text-alert" aria-hidden />
          <span className="min-w-0 flex-1">{actionError}</span>
          <button onClick={() => void load()} className="rounded border border-line-strong px-2 py-0.5 text-ink-dim hover:text-ink">
            Reload state
          </button>
          <button onClick={() => setActionError(null)} className="text-ink-mute hover:text-ink">
            Dismiss
          </button>
        </div>
      ) : null}

      <StatsStrip state={state} building={building} />

      <main className="grid flex-1 gap-3 p-3 xl:grid-cols-[320px_minmax(0,1fr)_360px] xl:items-start">
        <div className="max-h-[70vh] min-h-0 xl:sticky xl:top-[60px] xl:h-[calc(100vh-72px)] xl:max-h-none">
          <TranscriptPanel
            transcript={state.transcript}
            clues={state.clues}
            nextQuestion={state.next_question}
            turn={state.turn}
            totalTurns={state.total_turns}
            parserMode={state.parser_mode}
            busy={busy !== null}
            onSay={(text) => void run("say", () => api.say(text))}
          />
        </div>

        <div className="flex min-w-0 flex-col gap-3">
          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_176px]">
            <FloorPlanHeatmap
              building={building}
              posterior={state.posterior}
              plan={state.plan}
              clearedRoomIds={state.cleared_room_ids}
              floor={shownFloor}
              follow={follow}
              onSelectFloor={(f) => {
                setFollow(false);
                setFloor(f);
              }}
              onToggleFollow={() => setFollow((v) => !v)}
              roomMeta={meta?.rooms}
              onSelectRoom={setPinnedRoomId}
              planImage={planImage}
              rotationDeg={meta?.georeference?.rotation_deg ?? 0}
            />
            <FloorProbBar
              building={building}
              posterior={state.posterior}
              phoneFix={state.phone_fix}
              selectedFloor={shownFloor}
              onSelectFloor={(f) => {
                setFollow(false);
                setFloor(f);
              }}
            />
          </div>
          {meta && meta.matterport_models.length > 0 ? (
            <ScanViewer
              building={building}
              meta={meta}
              posterior={state.posterior}
              sdkKey={config?.matterport_sdk_key ?? null}
              pinnedRoomId={pinnedRoomId}
              onPin={setPinnedRoomId}
            />
          ) : null}
          <AerialApproach building={building} plan={state.plan} approach={state.approach} meta={meta} />
        </div>

        <div className="max-h-[80vh] min-h-0 xl:sticky xl:top-[60px] xl:h-[calc(100vh-72px)] xl:max-h-none">
          <SearchPlan
            building={building}
            plan={state.plan}
            posterior={state.posterior}
            clearedRoomIds={state.cleared_room_ids}
            clearingRoomId={clearingRoomId}
            busy={busy !== null}
            onClear={(roomId) => void run(`clear:${roomId}`, () => api.clearRoom(roomId))}
            scannedRoomIds={scannedRoomIds}
            onViewScan={setPinnedRoomId}
          />
        </div>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------

/** "28 scanned, 21 from suite records, 99 inferred", or "all scanned" for a whole-building scan */
function sourceSummary(sources: Record<string, number>, total: number): string {
  const scanned = sources.matterport ?? 0;
  if (scanned === total) return "all from one Matterport scan";
  const parts = [
    [scanned, "scanned"],
    [sources.costar ?? 0, "from suite records"],
    [sources.inferred ?? 0, "inferred"],
  ] as const;
  return parts
    .filter(([n]) => n > 0)
    .map(([n, label]) => `${n} ${label}`)
    .join(", ");
}

function ParserToggle({
  mode,
  available,
  calls,
  tokens,
  disabled,
  onChange,
}: {
  mode: ParserMode;
  available: boolean;
  calls: number;
  tokens: number;
  disabled: boolean;
  onChange: (mode: ParserMode) => void;
}) {
  const options: { key: ParserMode; label: string }[] = [
    { key: "rules", label: "Rules" },
    { key: "llm", label: "Claude Haiku" },
  ];
  return (
    <div
      className="flex items-center gap-1.5"
      role="group"
      aria-label="Clue parser"
      title={
        available
          ? `Clue parser. Claude Haiku has made ${calls} request${calls === 1 ? "" : "s"} (${tokens.toLocaleString()} tokens) in total; repeated utterances are served from a local cache.`
          : "Clue parser. Claude Haiku needs LLM_PROVIDER=anthropic and LLM_API_KEY in .env."
      }
    >
      <Sparkles size={14} className="text-ink-mute" aria-hidden />
      <span className="text-xs text-ink-mute">Parser</span>
      <div className="flex overflow-hidden rounded border border-line">
        {options.map((o) => {
          const active = mode === o.key;
          const off = disabled || (o.key === "llm" && !available);
          return (
            <button
              key={o.key}
              aria-pressed={active}
              disabled={off}
              onClick={() => !active && onChange(o.key)}
              className={`border-l border-line px-2 py-1 text-xs first:border-l-0 disabled:cursor-not-allowed ${
                active ? "bg-ink font-semibold text-bg" : "text-ink-dim hover:bg-raised hover:text-ink disabled:opacity-50"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LoadScreen({ loading, error, onRetry }: { loading: boolean; error: string | null; onRetry: () => void }) {
  return (
    <div className="flex min-h-full items-center justify-center p-6">
      {error ? (
        <div role="alert" className="w-full max-w-lg rounded-md border border-alert/50 bg-panel p-5">
          <div className="flex items-center gap-2.5">
            <TriangleAlert size={18} className="shrink-0 text-alert" aria-hidden />
            <h1 className="text-[15px] font-semibold text-ink">Can't reach the location backend</h1>
          </div>
          <p className="mt-2 text-[13px] text-ink-dim">{error}</p>
          <p className="mt-3 text-[13px] text-ink-dim">Start the API from the repo root, then retry:</p>
          <pre className="num mt-1.5 overflow-x-auto rounded border border-line bg-bg px-3 py-2 text-xs text-ink">{BACKEND_CMD}</pre>
          <button
            onClick={onRetry}
            disabled={loading}
            className="mt-4 rounded bg-accent px-3 py-1.5 text-[13px] font-semibold text-accent-ink hover:bg-accent-strong disabled:opacity-60"
          >
            {loading ? "Retrying…" : "Retry"}
          </button>
        </div>
      ) : (
        <p className="text-sm text-ink-mute" role="status">
          Loading call state…
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 44;
  const h = 16;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [
    1.5 + (i / (values.length - 1)) * (w - 3),
    1.5 + (1 - (v - min) / span) * (h - 3),
  ]);
  const last = pts[pts.length - 1]!;
  return (
    <svg width={w} height={h} className="shrink-0" aria-hidden>
      <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="#8394a6" strokeWidth={1.3} strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r={2} fill="#e6edf3" />
    </svg>
  );
}

/** Change since the previous call turn. For entropy and E[T], down is good. */
function Delta({ now, prev, format }: { now: number; prev: number | undefined; format: (d: number) => string }) {
  if (prev === undefined) return null;
  const d = now - prev;
  if (Math.abs(d) < 1e-6) return null;
  const down = d < 0;
  return (
    <span className={`num text-[11px] ${down ? "text-prob" : "text-ink-mute"}`} title="Change since the previous call turn">
      {down ? "▼" : "▲"} {format(Math.abs(d))}
    </span>
  );
}

function Stat({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col justify-center gap-0.5 bg-panel px-3 py-2" title={title}>
      <span className="eyebrow !text-[10px]">{label}</span>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">{children}</div>
    </div>
  );
}

function StatsStrip({ state, building }: { state: DemoState; building: Building }) {
  const { posterior, plan, history } = state;
  const rooms = useMemo(() => roomIndex(building), [building]);
  // history has one entry per revealed turn; the previous turn is the one before the latest.
  const prev: HistoryEntry | undefined = history.length >= 2 ? history[history.length - 2] : undefined;

  let topId: string | null = null;
  let topP = 0;
  for (const [id, p] of Object.entries(posterior.room_probs)) if (p > topP) [topId, topP] = [id, p];
  const topRoom = topId ? rooms.get(topId) : undefined;

  // trend ends at the live value (which also reflects cleared rooms / team changes)
  const entropyTrend = [...history.slice(0, -1).map((h) => h.entropy_bits), posterior.entropy_bits];
  const etTrend = [...history.slice(0, -1).map((h) => h.expected_time_to_find_s), plan.expected_time_to_find_s];

  return (
    <div className="grid grid-cols-2 gap-px border-b border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
      <Stat label="Top room" title="Most probable room right now">
        <span className="truncate text-[15px] font-semibold text-ink">{topRoom?.name ?? "-"}</span>
        <span className="num text-[13px] text-ink-dim">
          {topRoom ? `F${topRoom.floor_label} - ` : ""}
          {pct(topP)}
        </span>
      </Stat>
      <Stat label="Entropy" title="Shannon entropy of the room posterior. Lower = more certain.">
        <span className="num text-[15px] font-semibold text-ink">{posterior.entropy_bits.toFixed(2)}</span>
        <span className="text-[11px] text-ink-mute">bits</span>
        <Delta now={posterior.entropy_bits} prev={prev?.entropy_bits} format={(d) => d.toFixed(2)} />
        <Sparkline values={entropyTrend} />
      </Stat>
      <Stat label="Peaks" title="Connected clusters of high-probability rooms">
        <span className="num text-[15px] font-semibold text-ink">{posterior.peak_count}</span>
        <span className="text-[11px] text-ink-mute">{posterior.peak_count === 1 ? "cluster" : "clusters"}</span>
      </Stat>
      <Stat label="90% credible set" title="Smallest set of rooms holding 90% of the in-model probability">
        <span className="num text-[15px] font-semibold text-ink">{posterior.credible_set_90.length}</span>
        <span className="text-[11px] text-ink-mute">of {building.rooms.length} rooms</span>
      </Stat>
      <Stat label="Not in model" title="Probability the caller is somewhere the building model does not cover">
        <span className="num text-[15px] font-semibold text-ink">{pct(posterior.p_not_in_model)}</span>
      </Stat>
      <Stat label="E[T] to find" title="Expected time to find the caller following the current search plan">
        <span className="num text-[15px] font-semibold text-ink">{mmss(plan.expected_time_to_find_s)}</span>
        <Delta now={plan.expected_time_to_find_s} prev={prev?.expected_time_to_find_s} format={(d) => `${d.toFixed(0)} s`} />
        <Sparkline values={etTrend} />
      </Stat>
    </div>
  );
}

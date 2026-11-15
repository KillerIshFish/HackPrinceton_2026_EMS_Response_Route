import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Box, Building2, ChevronRight, Clapperboard, Layers, Map as MapIcon, RotateCcw, ScanLine, Sparkles, TriangleAlert } from "lucide-react";
import { api } from "./api";
import type {
  AskBody,
  Answer as CallerAnswer,
  Building,
  BuildingMeta,
  ClientConfig,
  DemoState,
  Gates,
  ParserMode,
  ScenarioSummary,
  SearchMode,
  SituationPatch,
  TimestampName,
} from "./types";
import { mostProbableFloor } from "./lib";
import { FloorPlanHeatmap } from "./components/FloorPlanHeatmap";
import { FloorProbBar } from "./components/FloorProbBar";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { SearchPlan, gatesWithRoom } from "./components/SearchPlan";
import { AerialApproach } from "./components/AerialApproach";
import { ScanViewer, useScanAvailability } from "./components/ScanViewer";
import { Building3D } from "./components/Building3D";
import { BuildingSearch } from "./components/BuildingSearch";
import { PolicyBanners, SituationBar } from "./components/SituationBar";
import { AnswerStrip } from "./components/AnswerStrip";
import { ViewTabs } from "./components/ViewTabs";
import type { ViewDef, ViewId } from "./components/ViewTabs";
import { IncidentBoard } from "./components/IncidentBoard";
import { StakeCard } from "./components/StakeCard";
import { PremiseFacts } from "./components/PremiseFacts";

const BACKEND_CMD = "uv run uvicorn api.server:app --reload --port 8000";
const VIEW_IDS: ViewId[] = ["plan", "3d", "scan", "approach"];

/**
 * What a request in flight is doing: "step" | "reset" | "building" | "generate" | "parser" |
 * "situation" | "scenario" | "say" | "ask" | "evidence:<id>" | "searched:<room id>" |
 * "clear:<room id>" | "found:<room id>" | "undo" | "gates" | "timestamp".
 */
type Busy = string | null;

/** Every mutation the console can make. Each one replaces the whole DemoState with the server's answer. */
export interface ConsoleActions {
  say: (text: string, speaker?: "caller" | "dispatcher", answers?: CallerAnswer | null) => void;
  ask: (question: AskBody) => void;
  rejectEvidence: (id: string) => void;
  restoreEvidence: (id: string) => void;
  searched: (roomId: string, mode?: SearchMode) => void;
  /** pre-Step-0 "Mark cleared": a rapid search recorded once per room */
  clearRoom: (roomId: string) => void;
  found: (roomId: string) => void;
  undo: () => void;
  setGates: (gates: Gates) => void;
  timestamp: (name: TimestampName) => void;
  patchSituation: (patch: SituationPatch) => void;
  scenario: (id: string) => void;
}

/** A small piece of UI state remembered in this browser (best effort: storage can be unavailable). */
function useRemembered<T extends string>(key: string, initial: T, allowed?: readonly T[]): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const v = window.localStorage.getItem(key) as T | null;
      if (v !== null && (!allowed || allowed.includes(v))) return v;
    } catch {
      /* private window or blocked storage */
    }
    return initial;
  });
  const set = useCallback(
    (v: T) => {
      setValue(v);
      try {
        window.localStorage.setItem(key, v);
      } catch {
        /* ignore */
      }
    },
    [key],
  );
  return [value, set];
}

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

  /** Generate a building's model now (address, property id or "mp:<model id>") and start a call on it. */
  const openGenerated = useCallback(async (query: string) => {
    setBusy("generate");
    setActionError(null);
    try {
      const report = await api.generate(query);
      setAllBuildings(await api.buildings());
      setState(await api.setBuilding(report.building_id));
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);

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

  // ?building=<id> opens that building: an existing one directly, a generated one ("prop-<id>", "mp-<id>") by building it.
  const wanted = useMemo(() => new URLSearchParams(window.location.search).get("building"), []);
  const [wantedDone, setWantedDone] = useState(false);
  useEffect(() => {
    if (!wanted || wantedDone || !state || allBuildings.length === 0) return;
    setWantedDone(true);
    if (state.building_id === wanted) return;
    if (allBuildings.some((b) => b.id === wanted)) void run("building", () => api.setBuilding(wanted));
    else if (wanted.startsWith("prop-")) void openGenerated(wanted.slice(5));
    else if (wanted.startsWith("mp-")) void openGenerated(`mp:${wanted.slice(3)}`);
  }, [wanted, wantedDone, state, allBuildings, run, openGenerated]);

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

  // A generated building's position on the map is refined when the OpenStreetMap lookup
  // returns (it can take a minute). While that is pending, ask for the metadata again and,
  // once it has settled, reload the buildings so entrances and staging carry the new position.
  const sitePending = meta?.building_id === buildingId && meta?.site.pending === true;
  useEffect(() => {
    if (!sitePending || !buildingId) return;
    let cancelled = false;
    let tries = 0;
    const timer = window.setInterval(() => {
      tries += 1;
      if (tries > 30) window.clearInterval(timer);
      api
        .buildingMeta(buildingId)
        .then(async (m) => {
          if (cancelled || m.site.pending) return;
          window.clearInterval(timer);
          const buildings = await api.buildings();
          if (cancelled) return;
          setAllBuildings(buildings);
          setMeta(m);
        })
        .catch(() => undefined);
    }, 6000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sitePending, buildingId]);

  // Which Matterport tours the scan viewer can show for this building. The Scan tab and every
  // "Show 3D scan" action follow it: a building with no tour that loads offers neither.
  const scan = useScanAvailability(building, meta);
  /** room metadata for the floor plan: a room whose tour cannot be shown carries no scan */
  const planRoomMeta = useMemo(() => {
    if (!meta) return undefined;
    const out: BuildingMeta["rooms"] = {};
    for (const [id, m] of Object.entries(meta.rooms)) {
      if (!m.model_id || scan.roomIds.has(id)) out[id] = m;
      else {
        const { model_id: _model, sweep_id: _sweep, ...rest } = m;
        out[id] = rest;
      }
    }
    return out;
  }, [meta, scan]);

  // Follow the most probable floor whenever the posterior changes (unless the user pinned a floor).
  const topFloor = useMemo(
    () => (building && state ? mostProbableFloor(building, state.posterior) : null),
    [building, state],
  );
  useEffect(() => {
    if (follow && topFloor !== null) setFloor(topFloor);
  }, [follow, topFloor]);


  // ---- console-wide UI state ----
  /** chosen map view, remembered across reloads */
  const [view, setView] = useRemembered<ViewId>("console.view", "plan", VIEW_IDS);
  const [detailsOpen, setDetailsOpen] = useRemembered<"open" | "closed">("console.details", "closed", ["open", "closed"]);
  /**
   * Rooms outlined on the map. One piece of state for the whole console:
   *   D2: TranscriptPanel / EvidenceChip call `onHoverRooms={setOutlineRoomIds}`
   *   D3: FloorPlanHeatmap reads `outlineRoomIds={outlineRoomIds}`
   * The answer strip already reads and sets it (hovering "Most likely").
   */
  const [outlineRoomIds, setOutlineRoomIds] = useState<string[] | null>(null);

  /** Handlers for D2/D3/D4: pass `actions.x` as a prop instead of adding request plumbing here. */
  const actions: ConsoleActions = useMemo(
    () => ({
      say: (text, speaker = "caller", answers = null) => void run("say", () => api.say(text, speaker, answers)),
      ask: (question) => void run("ask", () => api.ask(question)),
      rejectEvidence: (id) => void run(`evidence:${id}`, () => api.rejectEvidence(id)),
      restoreEvidence: (id) => void run(`evidence:${id}`, () => api.restoreEvidence(id)),
      searched: (roomId, mode = "rapid") => void run(`searched:${roomId}`, () => api.searched(roomId, mode)),
      clearRoom: (roomId) => void run(`clear:${roomId}`, () => api.clearRoom(roomId)),
      found: (roomId) => void run(`found:${roomId}`, () => api.found(roomId)),
      undo: () => void run("undo", api.undo),
      setGates: (gates) => void run("gates", () => api.setGates(gates)),
      timestamp: (name) => void run("timestamp", () => api.timestamp(name)),
      patchSituation: (patch) => void run("situation", () => api.setSituation(patch)),
      scenario: (id) => void run("scenario", () => api.scenario(id)),
    }),
    [run],
  );

  if (!building || !state) {
    return <LoadScreen loading={loading} error={loadError} onRetry={() => void load()} />;
  }

  const shownFloor = floor ?? topFloor ?? building.floors[0] ?? 1;
  const selectFloor = (f: number) => {
    setFollow(false);
    setFloor(f);
  };
  // Matterport's plan image of the shown floor (buildings built from a whole-building scan)
  const plan = meta?.building_id === building.id ? meta.floorplans?.[String(shownFloor)] : undefined;
  const planImage = plan ? { ...plan, href: api.floorplanUrl(building.id, shownFloor) } : null;
  const hasScan = meta !== null && scan.available;
  /** "Show 3D scan" on a room: pin it and open the Scan tab */
  const viewScan = (roomId: string) => {
    setPinnedRoomId(roomId);
    if (hasScan) setView("scan");
  };
  const { policy } = state;

  // -------------------------------------------------------------------------
  // SLOTS for step D4. Each renders nothing until D4 fills it; D4 edits only these three lines.
  // -------------------------------------------------------------------------
  /** D4: <IncidentBoard .../>. Replaces the search plan when policy.show_incident_board (active threat). */
  const incidentBoard: ReactNode = (
    <IncidentBoard facts={state.facts} transcript={state.transcript} building={building} callStartedAt={state.situation.call_started_at} />
  );
  /** D4: <StakeCard .../> from policy.stake. Sits directly under the answer strip. */
  const stakeCard: ReactNode = <StakeCard policy={policy} approach={state.approach} incidentType={state.situation.incident_type} />;
  /** D4: <PremiseFacts .../>. Sits under the search plan in the right column. */
  const premiseFacts: ReactNode = (
    <PremiseFacts facts={state.facts} situation={state.situation} building={building} elevatorAllowed={policy.elevator_allowed} />
  );

  // ---- map views (centre column). D3 owns the floor plan, D4 the 3D and approach views. ----
  const views: ViewDef[] = [
    {
      id: "plan",
      label: "Floor plan",
      icon: <Layers size={14} aria-hidden />,
      content: (
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_140px] sm:items-start xl:h-full 2xl:grid-cols-[minmax(0,1fr)_176px]">
          <FloorPlanHeatmap
            building={building}
            posterior={state.posterior}
            plan={state.plan}
            searched={state.searched}
            closedRoomIds={state.situation.closed_room_ids}
            outlineRoomIds={outlineRoomIds}
            routeEnabled={policy.route_enabled}
            showHazardClass={policy.show_hazard_class}
            onSearched={actions.searched}
            onSetClosed={(id, closed) => actions.setGates(gatesWithRoom(state.situation, id, closed))}
            busy={busy !== null}
            floor={shownFloor}
            follow={follow}
            onSelectFloor={selectFloor}
            onToggleFollow={() => setFollow((v) => !v)}
            roomMeta={planRoomMeta}
            onSelectRoom={hasScan ? viewScan : undefined}
            planImage={planImage}
            rotationDeg={meta?.georeference?.rotation_deg ?? 0}
          />
          <FloorProbBar
            building={building}
            posterior={state.posterior}
            phoneFix={state.phone_fix}
            selectedFloor={shownFloor}
            onSelectFloor={selectFloor}
          />
        </div>
      ),
    },
    {
      id: "3d",
      label: "3D",
      icon: <Box size={14} aria-hidden />,
      content: (
        <Building3D
          building={building}
          posterior={state.posterior}
          plan={state.plan}
          floor={shownFloor}
          onSelectFloor={selectFloor}
          onSelectRoom={setPinnedRoomId}
          policy={policy}
          closedRoomIds={state.situation.closed_room_ids}
        />
      ),
    },
    {
      id: "scan",
      label: "Scan",
      icon: <ScanLine size={14} aria-hidden />,
      disabled: !hasScan,
      unavailableReason: scan.reason ?? "No 3D scan is on file for this building",
      content:
        meta && hasScan ? (
          <ScanViewer
            building={building}
            meta={meta}
            scan={scan}
            posterior={state.posterior}
            sdkKey={config?.matterport_sdk_key ?? null}
            pinnedRoomId={pinnedRoomId}
            onPin={setPinnedRoomId}
          />
        ) : null,
    },
    {
      id: "approach",
      label: "Approach",
      icon: <MapIcon size={14} aria-hidden />,
      content: <AerialApproach building={building} plan={state.plan} approach={state.approach} meta={meta} policy={policy} />,
    },
  ];

  return (
    // `relative` makes this the containing block for absolutely positioned sr-only text inside
    // the scrolling panels, so it is clipped here instead of growing the document past 100vh.
    <div className="relative flex min-h-full flex-col xl:h-screen xl:overflow-hidden">
      <header className="z-20 flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-bg px-3 py-1.5 2xl:gap-x-4">
        <div className="flex min-w-0 items-center gap-2.5 xl:max-w-[220px] 2xl:max-w-[400px]">
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
            <p
              className="truncate text-[11px] leading-tight text-ink-mute"
              title={`${building.floors.length} floors, ${building.rooms.length} rooms modelled${
                meta?.room_sources ? ` (${sourceSummary(meta.room_sources, building.rooms.length)})` : ""
              }`}
            >
              {meta?.address ? `${meta.address} - ` : "Caller location - "}
              {building.floors.length} floors, {building.rooms.length} rooms modelled
              {meta?.room_sources ? ` (${sourceSummary(meta.room_sources, building.rooms.length)})` : ""}
            </p>
          </div>
        </div>

        <div className="w-[200px] 2xl:w-[250px]">
          <BuildingSearch
            disabled={busy !== null}
            generating={busy === "generate"}
            onChoose={(hit) => void openGenerated(String(hit.property_id))}
          />
        </div>

        {state.scenarios.length > 0 ? (
          <ScenarioPicker
            scenarios={state.scenarios}
            value={state.scenario_id}
            disabled={busy !== null}
            onChange={actions.scenario}
          />
        ) : null}

        <div className="flex items-center gap-2 rounded border border-alert/50 px-2 py-1">
          <span className="live-dot h-2 w-2 rounded-full bg-alert" aria-hidden />
          <span className="text-xs font-semibold tracking-wide text-ink">
            911 <span className="hidden 2xl:inline">CALL ACTIVE</span>
          </span>
          <span className="num border-l border-line pl-2 text-xs text-ink-dim" title="Scripted call turn">
            Turn {Math.min(state.turn + 1, state.total_turns)}/{state.total_turns}
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
            {state.done ? (
              "Script complete"
            ) : busy === "step" ? (
              "Updating…"
            ) : (
              <>
                Next <span className="hidden 2xl:inline">call&nbsp;</span>turn
              </>
            )}
            {state.done ? null : <ChevronRight size={15} aria-hidden />}
          </button>
        </div>
      </header>

      <div className="shrink-0">
        <SituationBar
          situation={state.situation}
          nTeams={state.n_teams}
          busy={busy !== null}
          onPatch={actions.patchSituation}
        />
        <PolicyBanners banners={policy.banners} prompts={policy.protocol_prompts} />

        {actionError ? (
          <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-alert/40 bg-alert/10 px-3 py-1.5 text-xs text-ink">
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

        <AnswerStrip
          state={state}
          building={building}
          outlineRoomIds={outlineRoomIds}
          onHoverRooms={setOutlineRoomIds}
          showDetails={detailsOpen === "open"}
          onToggleDetails={() => setDetailsOpen(detailsOpen === "open" ? "closed" : "open")}
        />

        {/* SLOT stakeCard (D4) */}
        {stakeCard}
      </div>

      <main className="grid gap-3 p-3 xl:min-h-0 xl:flex-1 xl:grid-cols-[280px_minmax(0,1fr)_320px] 2xl:grid-cols-[340px_minmax(0,1fr)_380px]">
        {/* Left: the call (D2 owns TranscriptPanel) */}
        <div className="max-h-[70vh] min-h-0 xl:h-full xl:max-h-none">
          <TranscriptPanel
            transcript={state.transcript}
            evidence={state.evidence}
            facts={state.facts}
            questions={state.questions}
            agentTrace={state.agent_trace}
            building={building}
            activeThreat={policy.show_incident_board || state.situation.incident_type === "active_threat"}
            turn={state.turn}
            totalTurns={state.total_turns}
            parserMode={state.parser_mode}
            busy={busy}
            onSay={actions.say}
            onAsk={(q) =>
              actions.ask({ question: q.question, id: q.id, kind: q.kind, clue_type: q.clue_type, value: q.value, partition: q.partition })
            }
            onRejectEvidence={actions.rejectEvidence}
            onRestoreEvidence={actions.restoreEvidence}
            onHoverRooms={setOutlineRoomIds}
          />
        </div>

        {/* Centre: one map view at a time */}
        <div className="min-h-0 min-w-0 xl:h-full">
          <ViewTabs views={views} value={view} onValueChange={setView} />
        </div>

        {/* Right: the search plan, or the incident board for an active threat; premise facts under it */}
        <div className="flex min-h-0 flex-col gap-3 xl:h-full">
          {policy.show_incident_board ? (
            <>
              {/* SLOT incidentBoard (D4) */}
              {incidentBoard}
            </>
          ) : (
            <div className="max-h-[80vh] min-h-0 flex-1 xl:max-h-none">
              <SearchPlan
                building={building}
                state={state}
                busy={busy}
                actions={actions}
                onHoverRooms={setOutlineRoomIds}
                scannedRoomIds={scan.roomIds}
                onViewScan={hasScan ? viewScan : undefined}
              />
            </div>
          )}
          {/* SLOT premiseFacts (D4) */}
          {premiseFacts}
        </div>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------

function ScenarioPicker({
  scenarios,
  value,
  disabled,
  onChange,
}: {
  scenarios: ScenarioSummary[];
  value: string | null;
  disabled: boolean;
  onChange: (id: string) => void;
}) {
  const current = scenarios.find((s) => s.id === value);
  return (
    <label className="flex items-center gap-1.5" title={current?.description ?? "Start a scripted call"}>
      <Clapperboard size={14} className="text-ink-mute" aria-hidden />
      <span className="hidden text-xs text-ink-mute 2xl:inline">Scenario</span>
      <select
        aria-label="Scenario"
        value={value ?? ""}
        disabled={disabled}
        onChange={(e) => e.target.value && onChange(e.target.value)}
        className="max-w-[170px] cursor-pointer truncate rounded border border-line-strong 2xl:max-w-[240px] bg-bg px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
      >
        {value === null ? <option value="">Live call (no script)</option> : null}
        {scenarios.map((s) => (
          <option key={s.id} value={s.id}>
            {s.title}
          </option>
        ))}
      </select>
    </label>
  );
}

/** "28 scanned, 21 from suite records, 99 inferred", or "all scanned" for a whole-building scan */
function sourceSummary(sources: Record<string, number>, total: number): string {
  const scanned = sources.matterport ?? 0;
  if (scanned === total) return "all from one Matterport scan";
  const parts = [
    [scanned, "scanned"],
    [sources.costar ?? 0, "from unit and tenant records"],
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
    // one model mode: the server treats "agent" as an alias of "llm" (same grounded interpreter)
    { key: "llm", label: "Claude Haiku" },
  ];
  return (
    <div
      className="flex items-center gap-1.5"
      role="group"
      aria-label="Clue parser"
      title={
        available
          ? `Clue parser. Claude Haiku reads each caller turn against this building's room list (one request per turn). It has made ${calls} request${calls === 1 ? "" : "s"} (${tokens.toLocaleString()} tokens) in total; repeated utterances are served from a local cache.`
          : "Clue parser. Claude Haiku needs LLM_PROVIDER=anthropic and LLM_API_KEY in .env."
      }
    >
      <Sparkles size={14} className="text-ink-mute" aria-hidden />
      <span className="hidden text-xs text-ink-mute 2xl:inline">Parser</span>
      <div className="flex overflow-hidden rounded border border-line">
        {options.map((o) => {
          const active = mode === o.key || (o.key === "llm" && mode === "agent");
          const off = disabled || (o.key !== "rules" && !available);
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


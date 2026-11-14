import { useEffect, useMemo, useRef, useState } from "react";
import { Box, LocateFixed } from "lucide-react";
import type { Building, BuildingMeta, Posterior } from "../types";
import { pct, roomIndex } from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  meta: BuildingMeta;
  posterior: Posterior;
  /** Matterport SDK key from the backend config; without it the viewer still embeds but cannot be steered */
  sdkKey: string | null;
  /** room pinned by the dispatcher (from the floor plan or the search plan); null = follow the posterior */
  pinnedRoomId: string | null;
  onPin: (roomId: string | null) => void;
}

/** Matterport's Embed SDK bootstrap, an ES module that exports `connect` (developers.matterport.com/embed-sdk). */
const SDK_MODULE = "https://api.matterport.com/sdk/bootstrap/3.1.0-0-g1cd7e0997a/sdk.es6.js";

/** The slice of the Matterport Embed SDK this panel uses. */
interface MpSdk {
  App: {
    state: { subscribe: (cb: (state: { phase?: string }) => void) => unknown };
    Phase?: { PLAYING?: string };
  };
  Sweep: {
    moveTo: (sweep: string, options?: Record<string, unknown>) => Promise<string>;
    Transition?: { FLY?: string };
    data: { subscribe: (observer: { onCollectionUpdated?: (collection: Record<string, unknown>) => void }) => unknown };
    current?: { subscribe: (cb: (sweep: { sid?: string; id?: string }) => void) => unknown };
  };
  Mode: {
    moveTo: (mode: string) => Promise<unknown>;
    Mode: { INSIDE: string; DOLLHOUSE: string; FLOORPLAN: string };
    current?: { subscribe: (cb: (mode: string) => void) => unknown };
  };
}

/** what the viewer itself reports: the sweep it is standing on and its view mode */
interface ViewerState {
  sweep: string | null;
  mode: string | null;
}

/** Matterport's own view modes: walking inside, the 3D dollhouse, or its top-down floor plan */
type ViewMode = "inside" | "dollhouse" | "floorplan";
const VIEW_MODES: { key: ViewMode; label: string }[] = [
  { key: "inside", label: "Inside" },
  { key: "dollhouse", label: "Dollhouse" },
  { key: "floorplan", label: "Floor plan" },
];
interface MpSdkModule {
  connect: (target: HTMLIFrameElement) => Promise<MpSdk>;
}

let sdkModule: Promise<MpSdkModule> | null = null;
function loadSdk(sdkKey: string): Promise<MpSdkModule> {
  if (!sdkModule) {
    const url = `${SDK_MODULE}?applicationKey=${encodeURIComponent(sdkKey)}`;
    sdkModule = (import(/* @vite-ignore */ url) as Promise<MpSdkModule>).catch((err) => {
      sdkModule = null;
      throw err;
    });
  }
  return sdkModule;
}

/**
 * Matterport's own viewer, embedded. It shows the scan of whichever scanned room is most
 * probable right now (or the room the dispatcher pinned) and flies to that room's sweep.
 */
export function ScanViewer({ building, meta, posterior, sdkKey, pinnedRoomId, onPin }: Props) {
  const rooms = useMemo(() => roomIndex(building), [building]);
  const scanned = useMemo(
    () => building.rooms.filter((r) => meta.rooms[r.id]?.model_id),
    [building, meta],
  );

  // Follow mode: the scanned room with the highest posterior probability.
  const topScanned = useMemo(() => {
    let best: string | null = null;
    let bestP = -1;
    for (const r of scanned) {
      const p = posterior.room_probs[r.id] ?? 0;
      if (p > bestP) [best, bestP] = [r.id, p];
    }
    return best;
  }, [scanned, posterior]);

  const pinned = pinnedRoomId && meta.rooms[pinnedRoomId]?.model_id ? pinnedRoomId : null;
  const roomId = pinned ?? topScanned;
  const room = roomId ? rooms.get(roomId) : undefined;
  const rm = roomId ? meta.rooms[roomId] : undefined;
  const modelId = rm?.model_id ?? null;
  const sweepId = rm?.sweep_id ?? null;
  const model = meta.matterport_models.find((m) => m.model_id === modelId);

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const sdkRef = useRef<{ modelId: string; sdk: MpSdk; sweeps: Map<string, string> } | null>(null);
  /** connecting = SDK handshake; loading = the scan is still loading (commands sent now freeze it); ready = playing */
  const [steer, setSteer] = useState<"off" | "connecting" | "loading" | "ready" | "unavailable">("off");
  const [mode, setMode] = useState<ViewMode>("inside");
  /** number of sweep ids the viewer has reported; a change re-runs the steering effect */
  const [sweepCount, setSweepCount] = useState(0);
  /** what the viewer itself says it is showing; the status line and retries are based on this, not on our requests */
  const [viewer, setViewer] = useState<ViewerState>({ sweep: null, mode: null });
  /** one command at a time: true while a move is in flight; `tick` re-runs the steering effect when it ends */
  const busyRef = useRef(false);
  const triesRef = useRef<{ key: string; n: number }>({ key: "", n: 0 });
  const [tick, setTick] = useState(0);
  const [gaveUp, setGaveUp] = useState(false);
  /** the viewer's own reason when it refuses a command */
  const [steerError, setSteerError] = useState<string | null>(null);

  // Connect the SDK to each newly loaded model.
  useEffect(() => {
    sdkRef.current = null;
    setMode("inside");
    setSweepCount(0);
    setViewer({ sweep: null, mode: null });
    busyRef.current = false;
    if (!modelId || !sdkKey) {
      setSteer("off");
      return;
    }
    let cancelled = false;
    setSteer("connecting");
    const iframe = iframeRef.current;
    const connect = async () => {
      try {
        const mod = await loadSdk(sdkKey);
        if (cancelled || !iframe) return;
        const sdk = await mod.connect(iframe);
        if (cancelled) return;
        // Sweep ids: the SDK keys its sweeps by its own short id; each sweep also carries
        // the id the Model API uses. Map every id we see to the SDK's key.
        const entry = { modelId, sdk, sweeps: new Map<string, string>() };
        sdk.Sweep.data.subscribe({
          onCollectionUpdated: (collection) => {
            const ids = new Map<string, string>();
            const add = (key: string, item: unknown) => {
              ids.set(key, key);
              if (item && typeof item === "object") {
                for (const field of ["id", "sid", "uuid"]) {
                  const v = (item as Record<string, unknown>)[field];
                  if (typeof v === "string") ids.set(v, key);
                }
              }
            };
            const iterable = collection as unknown as Iterable<[string, unknown]>;
            if (typeof iterable[Symbol.iterator] === "function") {
              for (const [key, item] of iterable) add(key, item);
            } else {
              for (const [key, item] of Object.entries(collection)) add(key, item);
            }
            entry.sweeps = ids;
            if (!cancelled) setSweepCount(ids.size);
          },
        });
        sdk.Sweep.current?.subscribe((sweep) => {
          if (!cancelled) setViewer((v) => ({ ...v, sweep: sweep.sid || sweep.id || null }));
        });
        sdk.Mode.current?.subscribe((current) => {
          if (!cancelled) setViewer((v) => ({ ...v, mode: current || null }));
        });
        sdkRef.current = entry;
        // The viewer accepts commands while it is still loading, but a move sent then leaves it
        // frozen on its start image. Only steer once it reports that it is playing.
        setSteer("loading");
        const playing = sdk.App.Phase?.PLAYING ?? "appphase.playing";
        sdk.App.state.subscribe((state) => {
          if (!cancelled && state.phase === playing) setSteer("ready");
        });
      } catch {
        if (!cancelled) setSteer("unavailable");
      }
    };
    const onLoad = () => void connect();
    iframe?.addEventListener("load", onLoad);
    return () => {
      cancelled = true;
      iframe?.removeEventListener("load", onLoad);
    };
  }, [modelId, sdkKey]);

  const targetSweep = sweepId ? (sdkRef.current?.sweeps.get(sweepId) ?? sweepId) : null;
  const wantKey = `${modelId}|${mode}|${mode === "inside" ? sweepId : ""}`;
  const inRoom = mode === "inside" && !!targetSweep && viewer.mode === "mode.inside" && viewer.sweep === targetSweep;
  const inMode = mode !== "inside" && viewer.mode === (mode === "dollhouse" ? "mode.dollhouse" : "mode.floorplan");

  // Steering: compare what the dispatcher wants (a room's sweep, or an overview mode) with what
  // the viewer reports, and send one command at a time until they agree (three tries per target).
  useEffect(() => {
    const entry = sdkRef.current;
    if (steer !== "ready" || !entry || entry.modelId !== modelId) return;
    if (triesRef.current.key !== wantKey) {
      triesRef.current = { key: wantKey, n: 0 };
      setGaveUp(false);
      setSteerError(null);
    }
    if (busyRef.current || viewer.mode === "mode.transitioning") return;
    const { sdk } = entry;
    let command: (() => Promise<unknown>) | null = null;
    if (mode !== "inside") {
      if (!inMode) command = () => sdk.Mode.moveTo(mode === "dollhouse" ? sdk.Mode.Mode.DOLLHOUSE : sdk.Mode.Mode.FLOORPLAN);
    } else if (targetSweep && sweepCount > 0 && !inRoom) {
      const fly = () => sdk.Sweep.moveTo(targetSweep, { transition: sdk.Sweep.Transition?.FLY, transitionTime: 1500 });
      // from the dollhouse or the floor plan, drop back inside first
      command = viewer.mode && viewer.mode !== "mode.inside" ? () => sdk.Mode.moveTo(sdk.Mode.Mode.INSIDE).then(fly) : fly;
    }
    if (!command) return;
    if (triesRef.current.n >= 3) {
      setGaveUp(true);
      return;
    }
    triesRef.current.n += 1;
    busyRef.current = true;
    command()
      .catch((err: unknown) => setSteerError(String((err as { message?: string })?.message ?? err)))
      .finally(() => {
        busyRef.current = false;
        // give the viewer a moment to report where it ended up before judging the result
        window.setTimeout(() => setTick((t) => t + 1), 400);
      });
  }, [steer, mode, modelId, wantKey, targetSweep, sweepCount, inRoom, inMode, viewer.mode, tick]);

  // Picking a room (floor plan, search plan, suite menu) always shows that room, even from an overview mode.
  useEffect(() => {
    if (pinnedRoomId) setMode("inside");
  }, [pinnedRoomId]);

  function changeMode(next: ViewMode) {
    if (!sdkRef.current || next === mode) return;
    setMode(next);
  }

  if (scanned.length === 0) return null;

  const src = modelId
    ? `https://my.matterport.com/show/?m=${encodeURIComponent(modelId)}&play=1&qs=1&brand=0&help=0&title=0&search=0${
        sdkKey ? `&applicationKey=${encodeURIComponent(sdkKey)}` : ""
      }`
    : null;
  const p = roomId ? (posterior.room_probs[roomId] ?? 0) : 0;
  const modeLabel = mode === "dollhouse" ? "dollhouse" : "floor plan";
  /** a generated building's scans are of unit types, not of the individual rooms */
  const unitTypeScans = meta.kind === "generated";
  /** an overview mode the scan's owner switched off in Matterport (the viewer refuses it) */
  const modeOff = (key: ViewMode) =>
    (key === "dollhouse" && model?.dollhouse_enabled === false) || (key === "floorplan" && model?.floorplan_enabled === false);
  // one option per scan: its largest room is the entry point
  const options = meta.matterport_models.map((m) => ({
    model: m,
    roomId: scanned.find((r) => meta.rooms[r.id]?.model_id === m.model_id)?.id ?? null,
  }));

  return (
    <Panel
      title="3D scan"
      icon={<Box size={15} aria-hidden />}
      actions={
        <>
          {steer === "ready" ? (
            <div className="flex overflow-hidden rounded border border-line" role="group" aria-label="Viewer mode">
              {VIEW_MODES.map((m) => {
                const off = modeOff(m.key);
                return (
                  <button
                    key={m.key}
                    aria-pressed={mode === m.key}
                    disabled={off}
                    title={off ? `The owner of this scan switched its ${m.label.toLowerCase()} view off in Matterport` : undefined}
                    onClick={() => changeMode(m.key)}
                    className={`border-l border-line px-2 py-1 text-xs first:border-l-0 ${
                      off
                        ? "cursor-not-allowed text-ink-mute line-through"
                        : mode === m.key
                          ? "bg-ink font-semibold text-bg"
                          : "text-ink-dim hover:bg-raised hover:text-ink"
                    }`}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
          ) : null}
          {options.length > 1 ? (
            <label className="flex items-center gap-1.5 text-xs text-ink-mute">
              <span className="sr-only">Scanned suite</span>
              <select
                value={modelId ?? ""}
                onChange={(e) => {
                  const opt = options.find((o) => o.model.model_id === e.target.value);
                  if (opt?.roomId) onPin(opt.roomId);
                }}
                className="rounded border border-line bg-bg px-1.5 py-1 text-xs text-ink"
              >
                {options.map((o) => (
                  <option key={o.model.model_id} value={o.model.model_id}>
                    {o.model.suite ? `Suite ${o.model.suite}` : o.model.name || "Whole building"}
                    {o.model.floor ? ` - floor ${o.model.floor}` : ""}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span className="text-xs text-ink-mute">{meta.kind === "full_scan" ? "One scan of the whole building" : null}</span>
          )}
          <button
            onClick={() => onPin(null)}
            aria-pressed={pinned === null}
            title="Show the scan of the most probable scanned room, and switch as the posterior changes"
            className={`flex items-center gap-1.5 rounded border px-2 py-1 text-xs ${
              pinned === null ? "border-prob/60 bg-prob/10 text-prob" : "border-line text-ink-dim hover:bg-raised hover:text-ink"
            }`}
          >
            <LocateFixed size={13} aria-hidden />
            Follow posterior
          </button>
        </>
      }
      bodyClassName="flex flex-col"
    >
      <div className="relative aspect-video max-h-[460px] w-full bg-bg">
        {src ? (
          <iframe
            key={modelId}
            ref={iframeRef}
            src={src}
            title={`Matterport scan of ${model?.suite ? `Suite ${model.suite}` : "the building"}`}
            allow="fullscreen; xr-spatial-tracking"
            data-sweeps={sweepCount}
            data-steer={steer}
            data-steer-error={steerError ?? ""}
            data-arrived={String(inRoom)}
            data-target-sweep={targetSweep ?? ""}
            data-viewer-sweep={viewer.sweep ?? ""}
            data-viewer-mode={viewer.mode ?? ""}
            className="absolute inset-0 h-full w-full border-0"
          />
        ) : null}
      </div>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-line px-3 py-2 text-xs text-ink-dim">
        <span>
          <span className="font-medium text-ink">{room?.name ?? "-"}</span>
          {room ? <span className="num"> - floor {room.floor_label}</span> : null}
        </span>
        <span title="Posterior probability the caller is in this room">
          P(room) <span className="num text-ink">{pct(p)}</span>
        </span>
        {rm?.tenant ? <span>Tenant: {rm.tenant}</span> : rm?.status ? <span>Suite {rm.status}</span> : null}
        <span className="ml-auto text-[11px] text-ink-mute">
          {unitTypeScans
            ? `${scanned.length} of ${building.rooms.length} units have a scan of their unit type`
            : `${scanned.length} of ${building.rooms.length} rooms are scanned`}
          {steer === "ready"
            ? mode !== "inside"
              ? inMode
                ? ` - Matterport ${modeLabel} view`
                : gaveUp
                  ? ` - the viewer did not switch to the ${modeLabel} view`
                  : ` - switching to the ${modeLabel} view…`
              : !sweepId
                ? unitTypeScans
                  ? " - showing the scan of this unit type"
                  : " - no scan position for this room"
                : inRoom
                  ? " - viewer is in this room"
                  : gaveUp
                    ? " - the viewer could not reach this room"
                    : " - moving to the room…"
            : steer === "loading"
              ? " - loading the scan…"
              : steer === "connecting"
                ? " - connecting to viewer…"
                : steer === "unavailable"
                  ? " - viewer loaded, room steering unavailable"
                  : ""}
        </span>
      </div>
    </Panel>
  );
}

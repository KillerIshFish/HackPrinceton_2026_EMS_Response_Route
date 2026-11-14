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
  Sweep: {
    moveTo: (sweep: string, options?: Record<string, unknown>) => Promise<string>;
    Transition?: { FLY?: string };
    data: { subscribe: (observer: { onCollectionUpdated?: (collection: Record<string, unknown>) => void }) => unknown };
  };
}
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
  const sdkRef = useRef<{ modelId: string; sdk: MpSdk; sweeps: Set<string> } | null>(null);
  const [steer, setSteer] = useState<"off" | "connecting" | "ready" | "unavailable">("off");

  // Connect the SDK to each newly loaded model.
  useEffect(() => {
    sdkRef.current = null;
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
        const entry = { modelId, sdk, sweeps: new Set<string>() };
        sdk.Sweep.data.subscribe({
          onCollectionUpdated: (collection) => {
            entry.sweeps = new Set(Object.keys(collection));
            if (!cancelled) setSteer("ready");
          },
        });
        sdkRef.current = entry;
        setSteer("ready");
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

  // Fly to the room's sweep whenever the target room changes within the loaded model.
  useEffect(() => {
    const entry = sdkRef.current;
    if (steer !== "ready" || !entry || entry.modelId !== modelId || !sweepId) return;
    if (entry.sweeps.size > 0 && !entry.sweeps.has(sweepId)) return; // id not known to this viewer
    entry.sdk.Sweep.moveTo(sweepId, { transition: entry.sdk.Sweep.Transition?.FLY, transitionTime: 1500 }).catch(() => {
      /* the viewer may still be loading; the next posterior update retries */
    });
  }, [steer, modelId, sweepId]);

  if (scanned.length === 0) return null;

  const src = modelId
    ? `https://my.matterport.com/show/?m=${encodeURIComponent(modelId)}&play=1&qs=1&brand=0&help=0&title=0&search=0${
        sdkKey ? `&applicationKey=${encodeURIComponent(sdkKey)}` : ""
      }`
    : null;
  const p = roomId ? (posterior.room_probs[roomId] ?? 0) : 0;
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
                  Suite {o.model.suite}
                  {o.model.floor ? ` - floor ${o.model.floor}` : ""}
                </option>
              ))}
            </select>
          </label>
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
            title={`Matterport scan of ${model ? `Suite ${model.suite}` : "the selected suite"}`}
            allow="fullscreen; xr-spatial-tracking"
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
          {scanned.length} of {building.rooms.length} rooms are scanned
          {steer === "ready"
            ? " - viewer moves to the room"
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

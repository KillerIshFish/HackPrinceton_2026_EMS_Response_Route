import { useMemo, useRef, useState } from "react";
import { Layers, LocateFixed } from "lucide-react";
import type { Building, Posterior, Room, SearchPlan } from "../types";
import {
  RAMP_CSS_GRADIENT,
  boundsOf,
  buildingBounds,
  floorLabelMap,
  heatT,
  pct,
  rampColor,
  rampInk,
  titleCase,
} from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  posterior: Posterior;
  plan: SearchPlan;
  clearedRoomIds: string[];
  floor: number;
  follow: boolean;
  onSelectFloor: (floor: number) => void;
  onToggleFollow: () => void;
}

const S = 20; // svg units per metre
const PAD_X = 14;
const PAD_Y = 30; // room for the N / S side labels
const CHAR_W = 6.1; // approx glyph advance at 11px
const MARKED_STOPS = 5; // stops per team that get a numbered marker

interface StopMark {
  team: number;
  index: number;
}

interface Hover {
  roomId: string;
  x: number;
  y: number;
  boxW: number;
}

/** Which way do the windows on one side of the plate look? ("parking_lot" / "street") */
function sideFacing(rooms: Room[], north: boolean, midY: number): string | null {
  const counts = new Map<string, number>();
  for (const r of rooms) {
    if (!r.window_facing) continue;
    if (r.centroid[1] > midY !== north) continue;
    counts.set(r.window_facing, (counts.get(r.window_facing) ?? 0) + 1);
  }
  let best: string | null = null;
  let n = 0;
  for (const [k, v] of counts) if (v > n) [best, n] = [k, v];
  return best;
}

export function FloorPlanHeatmap({
  building,
  posterior,
  plan,
  clearedRoomIds,
  floor,
  follow,
  onSelectFloor,
  onToggleFollow,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  const bounds = useMemo(() => buildingBounds(building), [building]);
  const labels = useMemo(() => floorLabelMap(building), [building]);
  const rooms = useMemo(
    () =>
      building.rooms
        .filter((r) => r.floor === floor)
        // big rooms first so small ones stay hoverable when polygons touch
        .sort((a, b) => b.area_m2 - a.area_m2),
    [building, floor],
  );

  const W = (bounds.maxX - bounds.minX) * S + PAD_X * 2;
  const H = (bounds.maxY - bounds.minY) * S + PAD_Y * 2;
  const X = (x: number) => (x - bounds.minX) * S + PAD_X;
  const Y = (y: number) => (bounds.maxY - y) * S + PAD_Y; // flip: north is up

  const probs = posterior.room_probs;
  const maxP = useMemo(() => Math.max(0, ...Object.values(probs)), [probs]);
  const credible = useMemo(() => new Set(posterior.credible_set_90), [posterior.credible_set_90]);
  const cleared = useMemo(() => new Set(clearedRoomIds), [clearedRoomIds]);

  const rank = useMemo(() => {
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    return new Map(sorted.map(([id], i) => [id, i + 1]));
  }, [probs]);

  const stopMarks = useMemo(() => {
    const m = new Map<string, StopMark>();
    plan.teams.forEach((stops, team) =>
      stops.slice(0, MARKED_STOPS).forEach((s, index) => {
        if (!m.has(s.room_id)) m.set(s.room_id, { team, index });
      }),
    );
    return m;
  }, [plan]);
  const multiTeam = plan.teams.length > 1;

  const midY = (bounds.minY + bounds.maxY) / 2;
  const northFacing = sideFacing(building.rooms, true, midY);
  const southFacing = sideFacing(building.rooms, false, midY);

  const floorLabel = labels.get(floor) ?? String(floor);
  const floorP = posterior.p_floor[floorLabel] ?? 0;
  const hovered = hover ? rooms.find((r) => r.id === hover.roomId) : undefined;

  function placeHover(roomId: string, clientX: number, clientY: number) {
    const box = wrapRef.current?.getBoundingClientRect();
    if (!box) return;
    setHover({ roomId, x: clientX - box.left, y: clientY - box.top, boxW: box.width });
  }

  function focusHover(room: Room, el: SVGElement) {
    const r = el.getBoundingClientRect();
    placeHover(room.id, r.left + r.width / 2, r.top + r.height / 2);
  }

  const floorsTopDown = [...building.floors].sort((a, b) => b - a);

  return (
    <Panel
      title={`Floor plan - Floor ${floorLabel}`}
      icon={<Layers size={15} aria-hidden />}
      actions={
        <>
          <span className="num text-xs text-ink-dim">P(floor) {pct(floorP)}</span>
          <div className="flex overflow-hidden rounded border border-line" role="tablist" aria-label="Floor">
            {floorsTopDown.map((f) => {
              const active = f === floor;
              return (
                <button
                  key={f}
                  role="tab"
                  aria-selected={active}
                  onClick={() => onSelectFloor(f)}
                  className={`num min-w-8 border-l border-line px-2 py-1 text-xs first:border-l-0 ${
                    active ? "bg-ink text-bg font-semibold" : "text-ink-dim hover:bg-raised hover:text-ink"
                  }`}
                >
                  {labels.get(f) ?? f}
                </button>
              );
            })}
          </div>
          <button
            onClick={onToggleFollow}
            aria-pressed={follow}
            title="Automatically switch to the most probable floor when the posterior changes"
            className={`flex items-center gap-1.5 rounded border px-2 py-1 text-xs ${
              follow
                ? "border-prob/60 bg-prob/10 text-prob"
                : "border-line text-ink-dim hover:bg-raised hover:text-ink"
            }`}
          >
            <LocateFixed size={13} aria-hidden />
            Follow top floor
          </button>
        </>
      }
      bodyClassName="flex flex-col"
    >
      <div ref={wrapRef} className="relative px-2 pt-2" onMouseLeave={() => setHover(null)}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block h-auto w-full select-none"
          role="img"
          aria-label={`Probability heatmap of floor ${floorLabel}. North is up.`}
        >
          <defs>
            <pattern id="hatch-cleared" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="8" height="8" fill="#0b1016" fillOpacity="0.55" />
              <line x1="0" y1="0" x2="0" y2="8" stroke="#8394a6" strokeWidth="2.2" />
            </pattern>
          </defs>

          {/* side labels */}
          <text x={W / 2} y={PAD_Y - 11} textAnchor="middle" fontSize="11" fontWeight="600" letterSpacing="1.2" fill="#8394a6">
            N{northFacing ? ` - ${titleCase(northFacing).toUpperCase()} SIDE` : ""}
          </text>
          <text x={W / 2} y={H - PAD_Y + 20} textAnchor="middle" fontSize="11" fontWeight="600" letterSpacing="1.2" fill="#8394a6">
            S{southFacing ? ` - ${titleCase(southFacing).toUpperCase()} SIDE` : ""}
          </text>

          {/* room fills */}
          {rooms.map((room) => {
            const p = probs[room.id] ?? 0;
            const t = heatT(p, maxP);
            const pts = room.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
            return (
              <polygon
                key={room.id}
                points={pts}
                fill={rampColor(t)}
                stroke="#0b1016"
                strokeWidth={1.5}
                className="room-shape cursor-crosshair outline-none"
                tabIndex={0}
                aria-label={`${room.name}, ${titleCase(room.room_type)}, probability ${pct(p)}`}
                onMouseMove={(e) => placeHover(room.id, e.clientX, e.clientY)}
                onFocus={(e) => focusHover(room, e.currentTarget)}
                onBlur={() => setHover(null)}
              />
            );
          })}

          {/* cleared hatch + credible-set outline + hover/focus ring (above fills) */}
          {rooms.map((room) => {
            const pts = room.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
            const isCleared = cleared.has(room.id);
            const isCred = credible.has(room.id);
            const isHover = hover?.roomId === room.id;
            if (!isCleared && !isCred && !isHover) return null;
            return (
              <g key={room.id} pointerEvents="none">
                {isCleared ? <polygon points={pts} fill="url(#hatch-cleared)" /> : null}
                {isCred ? (
                  <polygon points={pts} fill="none" stroke="#f1fbe3" strokeWidth={2.6} strokeLinejoin="round" />
                ) : null}
                {isHover ? (
                  <polygon points={pts} fill="none" stroke="#f2a93b" strokeWidth={2} strokeDasharray="5 3" />
                ) : null}
              </g>
            );
          })}

          {/* labels + stop markers */}
          {rooms.map((room) => {
            const p = probs[room.id] ?? 0;
            const t = heatT(p, maxP);
            const b = boundsOf(room.polygon);
            const w = (b.maxX - b.minX) * S;
            const h = (b.maxY - b.minY) * S;
            const cx = X((b.minX + b.maxX) / 2);
            const cy = Y((b.minY + b.maxY) / 2);
            const isCleared = cleared.has(room.id);
            const ink = isCleared ? "#e6edf3" : rampInk(t);
            const nameFits = w >= room.name.length * CHAR_W + 8 && h >= 34;
            const pctFits = w >= 34 && h >= 16 && p >= 0.001;
            const mark = stopMarks.get(room.id);
            const isNext = mark?.index === 0;
            const mx = X(b.maxX) - 13;
            const my = Y(b.maxY) + 13;
            const markLabel = mark ? `${multiTeam ? String.fromCharCode(65 + mark.team) : ""}${mark.index + 1}` : "";
            return (
              <g key={room.id} pointerEvents="none">
                {nameFits ? (
                  <text
                    x={cx}
                    y={pctFits ? cy - 3 : cy + 4}
                    textAnchor="middle"
                    fontSize="11"
                    fill={ink}
                    fillOpacity={isCleared ? 0.75 : 0.92}
                    textDecoration={isCleared ? "line-through" : undefined}
                  >
                    {room.name}
                  </text>
                ) : null}
                {pctFits ? (
                  <text
                    x={cx}
                    y={nameFits ? cy + 12 : cy + 4}
                    textAnchor="middle"
                    fontSize={nameFits ? 12 : 10.5}
                    fontWeight="700"
                    fill={ink}
                    style={{ fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums" }}
                  >
                    {pct(p)}
                  </text>
                ) : null}
                {mark && !isCleared ? (
                  <g>
                    {isNext ? <circle cx={mx} cy={my} r={10} fill="#f2a93b" className="next-stop-ring" /> : null}
                    <circle
                      cx={mx}
                      cy={my}
                      r={isNext ? 10 : 8.5}
                      fill={isNext ? "#f2a93b" : "#0b1016"}
                      stroke={isNext ? "#0b1016" : "#f2a93b"}
                      strokeWidth={1.5}
                    />
                    <text
                      x={mx}
                      y={my + (isNext ? 4 : 3.5)}
                      textAnchor="middle"
                      fontSize={markLabel.length > 1 ? 9 : isNext ? 11.5 : 10}
                      fontWeight="700"
                      fill={isNext ? "#1a1204" : "#f2a93b"}
                    >
                      {markLabel}
                    </text>
                  </g>
                ) : null}
              </g>
            );
          })}
        </svg>

        {hover && hovered ? (
          <RoomTooltip
            room={hovered}
            p={probs[hovered.id] ?? 0}
            rank={rank.get(hovered.id)}
            inCredible={credible.has(hovered.id)}
            isCleared={cleared.has(hovered.id)}
            mark={stopMarks.get(hovered.id)}
            multiTeam={multiTeam}
            hover={hover}
          />
        ) : null}
      </div>

      {/* legend */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line px-3 py-2 text-xs text-ink-dim">
        <div className="flex items-center gap-2">
          <span className="num">0</span>
          <span
            className="relative block h-2.5 w-36 rounded-sm"
            style={{ background: RAMP_CSS_GRADIENT }}
            aria-hidden
          >
            <span className="absolute -bottom-0.5 left-1/2 h-3.5 w-px bg-bg" />
          </span>
          <span className="num">{pct(maxP)}</span>
          <span className="text-ink-mute">P(room), sqrt scale to current max - mid tick {pct(maxP / 4)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="block h-3 w-4 rounded-[2px] border-2 border-[#f1fbe3]" aria-hidden />
          90% credible set
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="16" height="12" aria-hidden>
            <rect width="16" height="12" fill="url(#hatch-cleared)" stroke="#34465a" />
          </svg>
          Cleared
        </div>
        <div className="flex items-center gap-1.5">
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-accent text-[10px] font-bold text-accent-ink" aria-hidden>
            1
          </span>
          Next stop
        </div>
      </div>
    </Panel>
  );
}

function RoomTooltip({
  room,
  p,
  rank,
  inCredible,
  isCleared,
  mark,
  multiTeam,
  hover,
}: {
  room: Room;
  p: number;
  rank: number | undefined;
  inCredible: boolean;
  isCleared: boolean;
  mark: StopMark | undefined;
  multiTeam: boolean;
  hover: Hover;
}) {
  const flip = hover.x > hover.boxW / 2;
  const features: string[] = [];
  features.push(room.has_windows ? `Windows${room.window_facing ? ` facing ${titleCase(room.window_facing).toLowerCase()}` : ""}` : "No windows");
  if (room.near_elevator) features.push("Near elevator");
  if (room.near_stairs) features.push("Near stairs");
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-10 w-56 rounded border border-line-strong bg-bg/95 p-2.5 text-xs shadow-lg shadow-black/40"
      style={{
        left: flip ? undefined : hover.x + 14,
        right: flip ? hover.boxW - hover.x + 14 : undefined,
        top: Math.max(4, hover.y - 20),
      }}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-semibold text-ink">{room.name}</span>
        <span className="num text-[13px] font-semibold text-ink">{pct(p)}</span>
      </div>
      <div className="mt-0.5 text-ink-mute">
        {titleCase(room.room_type)} - Floor {room.floor_label} - <span className="num">{room.area_m2.toFixed(0)} m²</span>
        {rank ? <> - rank <span className="num">#{rank}</span></> : null}
      </div>
      <ul className="mt-1.5 space-y-0.5 text-ink-dim">
        {features.map((f) => (
          <li key={f}>{f}</li>
        ))}
      </ul>
      {inCredible || isCleared || mark ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5 border-t border-line pt-1.5 text-[11px]">
          {mark && !isCleared ? (
            <span className="rounded-sm bg-accent/15 px-1.5 py-0.5 font-medium text-accent">
              {multiTeam ? `Team ${String.fromCharCode(65 + mark.team)} ` : ""}stop {mark.index + 1}
            </span>
          ) : null}
          {inCredible ? <span className="rounded-sm bg-ink/10 px-1.5 py-0.5 text-ink">In 90% set</span> : null}
          {isCleared ? <span className="rounded-sm bg-ink/10 px-1.5 py-0.5 text-ink-dim">Cleared</span> : null}
        </div>
      ) : null}
    </div>
  );
}

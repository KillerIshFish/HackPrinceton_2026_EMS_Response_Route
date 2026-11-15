import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import * as Popover from "@radix-ui/react-popover";
import { Ban, Box, Layers, LocateFixed, LockOpen, Search, X as Close } from "lucide-react";
import type {
  Building,
  FloorPlanImage,
  HazardClass,
  Posterior,
  Room,
  RoomMeta,
  SearchMode,
  SearchPlan,
  SearchedRoom,
} from "../types";
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
import { HAZARD_LABEL, HazardIcon, MODE_LABEL, clockTime, looksByRoom } from "./SearchPlan";

interface Props {
  building: Building;
  posterior: Posterior;
  plan: SearchPlan;
  /** every look so far, oldest first (state.searched) */
  searched: SearchedRoom[];
  /** rooms the officer closed (state.situation.closed_room_ids) */
  closedRoomIds: string[];
  /** rooms to outline: the ones a hovered clue, stop or answer points at */
  outlineRoomIds?: string[] | null;
  /** policy.route_enabled. false: no stop numbers, no route, no search actions */
  routeEnabled: boolean;
  /** policy.show_hazard_class: hazard chips from the plan's stops, for information only */
  showHazardClass?: boolean;
  /** record a search from the room popup */
  onSearched?: (roomId: string, mode: SearchMode) => void;
  /** record the officer closing or reopening a room from the room popup */
  onSetClosed?: (roomId: string, closed: boolean) => void;
  /** a request is in flight: popup actions are disabled */
  busy?: boolean;
  floor: number;
  follow: boolean;
  onSelectFloor: (floor: number) => void;
  onToggleFollow: () => void;
  /** where each room's data comes from (real buildings); shown in the tooltip */
  roomMeta?: Record<string, RoomMeta>;
  /** "Show 3D scan" in the popup of a room that has one */
  onSelectRoom?: (roomId: string) => void;
  /** Matterport's plan image of this floor, drawn under the heatmap (whole-building scans) */
  planImage?: (FloorPlanImage & { href: string }) | null;
  /** counter-clockwise angle of the plan's +x axis from east; 0 means north is up */
  rotationDeg?: number;
}

const SOURCE_LABEL: Record<RoomMeta["source"], string> = {
  matterport: "Matterport scan",
  costar: "CoStar suite record - interior not scanned",
  inferred: "Inferred - not in any dataset",
  floor_plan: "Traced from the building's published floor plan",
};

// The viewBox is the measured size of the drawing area, so one svg unit is one CSS pixel and the
// 10px labels stay 10px whatever the panel size. The plate is scaled to fit both width and height.
const S_MAX = 40; // px per metre at most
const PAD_X = 10;
const PAD_Y = 24; // room for the N / S side labels
const CHAR_W = 5.3; // approx glyph advance at 10px

/** Room name that fits in `w` px: whole, ellipsised, or null when too narrow. */
function fitName(name: string, w: number): string | null {
  const maxChars = Math.floor((w - 8) / CHAR_W);
  if (name.length <= maxChars) return name;
  return maxChars >= 6 ? `${name.slice(0, maxChars - 1).trimEnd()}…` : null;
}
const MARKED_STOPS = 5; // stops per team that get a numbered marker

const C_CLOSED = "#e8c547"; // --color-warn: an officer decision is a status
const C_OUTLINE = "#79a8d8"; // --color-info: rooms another panel points at

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

/** floors beyond this are chosen from a menu instead of a row of tabs */
const MAX_FLOOR_TABS = 8;
const BADGE_R = 7.5;
const BADGE_R_NEXT = 9;

export function FloorPlanHeatmap({
  building,
  posterior,
  plan,
  searched,
  closedRoomIds,
  outlineRoomIds,
  routeEnabled,
  showHazardClass = false,
  onSearched,
  onSetClosed,
  busy = false,
  floor,
  follow,
  onSelectFloor,
  onToggleFollow,
  roomMeta,
  onSelectRoom,
  planImage,
  rotationDeg = 0,
}: Props) {
  const rotated = Math.abs(rotationDeg) > 0.5;
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [popup, setPopup] = useState<Hover | null>(null);

  // measured drawing area
  const [size, setSize] = useState({ w: 700, h: 360 });
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return; // hidden tab
      setSize((s) => (Math.abs(s.w - r.width) < 1 && Math.abs(s.h - r.height) < 1 ? s : { w: r.width, h: r.height }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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

  const spanX = Math.max(1, bounds.maxX - bounds.minX);
  const spanY = Math.max(1, bounds.maxY - bounds.minY);
  const W = size.w;
  const H = size.h;
  const S = Math.max(1, Math.min(S_MAX, (W - PAD_X * 2) / spanX, (H - PAD_Y * 2) / spanY));
  const offX = (W - spanX * S) / 2;
  const offY = (H - spanY * S) / 2;
  const X = (x: number) => (x - bounds.minX) * S + offX;
  const Y = (y: number) => (bounds.maxY - y) * S + offY; // flip: north is up
  // below xl the column has no fixed height: the drawing area keeps the plate's aspect ratio
  const aspect = `${Math.round(spanX * 14 + PAD_X * 2)} / ${Math.round(spanY * 14 + PAD_Y * 2)}`;

  const probs = posterior.room_probs;
  const maxP = useMemo(() => Math.max(0, ...Object.values(probs)), [probs]);
  const credible = useMemo(() => new Set(posterior.credible_set_90), [posterior.credible_set_90]);
  const looks = useMemo(() => looksByRoom(searched), [searched]);
  const closed = useMemo(() => new Set(closedRoomIds), [closedRoomIds]);
  const outlined = useMemo(() => new Set(outlineRoomIds ?? []), [outlineRoomIds]);
  const hazards = useMemo(() => {
    const m = new Map<string, HazardClass>();
    if (showHazardClass) for (const t of plan.teams) for (const s of t) if (s.hazard_class) m.set(s.room_id, s.hazard_class);
    return m;
  }, [plan, showHazardClass]);

  const rank = useMemo(() => {
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    return new Map(sorted.map(([id], i) => [id, i + 1]));
  }, [probs]);

  const stopMarks = useMemo(() => {
    const m = new Map<string, StopMark>();
    if (!routeEnabled) return m; // route planning off: no stop numbers anywhere
    plan.teams.forEach((stops, team) =>
      stops.slice(0, MARKED_STOPS).forEach((s, index) => {
        if (!m.has(s.room_id)) m.set(s.room_id, { team, index });
      }),
    );
    return m;
  }, [plan, routeEnabled]);
  const multiTeam = plan.teams.length > 1;

  // Where each stop badge goes on this floor. A badge sits in its room's top-right corner
  // when the room is big enough to hold it, otherwise on the room's centre. Badges are
  // placed in stop order and one that would cover an earlier badge is left out (the stop is
  // still in the search plan list), so a floor of small rooms never turns into a pile of discs.
  const badges = useMemo(() => {
    const placed: { roomId: string; x: number; y: number; r: number; centred: boolean }[] = [];
    const onFloor = rooms
      .map((room) => ({ room, mark: stopMarks.get(room.id) }))
      .filter((m): m is { room: Room; mark: StopMark } => !!m.mark)
      .sort((a, b) => a.mark.index - b.mark.index || a.mark.team - b.mark.team);
    for (const { room, mark } of onFloor) {
      const b = boundsOf(room.polygon);
      const w = (b.maxX - b.minX) * S;
      const h = (b.maxY - b.minY) * S;
      const r = mark.index === 0 ? BADGE_R_NEXT : BADGE_R;
      const roomy = w >= 4 * r && h >= 4 * r;
      const x = roomy ? (b.maxX - bounds.minX) * S + offX - r - 2 : ((b.minX + b.maxX) / 2 - bounds.minX) * S + offX;
      const y = roomy ? (bounds.maxY - b.maxY) * S + offY + r + 2 : (bounds.maxY - (b.minY + b.maxY) / 2) * S + offY;
      if (placed.some((p) => Math.hypot(p.x - x, p.y - y) < p.r + r + 1.5)) continue;
      placed.push({ roomId: room.id, x, y, r, centred: !roomy });
    }
    return new Map(placed.map((p) => [p.roomId, p]));
  }, [rooms, stopMarks, S, offX, offY, bounds]);

  const midY = (bounds.minY + bounds.maxY) / 2;
  const northFacing = sideFacing(building.rooms, true, midY);
  const southFacing = sideFacing(building.rooms, false, midY);

  const floorLabel = labels.get(floor) ?? String(floor);
  const floorP = posterior.p_floor[floorLabel] ?? 0;
  const hovered = hover && !popup ? rooms.find((r) => r.id === hover.roomId) : undefined;
  const popRoom = popup ? building.rooms.find((r) => r.id === popup.roomId) : undefined;
  const floorSearched = rooms.some((r) => looks.has(r.id));
  const floorClosed = rooms.some((r) => closed.has(r.id));
  const floorOutlined = rooms.some((r) => outlined.has(r.id));

  function relative(clientX: number, clientY: number, roomId: string): Hover | null {
    const box = wrapRef.current?.getBoundingClientRect();
    if (!box) return null;
    return { roomId, x: clientX - box.left, y: clientY - box.top, boxW: box.width };
  }

  function centreOf(el: SVGElement): [number, number] {
    const r = el.getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  }

  const floorsTopDown = [...building.floors].sort((a, b) => b - a);

  return (
    <Panel
      title={`Floor plan - Floor ${floorLabel}`}
      icon={<Layers size={15} aria-hidden />}
      actions={
        <>
          <span className="num text-xs text-ink-dim">P(floor) {pct(floorP)}</span>
          {floorsTopDown.length > MAX_FLOOR_TABS ? (
            <select
              aria-label="Floor"
              value={floor}
              onChange={(e) => onSelectFloor(Number(e.target.value))}
              className="num rounded border border-line bg-bg px-1.5 py-1 text-xs text-ink"
            >
              {floorsTopDown.map((f) => (
                <option key={f} value={f}>
                  Floor {labels.get(f) ?? f}
                </option>
              ))}
            </select>
          ) : null}
          <div
            className={`overflow-hidden rounded border border-line ${floorsTopDown.length > MAX_FLOOR_TABS ? "hidden" : "flex"}`}
            role="tablist"
            aria-label="Floor"
          >
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
      className="xl:h-full"
      bodyClassName="flex flex-col"
    >
      <Popover.Root open={!!popup && !!popRoom} onOpenChange={(o) => (o ? undefined : setPopup(null))}>
        <div
          ref={wrapRef}
          className="relative aspect-(--plate-aspect) w-full xl:aspect-auto xl:min-h-[220px] xl:flex-1"
          style={{ "--plate-aspect": aspect } as CSSProperties}
          onMouseLeave={() => setHover(null)}
        >
          <svg
            viewBox={`0 0 ${W} ${H}`}
            width={W}
            height={H}
            className="absolute inset-0 block select-none"
            role="img"
            aria-label={`Probability heatmap of floor ${floorLabel}. North is up.`}
          >
            <defs>
              {/* searched: two-tone diagonal lines over the probability colour, legible on dark and light fills */}
              <pattern id="hatch-searched" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="1" y1="0" x2="1" y2="9" stroke="#0b1016" strokeOpacity="0.75" strokeWidth="1.8" />
                <line x1="3.2" y1="0" x2="3.2" y2="9" stroke="#e6edf3" strokeOpacity="0.6" strokeWidth="1" />
              </pattern>
            </defs>

            {/* side labels */}
            <text x={W / 2} y={offY - 9} textAnchor="middle" fontSize="9.5" fontWeight="600" letterSpacing="1.1" fill="#8394a6">
              {rotated ? "" : "N"}
              {northFacing ? `${rotated ? "" : " - "}${titleCase(northFacing).toUpperCase()} SIDE` : ""}
            </text>
            <text
              x={W / 2}
              y={offY + spanY * S + 16}
              textAnchor="middle"
              fontSize="9.5"
              fontWeight="600"
              letterSpacing="1.1"
              fill="#8394a6"
            >
              {rotated ? "" : "S"}
              {southFacing ? `${rotated ? "" : " - "}${titleCase(southFacing).toUpperCase()} SIDE` : ""}
            </text>

            {/* the plan is not north-up: show where north is */}
            {rotated ? (
              <g transform={`translate(${W - 20}, 12)`} aria-hidden>
                <g transform={`rotate(${rotationDeg})`}>
                  <path d="M 0 -8 L 4.5 6 L 0 3 L -4.5 6 Z" fill="#8394a6" />
                </g>
                <text x={-10} y={3.5} textAnchor="end" fontSize="9.5" fontWeight="600" fill="#8394a6">
                  N
                </text>
              </g>
            ) : null}

            {/* Matterport's plan image of the floor, in the same coordinates as the rooms */}
            {planImage ? (
              <image
                href={planImage.href}
                x={X(planImage.origin[0])}
                y={Y(planImage.origin[1] + planImage.height / planImage.resolution)}
                width={(planImage.width / planImage.resolution) * S}
                height={(planImage.height / planImage.resolution) * S}
                preserveAspectRatio="none"
                pointerEvents="none"
              />
            ) : null}

            {/* room fills: the probability colour stays on searched rooms */}
            {rooms.map((room) => {
              const p = probs[room.id] ?? 0;
              const t = heatT(p, maxP);
              const pts = room.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
              return (
                <polygon
                  key={room.id}
                  points={pts}
                  fill={rampColor(t)}
                  fillOpacity={planImage ? 0.42 + 0.4 * t : 1}
                  stroke={planImage ? "#e6edf3" : "#0b1016"}
                  strokeOpacity={planImage ? 0.55 : 1}
                  strokeWidth={planImage ? 1 : 1.5}
                  className="room-shape cursor-pointer outline-none"
                  tabIndex={0}
                  role="button"
                  aria-haspopup="dialog"
                  aria-label={`${room.name}, ${titleCase(room.room_type)}, probability ${pct(p)}${
                    looks.has(room.id) ? ", searched" : ""
                  }${closed.has(room.id) ? ", closed by officer" : ""}`}
                  onMouseMove={(e) => setHover(relative(e.clientX, e.clientY, room.id))}
                  onFocus={(e) => setHover(relative(...centreOf(e.currentTarget), room.id))}
                  onBlur={() => setHover(null)}
                  onClick={(e) => setPopup(relative(e.clientX, e.clientY, room.id))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setPopup(relative(...centreOf(e.currentTarget), room.id));
                    }
                  }}
                />
              );
            })}

            {/* searched hatch, 90% set, closed by officer, outlined rooms, hover ring (above fills) */}
            {rooms.map((room) => {
              const pts = room.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
              const isSearched = looks.has(room.id);
              const isCred = credible.has(room.id);
              const isClosed = closed.has(room.id);
              const isOutlined = outlined.has(room.id);
              const isHover = hover?.roomId === room.id || popup?.roomId === room.id;
              if (!isSearched && !isCred && !isClosed && !isOutlined && !isHover) return null;
              return (
                <g key={room.id} pointerEvents="none">
                  {isSearched ? <polygon points={pts} fill="url(#hatch-searched)" /> : null}
                  {isCred ? <polygon points={pts} fill="none" stroke="#f1fbe3" strokeWidth={2.6} strokeLinejoin="round" /> : null}
                  {isClosed ? (
                    <>
                      <polygon points={pts} fill="none" stroke="#0b1016" strokeWidth={5} strokeLinejoin="round" />
                      <polygon points={pts} fill="none" stroke={C_CLOSED} strokeWidth={2.6} strokeDasharray="8 4" strokeLinejoin="round" />
                    </>
                  ) : null}
                  {isOutlined ? <polygon points={pts} fill="none" stroke={C_OUTLINE} strokeWidth={3.5} strokeLinejoin="round" /> : null}
                  {isHover ? <polygon points={pts} fill="none" stroke="#e6edf3" strokeWidth={2} strokeDasharray="2 3" /> : null}
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
              const isSearched = looks.has(room.id);
              const isClosed = closed.has(room.id);
              const ink = rampInk(t);
              // a searched room always shows what remains, however small: "0.7% left", or "0.7%" when narrow
              const fitsText = (s: string) => w >= s.length * 7 + 6;
              const pctText = isSearched && fitsText(`${pct(p)} left`) ? `${pct(p)} left` : pct(p);
              const pctFits = fitsText(pctText) && h >= 14 && (p >= 0.001 || isSearched);
              // text over the searched hatch gets a halo in the room's own colour
              const halo = isSearched ? { stroke: rampColor(t), strokeWidth: 3, paintOrder: "stroke" as const, strokeLinejoin: "round" as const } : {};
              const shownName = h >= (pctFits ? 30 : 14) ? fitName(room.name, w) : null;
              const nameFits = shownName !== null;
              const badge = badges.get(room.id);
              const mark = badge ? stopMarks.get(room.id) : undefined;
              const isNext = mark?.index === 0;
              const mx = badge?.x ?? 0;
              const my = badge?.y ?? 0;
              // a badge centred on a small room takes the place of that room's text
              const textHidden = !!badge?.centred;
              const markLabel = mark ? `${multiTeam ? String.fromCharCode(65 + mark.team) : ""}${mark.index + 1}` : "";
              return (
                <g key={room.id} pointerEvents="none">
                  {nameFits && !textHidden ? (
                    <text x={cx} y={pctFits ? cy - 3 : cy + 3.5} textAnchor="middle" fontSize="10" fill={ink} fillOpacity={0.92} {...halo}>
                      {shownName}
                    </text>
                  ) : null}
                  {pctFits && !textHidden ? (
                    <text
                      x={cx}
                      y={nameFits ? cy + 12 : cy + 4}
                      textAnchor="middle"
                      fontSize={nameFits ? 11.5 : 10.5}
                      fontWeight="700"
                      fill={ink}
                      {...halo}
                      style={{ fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums" }}
                    >
                      {pctText}
                    </text>
                  ) : null}
                  {isClosed && w >= 48 && h >= 18 ? (
                    <g>
                      <rect x={X(b.minX) + 4} y={Y(b.maxY) + 4} width={42} height={12} rx={1.5} fill="#0b1016" />
                      <text
                        x={X(b.minX) + 25}
                        y={Y(b.maxY) + 13}
                        textAnchor="middle"
                        fontSize="8"
                        fontWeight="700"
                        letterSpacing="0.6"
                        fill={C_CLOSED}
                      >
                        CLOSED
                      </text>
                    </g>
                  ) : null}
                  {mark ? (
                    <g>
                      {isNext ? <circle cx={mx} cy={my} r={BADGE_R_NEXT} fill="#f2a93b" className="next-stop-ring" /> : null}
                      <circle
                        cx={mx}
                        cy={my}
                        r={isNext ? BADGE_R_NEXT : BADGE_R}
                        fill={isNext ? "#f2a93b" : "#0b1016"}
                        stroke={isNext ? "#0b1016" : "#f2a93b"}
                        strokeWidth={1.5}
                      />
                      <text
                        x={mx}
                        y={my + (isNext ? 3.7 : 3.2)}
                        textAnchor="middle"
                        fontSize={markLabel.length > 1 ? 8 : isNext ? 10.5 : 9}
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
              looks={looks.get(hovered.id) ?? []}
              isClosed={closed.has(hovered.id)}
              hazard={hazards.get(hovered.id)}
              mark={stopMarks.get(hovered.id)}
              multiTeam={multiTeam}
              hover={hover}
              meta={roomMeta?.[hovered.id]}
            />
          ) : null}

          {popup ? (
            <Popover.Anchor asChild>
              <span className="pointer-events-none absolute h-px w-px" style={{ left: popup.x, top: popup.y }} aria-hidden />
            </Popover.Anchor>
          ) : null}
        </div>

        {popRoom ? (
          <Popover.Portal>
            <Popover.Content
              side="right"
              align="start"
              sideOffset={10}
              collisionPadding={8}
              className="z-50 w-64 rounded border border-line-strong bg-bg p-2.5 text-xs shadow-lg shadow-black/50"
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              <RoomPopup
                room={popRoom}
                p={probs[popRoom.id] ?? 0}
                looks={looks.get(popRoom.id) ?? []}
                isClosed={closed.has(popRoom.id)}
                hazard={hazards.get(popRoom.id)}
                routeEnabled={routeEnabled}
                busy={busy}
                hasScan={!!roomMeta?.[popRoom.id]?.model_id && !!onSelectRoom}
                onSearched={onSearched ? (mode) => onSearched(popRoom.id, mode) : undefined}
                onSetClosed={onSetClosed ? (c) => onSetClosed(popRoom.id, c) : undefined}
                onShowScan={() => {
                  onSelectRoom?.(popRoom.id);
                  setPopup(null);
                }}
              />
              <Popover.Close
                aria-label="Close"
                className="absolute right-1.5 top-1.5 rounded p-0.5 text-ink-mute hover:bg-raised hover:text-ink"
              >
                <Close size={13} aria-hidden />
              </Popover.Close>
            </Popover.Content>
          </Popover.Portal>
        ) : null}
      </Popover.Root>

      {/* legend: each mark has a swatch and a word, never colour alone */}
      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-line px-3 py-1.5 text-xs text-ink-dim">
        <div className="flex items-center gap-2">
          <span className="num">0</span>
          <span className="relative block h-2.5 w-28 rounded-sm" style={{ background: RAMP_CSS_GRADIENT }} aria-hidden>
            <span className="absolute -bottom-0.5 left-1/2 h-3.5 w-px bg-bg" />
          </span>
          <span className="num">{pct(maxP)}</span>
          <span className="text-ink-mute" title={`Square-root scale to the current maximum; the mid tick is ${pct(maxP / 4)}`}>
            P(room)
          </span>
        </div>
        <LegendItem swatch={<span className="block h-3 w-4 rounded-[2px] border-2 border-[#f1fbe3]" />}>90% set</LegendItem>
        <LegendItem
          swatch={
            <svg width="16" height="12">
              <rect width="16" height="12" fill={rampColor(0.5)} />
              <rect width="16" height="12" fill="url(#hatch-searched)" stroke="#34465a" />
            </svg>
          }
          dim={!floorSearched}
        >
          Searched, % left
        </LegendItem>
        <LegendItem
          swatch={
            <svg width="16" height="12">
              <rect x="1.5" y="1.5" width="13" height="9" fill="none" stroke={C_CLOSED} strokeWidth="2" strokeDasharray="4 2" />
            </svg>
          }
          dim={!floorClosed}
        >
          Closed by officer
        </LegendItem>
        {floorOutlined ? (
          <LegendItem swatch={<span className="block h-3 w-4 rounded-[2px] border-2" style={{ borderColor: C_OUTLINE }} />}>
            Pointed at
          </LegendItem>
        ) : null}
        {routeEnabled ? (
          <LegendItem
            swatch={
              <span className="flex h-4 w-4 items-center justify-center rounded-full bg-accent text-[10px] font-bold text-accent-ink">1</span>
            }
          >
            Next stop
          </LegendItem>
        ) : null}
        {planImage ? <div className="text-ink-mute">Floor plan: Matterport scan, rooms traced from its walls</div> : null}
        <div className="text-ink-mute">Click a room for actions</div>
        {showHazardClass && hazards.size > 0 ? (
          <div className="basis-full text-ink-mute">
            Hazard labels: shown for information. A closed door stops protecting once it is opened.
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

function LegendItem({ swatch, dim = false, children }: { swatch: ReactNode; dim?: boolean; children: ReactNode }) {
  return (
    <div className={`flex items-center gap-1.5 ${dim ? "opacity-60" : ""}`}>
      <span aria-hidden className="flex">
        {swatch}
      </span>
      {children}
    </div>
  );
}

function RoomPopup({
  room,
  p,
  looks,
  isClosed,
  hazard,
  routeEnabled,
  busy,
  hasScan,
  onSearched,
  onSetClosed,
  onShowScan,
}: {
  room: Room;
  p: number;
  looks: SearchedRoom[];
  isClosed: boolean;
  hazard: HazardClass | undefined;
  routeEnabled: boolean;
  busy: boolean;
  hasScan: boolean;
  onSearched?: (mode: SearchMode) => void;
  onSetClosed?: (closed: boolean) => void;
  onShowScan: () => void;
}) {
  const btn =
    "flex items-center justify-center gap-1 rounded border border-line-strong px-2 py-1 text-[11px] font-medium text-ink-dim hover:border-ink-dim hover:text-ink disabled:cursor-not-allowed disabled:opacity-50";
  return (
    <div>
      <div className="pr-5 text-[13px] font-semibold text-ink">{room.name}</div>
      <div className="num mt-0.5 text-ink-mute">
        Floor {room.floor_label} - {looks.length > 0 ? `${pct(p)} left` : `P ${pct(p)}`}
      </div>
      {looks.length > 0 ? (
        <div className="num mt-1 text-[11px] text-ink-dim">
          Searched: {looks.map((l) => `${MODE_LABEL[l.mode]} ${clockTime(l.at)}`).join(", ")}
        </div>
      ) : null}
      {hazard ? (
        <div className="mt-1 flex items-center gap-1 text-[11px] text-ink">
          <HazardIcon hazard={hazard} />
          {HAZARD_LABEL[hazard]} <span className="text-ink-mute">(for information)</span>
        </div>
      ) : null}

      {routeEnabled && onSearched ? (
        <div className="mt-2 border-t border-line pt-2">
          <div className="mb-1 text-[10.5px] font-semibold uppercase tracking-wider text-ink-mute">Crew searched it</div>
          <div className="grid grid-cols-2 gap-1.5">
            {(["rapid", "thorough"] as const).map((mode) => (
              <button key={mode} className={btn} disabled={busy} onClick={() => onSearched(mode)}>
                <Search size={11} aria-hidden />
                {MODE_LABEL[mode]}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {onSetClosed ? (
        <div className="mt-2 border-t border-line pt-2">
          <button
            className={`${btn} w-full ${isClosed ? "border-warn/70 text-warn" : ""}`}
            disabled={busy}
            aria-pressed={isClosed}
            onClick={() => onSetClosed(!isClosed)}
          >
            {isClosed ? <LockOpen size={11} aria-hidden /> : <Ban size={11} aria-hidden />}
            {isClosed ? "Reopen (officer's decision)" : "Closed by officer"}
          </button>
          <p className="mt-1 text-[10.5px] leading-snug text-ink-mute">
            {isClosed
              ? "Closed on the officer's word. Reopening puts it back in the plan."
              : "Records the officer's decision. The plan leaves the room out until reopened."}
          </p>
        </div>
      ) : null}

      {hasScan ? (
        <button className={`${btn} mt-2 w-full`} onClick={onShowScan}>
          <Box size={11} aria-hidden />
          Show 3D scan
        </button>
      ) : null}
    </div>
  );
}

function RoomTooltip({
  room,
  p,
  rank,
  inCredible,
  looks,
  isClosed,
  hazard,
  mark,
  multiTeam,
  hover,
  meta,
}: {
  room: Room;
  p: number;
  rank: number | undefined;
  inCredible: boolean;
  looks: SearchedRoom[];
  isClosed: boolean;
  hazard: HazardClass | undefined;
  mark: StopMark | undefined;
  multiTeam: boolean;
  hover: Hover;
  meta?: RoomMeta;
}) {
  const flip = hover.x > hover.boxW / 2;
  const features: string[] = [];
  features.push(room.has_windows ? `Windows${room.window_facing ? ` facing ${titleCase(room.window_facing).toLowerCase()}` : ""}` : "No windows");
  if (room.near_elevator) features.push("Near elevator");
  if (room.near_stairs) features.push("Near stairs");
  const searched = looks.length > 0;
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
        <span className="num shrink-0 text-[13px] font-semibold text-ink">{searched ? `${pct(p)} left` : pct(p)}</span>
      </div>
      <div className="mt-0.5 text-ink-mute">
        {titleCase(room.room_type)} - Floor {room.floor_label} - <span className="num">{room.area_m2.toFixed(0)} m²</span>
        {rank ? <> - rank <span className="num">#{rank}</span></> : null}
      </div>
      <ul className="mt-1.5 space-y-0.5 text-ink-dim">
        {features.map((f) => (
          <li key={f}>{f}</li>
        ))}
        {meta?.tenant ? <li>Tenant: {meta.tenant}</li> : meta?.status === "available" ? <li>Vacant, marketed for lease</li> : null}
      </ul>
      {searched ? (
        <div className="num mt-1.5 border-t border-line pt-1.5 text-[11px] text-ink-dim">
          Searched: {looks.map((l) => `${MODE_LABEL[l.mode]} ${clockTime(l.at)}`).join(", ")}. One pass can miss a person.
        </div>
      ) : null}
      {meta ? <div className="mt-1.5 border-t border-line pt-1.5 text-[11px] text-ink-mute">{SOURCE_LABEL[meta.source]}</div> : null}
      {inCredible || isClosed || mark || hazard ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5 border-t border-line pt-1.5 text-[11px]">
          {mark ? (
            <span className="rounded-sm bg-accent/15 px-1.5 py-0.5 font-medium text-accent">
              {multiTeam ? `Team ${String.fromCharCode(65 + mark.team)} ` : ""}stop {mark.index + 1}
            </span>
          ) : null}
          {inCredible ? <span className="rounded-sm bg-ink/10 px-1.5 py-0.5 text-ink">In 90% set</span> : null}
          {isClosed ? (
            <span className="flex items-center gap-1 rounded-sm border border-warn/60 px-1.5 py-0.5 font-medium text-warn">
              <Ban size={10} aria-hidden />
              Closed by officer
            </span>
          ) : null}
          {hazard ? (
            <span className="flex items-center gap-1 rounded-sm border border-line-strong px-1.5 py-0.5 text-ink">
              <HazardIcon hazard={hazard} size={10} />
              {HAZARD_LABEL[hazard]}
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="mt-1.5 text-[10.5px] text-ink-mute">Click for actions</div>
    </div>
  );
}

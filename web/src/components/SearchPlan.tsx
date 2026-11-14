import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  Ban,
  Box,
  Check,
  ChevronDown,
  DoorClosed,
  Flame,
  Info,
  ListOrdered,
  LockOpen,
  MapPin,
  Repeat2,
  Search,
  Undo2,
  Wind,
} from "lucide-react";
import type {
  Building,
  DemoState,
  Gates,
  HazardClass,
  Room,
  SearchMode,
  SearchPlan as SearchPlanData,
  SearchedRoom,
  Stop,
  TravelProfile,
} from "../types";
import type { ConsoleActions } from "../App";
import { mmss, pct, roomIndex } from "../lib";
import { Panel } from "./Panel";
import { WidenSignal } from "./WidenSignal";

// ---------------------------------------------------------------------------
// Shared with FloorPlanHeatmap
// ---------------------------------------------------------------------------

export const MODE_LABEL: Record<SearchMode, string> = { rapid: "Rapid", thorough: "Thorough" };

/** Display only: the class never changes the order (labels fixed by docs/BUILD_PLAN.md D4). */
export const HAZARD_LABEL: Record<HazardClass, string> = {
  fire_room: "Fire room",
  open_path: "Open to the fire",
  closed_door: "Behind a closed door",
};

export function HazardIcon({ hazard, size = 11 }: { hazard: HazardClass; size?: number }) {
  if (hazard === "fire_room") return <Flame size={size} aria-hidden />;
  if (hazard === "closed_door") return <DoorClosed size={size} aria-hidden />;
  return <Wind size={size} aria-hidden />;
}

/** Every look at one room, oldest first. */
export type Looks = Map<string, SearchedRoom[]>;

export function looksByRoom(searched: SearchedRoom[]): Looks {
  const m: Looks = new Map();
  for (const s of searched) {
    const list = m.get(s.room_id);
    if (list) list.push(s);
    else m.set(s.room_id, [s]);
  }
  return m;
}

/** "19:09" in the dispatcher's local time; "-" when unknown. */
export function clockTime(iso: string | null | undefined): string {
  if (!iso) return "-";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "-";
  return new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

/** "+7:35" from the start of the call; "" when either time is unknown. */
export function sinceCall(iso: string | null | undefined, callStartedAt: string | null | undefined): string {
  if (!iso || !callStartedAt) return "";
  const d = (Date.parse(iso) - Date.parse(callStartedAt)) / 1000;
  return Number.isFinite(d) ? `+${mmss(d)}` : "";
}

/** "Rapid 19:09, Thorough 19:12" */
export function looksText(looks: SearchedRoom[]): string {
  return looks.map((l) => `${MODE_LABEL[l.mode]} ${clockTime(l.at)}`).join(", ");
}

/** The gates body that closes (or reopens) one room and keeps everything else the officer set. */
export function gatesWithRoom(situation: DemoState["situation"], roomId: string, closed: boolean): Gates {
  const ids = situation.closed_room_ids.filter((id) => id !== roomId);
  return { closed_room_ids: closed ? [...ids, roomId] : ids, closed_entrance_ids: situation.closed_entrance_ids };
}

// ---------------------------------------------------------------------------

type PlanActions = Pick<ConsoleActions, "searched" | "found" | "undo" | "setGates" | "timestamp">;

interface Props {
  building: Building;
  state: DemoState;
  /** the request in flight (App's busy key, e.g. "searched:<room id>"), or null */
  busy: string | null;
  actions: PlanActions;
  /** hovering a stop outlines its room on the floor plan */
  onHoverRooms?: (roomIds: string[] | null) => void;
  /** rooms that have a 3D scan; those stops get a button that opens it */
  scannedRoomIds?: Set<string>;
  onViewScan?: (roomId: string) => void;
}

const VISIBLE_STOPS = 8;

/**
 * The search plan: what to look at next, what has been looked at, and what the officer closed.
 * Every control records something the crew or the officer reported; nothing here closes a room or
 * ends a search by itself. Renders nothing when the policy turns route planning off.
 */
export function SearchPlan({ building, state, busy, actions, onHoverRooms, scannedRoomIds, onViewScan }: Props) {
  const { plan, posterior, policy, situation } = state;
  const rooms = useMemo(() => roomIndex(building), [building]);
  const looks = useMemo(() => looksByRoom(state.searched), [state.searched]);
  const closed = useMemo(() => new Set(situation.closed_room_ids), [situation.closed_room_ids]);
  const seenAt = useFoundTime(state.found_room_id);
  const foundAt = state.found_at ?? seenAt;

  if (!policy.route_enabled) return null;

  const multiTeam = plan.teams.length > 1;
  const totalStops = plan.teams.reduce((n, t) => n + t.length, 0);
  const anyBusy = busy !== null;
  const found = state.found_room_id ? (rooms.get(state.found_room_id) ?? null) : null;

  const setClosed = (roomId: string, close: boolean) => actions.setGates(gatesWithRoom(situation, roomId, close));

  return (
    <Tooltip.Provider delayDuration={250} skipDelayDuration={100}>
      <Panel
        title={found ? "Search ended" : "Search plan"}
        icon={<ListOrdered size={15} aria-hidden />}
        actions={
          <>
            {!found ? (
              <span className="num text-xs text-ink-mute">
                {totalStops} stop{totalStops === 1 ? "" : "s"}
              </span>
            ) : null}
            <button
              onClick={actions.undo}
              disabled={!state.can_undo || anyBusy}
              title={
                state.can_undo
                  ? "Undo the last Searched, Found here or Closed by officer entry"
                  : "Nothing to undo"
              }
              className="flex items-center gap-1 rounded border border-line-strong px-2 py-1 text-[11px] font-medium text-ink-dim hover:border-ink-dim hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Undo2 size={12} aria-hidden />
              {busy === "undo" ? "Undoing…" : "Undo last"}
            </button>
          </>
        }
        className="h-full"
        bodyClassName="flex flex-col"
      >
        {!found ? <WidenSignal plan={plan} posterior={posterior} historyKey={`${state.building_id}|${state.scenario_id ?? ""}`} /> : null}

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {found ? (
            <FoundSummary
              room={found}
              foundAt={foundAt}
              callStartedAt={situation.call_started_at}
              contactAt={situation.timestamps.patient_contact ?? null}
              busy={anyBusy}
              logging={busy === "timestamp"}
              onLogContact={() => actions.timestamp("patient_contact")}
            />
          ) : (
            <>
              {policy.travel.notes.length > 0 ? <TravelNotes notes={policy.travel.notes} /> : null}
              {totalStops === 0 ? (
                <p className="p-3 text-xs text-ink-mute">The plan has no further rooms to look at.</p>
              ) : (
                plan.teams.map((stops, team) => (
                  <TeamList
                    key={team}
                    team={team}
                    stops={stops}
                    multiTeam={multiTeam}
                    building={building}
                    rooms={rooms}
                    plan={plan}
                    travel={policy.travel}
                    roomProbs={posterior.room_probs}
                    looks={looks}
                    closed={closed}
                    showHazard={policy.show_hazard_class}
                    busy={busy}
                    actions={actions}
                    onSetClosed={setClosed}
                    onHoverRooms={onHoverRooms}
                    scannedRoomIds={scannedRoomIds}
                    onViewScan={onViewScan}
                  />
                ))
              )}
            </>
          )}

          <SearchedList
            looks={looks}
            rooms={rooms}
            roomProbs={posterior.room_probs}
            callStartedAt={situation.call_started_at}
            onHoverRooms={onHoverRooms}
          />
          <ClosedList
            closedIds={situation.closed_room_ids}
            rooms={rooms}
            busy={anyBusy}
            onReopen={(id) => setClosed(id, false)}
            onHoverRooms={onHoverRooms}
          />
          {!found ? <PlanFooter plan={plan} /> : null}
        </div>
      </Panel>
    </Tooltip.Provider>
  );
}

/** When this browser first saw the current found_room_id: the fallback when the state has no found_at. */
function useFoundTime(foundRoomId: string | null): string | null {
  const [seen, setSeen] = useState<{ id: string | null; at: string | null }>({ id: foundRoomId, at: null });
  if (seen.id !== foundRoomId) setSeen({ id: foundRoomId, at: foundRoomId ? new Date().toISOString() : null });
  return seen.at;
}

// ---------------------------------------------------------------------------

function TravelNotes({ notes }: { notes: string[] }) {
  return (
    <div className="flex gap-2 border-b border-line bg-info/[0.07] px-3 py-1.5 text-[11px] leading-snug text-ink-dim">
      <Info size={13} className="mt-0.5 shrink-0 text-info" aria-hidden />
      <ul className="space-y-0.5">
        {notes.map((n) => (
          <li key={n}>{n}</li>
        ))}
      </ul>
    </div>
  );
}

function TeamList({
  team,
  stops,
  multiTeam,
  building,
  rooms,
  plan,
  travel,
  roomProbs,
  looks,
  closed,
  showHazard,
  busy,
  actions,
  onSetClosed,
  onHoverRooms,
  scannedRoomIds,
  onViewScan,
}: {
  team: number;
  stops: Stop[];
  multiTeam: boolean;
  building: Building;
  rooms: Map<string, Room>;
  plan: SearchPlanData;
  travel: TravelProfile;
  roomProbs: Record<string, number>;
  looks: Looks;
  closed: Set<string>;
  showHazard: boolean;
  busy: string | null;
  actions: PlanActions;
  onSetClosed: (roomId: string, close: boolean) => void;
  onHoverRooms?: (roomIds: string[] | null) => void;
  scannedRoomIds?: Set<string>;
  onViewScan?: (roomId: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? stops : stops.slice(0, VISIBLE_STOPS);
  const hidden = stops.length - visible.length;
  const teamName = `Team ${String.fromCharCode(65 + team)}`;
  const startFloor = useMemo(() => {
    const ent = building.entrances.find((e) => e.id === plan.entrance_id) ?? building.entrances[0];
    return ent ? (rooms.get(ent.room_id)?.floor ?? null) : null;
  }, [building, plan.entrance_id, rooms]);

  return (
    <div className="border-b border-line last:border-b-0">
      {multiTeam ? (
        <div className="flex items-baseline justify-between px-3 pt-2.5">
          <span className="eyebrow">{teamName}</span>
          <span className="num text-[11px] text-ink-mute">{stops.length} stops</span>
        </div>
      ) : null}
      {stops.length === 0 ? (
        <p className="px-3 py-2.5 text-xs text-ink-mute">No stops assigned.</p>
      ) : (
        <ol className="divide-y divide-line/70">
          {visible.map((stop, i) => {
            const prevFloor = i === 0 ? startFloor : (rooms.get(stops[i - 1]!.room_id)?.floor ?? null);
            return (
              <StopRow
                key={`${stop.room_id}-${stop.kind}-${i}`}
                stop={stop}
                index={i}
                isNext={i === 0}
                fromFloor={prevFloor}
                room={rooms.get(stop.room_id)}
                travel={travel}
                residual={roomProbs[stop.room_id] ?? 0}
                looks={looks.get(stop.room_id) ?? []}
                isClosed={closed.has(stop.room_id)}
                showHazard={showHazard}
                busy={busy}
                actions={actions}
                onSetClosed={onSetClosed}
                onHoverRooms={onHoverRooms}
                hasScan={!!onViewScan && !!scannedRoomIds?.has(stop.room_id)}
                onViewScan={onViewScan}
              />
            );
          })}
        </ol>
      )}
      {stops.length > VISIBLE_STOPS ? (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="w-full border-t border-line/70 px-3 py-1.5 text-left text-[11px] text-ink-dim hover:bg-raised hover:text-ink"
        >
          {showAll ? `Show first ${VISIBLE_STOPS}` : `Show all ${stops.length} stops (+${hidden})`}
        </button>
      ) : null}
    </div>
  );
}

/** Stair time between two floors with the profile in use (the planner's own rates). */
function stairSeconds(travel: TravelProfile, from: number, to: number): number {
  let s = 0;
  const lo = Math.min(from, to);
  const hi = Math.max(from, to);
  for (let f = lo; f < hi; f++) {
    const high = travel.high_floor_from !== null && f + 1 >= travel.high_floor_from;
    s += high ? travel.stair_s_per_floor_high : travel.stair_s_per_floor;
  }
  return s;
}

function StopRow({
  stop,
  index,
  isNext,
  fromFloor,
  room,
  travel,
  residual,
  looks,
  isClosed,
  showHazard,
  busy,
  actions,
  onSetClosed,
  onHoverRooms,
  hasScan,
  onViewScan,
}: {
  stop: Stop;
  index: number;
  isNext: boolean;
  fromFloor: number | null;
  room: Room | undefined;
  travel: TravelProfile;
  residual: number;
  looks: SearchedRoom[];
  isClosed: boolean;
  showHazard: boolean;
  busy: string | null;
  actions: PlanActions;
  onSetClosed: (roomId: string, close: boolean) => void;
  onHoverRooms?: (roomIds: string[] | null) => void;
  hasScan: boolean;
  onViewScan?: (roomId: string) => void;
}) {
  const anyBusy = busy !== null;
  const recording = busy === `searched:${stop.room_id}`;
  const finding = busy === `found:${stop.room_id}`;
  const reSearch = stop.kind === "re_search";
  const lastLook = looks[looks.length - 1];

  // time breakdown: the planner reports one travel leg; the stair share is estimated from the floors
  const travelS = stop.travel_s;
  const stairsS =
    travelS !== null && fromFloor !== null && room ? Math.min(travelS, stairSeconds(travel, fromFloor, room.floor)) : null;
  const walkS = travelS !== null ? Math.max(0, travelS - (stairsS ?? 0)) : null;

  const reason = reSearch
    ? `${pct(residual)} still likely after ${
        looks.length > 1 ? `${looks.length} searches` : `a ${(lastLook?.mode ?? "rapid") === "rapid" ? "rapid" : "thorough"} search`
      }`
    : null;

  return (
    <li
      className={`px-3 py-2 ${isNext ? "bg-accent/[0.06]" : ""} ${isClosed ? "opacity-60" : ""}`}
      onMouseEnter={() => onHoverRooms?.([stop.room_id])}
      onMouseLeave={() => onHoverRooms?.(null)}
    >
      <div className="flex items-center gap-2.5">
        <span
          className={`num flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
            isNext ? "bg-accent text-accent-ink" : "border border-line-strong text-ink-dim"
          }`}
          aria-label={`Stop ${index + 1}`}
        >
          {index + 1}
        </span>
        <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span className="min-w-0 truncate text-[13px] font-medium text-ink" title={stop.room_name}>
            {stop.room_name}
          </span>
          <span className="num shrink-0 rounded-sm border border-line px-1 text-[10px] text-ink-dim">F{stop.floor_label}</span>
          {isNext ? <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-accent">Next</span> : null}
        </div>
        {hasScan ? (
          <button
            onClick={() => onViewScan?.(stop.room_id)}
            aria-label={`Show the 3D scan of ${stop.room_name}`}
            title="Show this room in the 3D scan"
            className="flex shrink-0 items-center gap-1 rounded border border-line-strong px-1.5 py-0.5 text-[11px] font-medium text-ink-dim hover:border-ink-dim hover:text-ink"
          >
            <Box size={12} aria-hidden />
            3D
          </button>
        ) : null}
      </div>

      {reSearch || isClosed || (showHazard && stop.hazard_class) ? (
        <div className="mt-1 flex flex-wrap items-center gap-1.5 pl-[34px]">
          {reSearch ? (
            <span className="flex items-center gap-1 rounded-sm border border-info/50 bg-info/10 px-1.5 py-px text-[10.5px] font-semibold text-info">
              <Repeat2 size={11} aria-hidden />
              Second look
            </span>
          ) : null}
          {isClosed ? (
            <span className="flex items-center gap-1 rounded-sm border border-warn/60 px-1.5 py-px text-[10.5px] font-semibold text-warn">
              <Ban size={11} aria-hidden />
              Closed by officer
            </span>
          ) : null}
          {showHazard && stop.hazard_class ? (
            <span
              className="flex items-center gap-1 rounded-sm border border-line-strong px-1.5 py-px text-[10.5px] text-ink"
              title="Shown for information. A closed door stops protecting once it is opened."
            >
              <HazardIcon hazard={stop.hazard_class} />
              {HAZARD_LABEL[stop.hazard_class]}
            </span>
          ) : null}
          {reason ? <span className="text-[11px] text-ink-dim">{reason}</span> : null}
        </div>
      ) : null}

      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 pl-[34px]">
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <span
              tabIndex={0}
              aria-label={`Time breakdown for ${stop.room_name}`}
              className="num flex cursor-help gap-2.5 whitespace-nowrap rounded-sm text-[11px] text-ink-mute decoration-dotted underline-offset-2 hover:underline"
            >
              <span>
                ETA <span className="text-ink-dim">{mmss(stop.eta_s)}</span>
              </span>
              {stop.p_find !== null ? (
                <span title="Chance this stop is the one that finds the caller">
                  finds <span className="text-ink-dim">{pct(stop.p_find)}</span>
                </span>
              ) : null}
            </span>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content
              side="left"
              align="start"
              sideOffset={8}
              collisionPadding={8}
              className="z-50 w-64 rounded border border-line-strong bg-bg p-2.5 text-xs text-ink-dim shadow-lg shadow-black/50"
            >
              <TimeBreakdown stop={stop} walkS={walkS} stairsS={stairsS} />
              <Tooltip.Arrow className="fill-line-strong" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>

        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          <div className="flex">
            <button
              onClick={() => actions.searched(stop.room_id, stop.mode)}
              disabled={anyBusy}
              title={`Record a ${stop.mode} search of ${stop.room_name}`}
              className={`flex items-center gap-1 rounded-l border px-2 py-0.5 text-[11px] font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
                isNext
                  ? "border-accent/70 text-accent hover:bg-accent hover:text-accent-ink"
                  : "border-line-strong text-ink-dim hover:border-ink-dim hover:text-ink"
              }`}
            >
              <Search size={11} aria-hidden />
              {recording ? "Saving…" : "Searched"}
            </button>
            <DropdownMenu.Root modal={false}>
              <DropdownMenu.Trigger
                disabled={anyBusy}
                aria-label={`More for ${stop.room_name}: search mode, closed by officer`}
                className={`rounded-r border border-l-0 px-1 py-0.5 disabled:cursor-not-allowed disabled:opacity-50 data-[state=open]:bg-raised ${
                  isNext
                    ? "border-accent/70 text-accent hover:bg-accent hover:text-accent-ink"
                    : "border-line-strong text-ink-dim hover:border-ink-dim hover:text-ink"
                }`}
              >
                <ChevronDown size={12} aria-hidden />
              </DropdownMenu.Trigger>
              <StopMenu
                stop={stop}
                isClosed={isClosed}
                onSearched={(mode) => actions.searched(stop.room_id, mode)}
                onSetClosed={(c) => onSetClosed(stop.room_id, c)}
              />
            </DropdownMenu.Root>
          </div>
          <button
            onClick={() => actions.found(stop.room_id)}
            disabled={anyBusy}
            title={`Record that the caller was found in ${stop.room_name}. This ends the search; Undo reverses it.`}
            className="flex items-center gap-1 rounded border border-line-strong px-2 py-0.5 text-[11px] font-medium text-ink-dim hover:border-ink hover:bg-ink hover:text-bg disabled:cursor-not-allowed disabled:opacity-50"
          >
            <MapPin size={11} aria-hidden />
            {finding ? "Saving…" : "Found here"}
          </button>
        </div>
      </div>
    </li>
  );
}

function TimeBreakdown({ stop, walkS, stairsS }: { stop: Stop; walkS: number | null; stairsS: number | null }) {
  const rows: [string, number | null, string?][] = [
    ["Walk", walkS],
    ["Stairs", stairsS, "estimated from the floors crossed"],
    ["Search", stop.search_s, "unverified setting"],
  ];
  const done = stop.search_s !== null ? stop.eta_s + stop.search_s : null;
  return (
    <div>
      <div className="mb-1.5 font-semibold text-ink">{stop.room_name}: this leg</div>
      {stop.travel_s === null && stop.search_s === null ? (
        <p>The planner did not report a time breakdown for this stop.</p>
      ) : (
        <table className="w-full">
          <tbody>
            {rows.map(([label, s, note]) => (
              <tr key={label}>
                <td className="py-px pr-2 text-ink-dim">{label}</td>
                <td className="num py-px pr-2 text-right text-ink">{s === null ? "-" : mmss(s)}</td>
                <td className="py-px text-[10.5px] text-ink-mute">{note ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="num mt-1.5 border-t border-line pt-1.5 text-[11px]">
        Arrives {mmss(stop.eta_s)} from now{done !== null ? `, done by ${mmss(done)}` : ""}
        {stop.mode === "thorough" ? " (thorough)" : ""}
        <div className="text-ink-mute">
          {stop.p_find !== null ? `Finds the caller here: ${pct(stop.p_find)}. ` : ""}
          Plan covers {pct(stop.cumulative_prob, 0)} of the probability by this stop.
        </div>
      </div>
    </div>
  );
}

const MENU_ITEM =
  "flex cursor-default select-none items-start gap-2 rounded-sm px-2 py-1.5 text-xs text-ink outline-none data-[highlighted]:bg-raised data-[disabled]:opacity-50";

function StopMenu({
  stop,
  isClosed,
  onSearched,
  onSetClosed,
}: {
  stop: Stop;
  isClosed: boolean;
  onSearched: (mode: SearchMode) => void;
  onSetClosed: (closed: boolean) => void;
}) {
  return (
    <DropdownMenu.Portal>
      <DropdownMenu.Content
        align="end"
        sideOffset={4}
        collisionPadding={8}
        className="z-50 w-64 rounded border border-line-strong bg-bg p-1 shadow-lg shadow-black/50"
      >
        <DropdownMenu.Label className="px-2 pb-1 pt-1.5 text-[10.5px] font-semibold uppercase tracking-wider text-ink-mute">
          Crew searched {stop.room_name}
        </DropdownMenu.Label>
        {(["rapid", "thorough"] as const).map((mode) => (
          <DropdownMenu.Item key={mode} className={MENU_ITEM} onSelect={() => onSearched(mode)}>
            <Search size={13} className="mt-0.5 shrink-0 text-ink-mute" aria-hidden />
            <span>
              <span className="font-medium">{MODE_LABEL[mode]}</span>
              {mode === stop.mode ? <span className="ml-1.5 text-[10.5px] text-ink-mute">(default)</span> : null}
              <span className="block text-[11px] text-ink-mute">
                {mode === "rapid" ? "One quick pass. Can miss a person." : "Slower, misses less."}
              </span>
            </span>
          </DropdownMenu.Item>
        ))}
        <DropdownMenu.Separator className="my-1 h-px bg-line" />
        <DropdownMenu.CheckboxItem className={MENU_ITEM} checked={isClosed} onCheckedChange={(c) => onSetClosed(c === true)}>
          <span className="mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-sm border border-line-strong">
            <DropdownMenu.ItemIndicator>
              <Check size={11} aria-hidden />
            </DropdownMenu.ItemIndicator>
          </span>
          <span>
            <span className="font-medium">Closed by officer</span>
            <span className="block text-[11px] text-ink-mute">
              Records the officer's decision. The plan leaves the room out until reopened.
            </span>
          </span>
        </DropdownMenu.CheckboxItem>
        <p className="px-2 pb-1 pt-1.5 text-[10.5px] leading-snug text-ink-mute">
          Detection rates and search times are unverified settings.
        </p>
      </DropdownMenu.Content>
    </DropdownMenu.Portal>
  );
}

function FoundSummary({
  room,
  foundAt,
  callStartedAt,
  contactAt,
  busy,
  logging,
  onLogContact,
}: {
  room: Room;
  foundAt: string | null;
  callStartedAt: string | null;
  contactAt: string | null;
  busy: boolean;
  logging: boolean;
  onLogContact: () => void;
}) {
  const since = sinceCall(foundAt, callStartedAt);
  return (
    <div className="border-b border-line p-3">
      <div className="rounded border border-ink/40 bg-raised p-3">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
          <MapPin size={13} aria-hidden />
          Caller found
        </div>
        <div className="mt-1 text-[17px] font-semibold leading-tight text-ink">{room.name}</div>
        <div className="num mt-0.5 text-[12px] text-ink-dim">
          Floor {room.floor_label}
          {foundAt ? (
            <>
              {" "}- recorded {clockTime(foundAt)}
              {since ? `, ${since} from call start` : ""}
            </>
          ) : null}
        </div>
        <p className="mt-2 text-[11.5px] text-ink-mute">
          The search plan is closed. Use Undo last if this was recorded by mistake.
        </p>
        <div className="mt-2.5">
          {contactAt ? (
            <div className="num flex items-center gap-1.5 text-[12px] text-ink">
              <Check size={13} aria-hidden />
              Patient contact logged {clockTime(contactAt)}
              {sinceCall(contactAt, callStartedAt) ? ` (${sinceCall(contactAt, callStartedAt)})` : ""}
            </div>
          ) : (
            <button
              onClick={onLogContact}
              disabled={busy}
              className="rounded bg-accent px-3 py-1.5 text-[12px] font-semibold text-accent-ink hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60"
            >
              {logging ? "Logging…" : "Log patient contact"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ListHeading({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="eyebrow mb-1.5 flex items-center gap-1.5">
      {icon}
      {children}
    </div>
  );
}

function SearchedList({
  looks,
  rooms,
  roomProbs,
  callStartedAt,
  onHoverRooms,
}: {
  looks: Looks;
  rooms: Map<string, Room>;
  roomProbs: Record<string, number>;
  callStartedAt: string | null;
  onHoverRooms?: (roomIds: string[] | null) => void;
}) {
  if (looks.size === 0) return null;
  const nLooks = [...looks.values()].reduce((n, l) => n + l.length, 0);
  return (
    <div className="border-t border-line p-3">
      <ListHeading icon={<Search size={12} aria-hidden />}>
        Searched ({looks.size} room{looks.size === 1 ? "" : "s"}
        {nLooks > looks.size ? `, ${nLooks} looks` : ""})
      </ListHeading>
      <ul className="space-y-1.5">
        {[...looks.entries()].map(([id, list]) => {
          const room = rooms.get(id);
          return (
            <li
              key={id}
              className="text-xs"
              onMouseEnter={() => onHoverRooms?.([id])}
              onMouseLeave={() => onHoverRooms?.(null)}
            >
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 truncate text-ink">{room?.name ?? id}</span>
                {room ? <span className="num shrink-0 text-[10px] text-ink-mute">F{room.floor_label}</span> : null}
                <span
                  className="num ml-auto shrink-0 text-ink-dim"
                  title="Chance the caller is still in this room after the searches so far. One pass can miss a person."
                >
                  {pct(roomProbs[id] ?? 0)} remains
                </span>
              </div>
              <div className="num text-[11px] text-ink-mute">
                {list.map((l, k) => (
                  <span key={k}>
                    {k > 0 ? ", " : ""}
                    {MODE_LABEL[l.mode]} {clockTime(l.at)}
                    {sinceCall(l.at, callStartedAt) ? ` (${sinceCall(l.at, callStartedAt)})` : ""}
                  </span>
                ))}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ClosedList({
  closedIds,
  rooms,
  busy,
  onReopen,
  onHoverRooms,
}: {
  closedIds: string[];
  rooms: Map<string, Room>;
  busy: boolean;
  onReopen: (roomId: string) => void;
  onHoverRooms?: (roomIds: string[] | null) => void;
}) {
  if (closedIds.length === 0) return null;
  return (
    <div className="border-t border-line p-3">
      <ListHeading icon={<Ban size={12} aria-hidden />}>Closed by officer ({closedIds.length})</ListHeading>
      <p className="mb-1.5 text-[11px] text-ink-mute">Recorded on the officer's word. Left out of the plan until reopened.</p>
      <ul className="space-y-1">
        {closedIds.map((id) => {
          const room = rooms.get(id);
          return (
            <li
              key={id}
              className="flex items-center gap-2 text-xs"
              onMouseEnter={() => onHoverRooms?.([id])}
              onMouseLeave={() => onHoverRooms?.(null)}
            >
              <span className="min-w-0 truncate text-ink">{room?.name ?? id}</span>
              {room ? <span className="num shrink-0 text-[10px] text-ink-mute">F{room.floor_label}</span> : null}
              <button
                onClick={() => onReopen(id)}
                disabled={busy}
                title="Record that the officer reopened this room"
                className="ml-auto flex shrink-0 items-center gap-1 rounded border border-line-strong px-1.5 py-0.5 text-[11px] text-ink-dim hover:border-ink-dim hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
              >
                <LockOpen size={11} aria-hidden />
                Reopen
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function PlanFooter({ plan }: { plan: SearchPlanData }) {
  const solver =
    plan.solver === "exact" ? "Exact order (best for these settings)" : "Heuristic order (fast, not proven best)";
  const objective = plan.objective === "survival" ? "ranked by survival chance" : "ranked by expected time to find";
  return (
    <div className="border-t border-line bg-bg/40 px-3 py-2.5 text-[11.5px] text-ink-dim">
      <dl className="num grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5">
        <dt>Time to cover every room once</dt>
        <dd className="text-right text-ink">{plan.time_to_cover_all_s !== null ? mmss(plan.time_to_cover_all_s) : "-"}</dd>
        <dt>Expected time to find, this plan</dt>
        <dd className="text-right text-ink">{mmss(plan.expected_time_to_find_s)}</dd>
        <dt className="text-ink-mute">Nearest room first</dt>
        <dd className="text-right">{mmss(plan.baseline_nearest_s)}</dd>
        <dt className="text-ink-mute">Most likely room first</dt>
        <dd className="text-right">{mmss(plan.baseline_highest_prob_s)}</dd>
      </dl>
      <div className="mt-1.5 text-[11px] text-ink-mute">
        {solver}, {objective}.
      </div>
      {plan.notes.length > 0 ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] text-ink-mute">
          {plan.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { ArrowUp, ChevronDown, DoorOpen, MapPin, Search, Waypoints } from "lucide-react";
import type { Building, DemoState } from "../types";
import { mmss, pct, roomIndex } from "../lib";

interface Props {
  state: DemoState;
  building: Building;
  /** rooms outlined on the map right now (hovered evidence, hovered cell) */
  outlineRoomIds: string[] | null;
  /** hovering "Most likely" outlines that room on the map */
  onHoverRooms: (roomIds: string[] | null) => void;
  showDetails: boolean;
  onToggleDetails: () => void;
}

/** A second floor joins the band when it holds at least this share. */
const SECOND_FLOOR_MIN = 0.05;

/**
 * Four plain-language answers for a dispatcher who is talking, not reading statistics:
 * where, how in, how much is searched, and whether the caller may be somewhere else.
 * Entropy, peaks and expected time sit behind "Details".
 */
export function AnswerStrip({ state, building, outlineRoomIds, onHoverRooms, showDetails, onToggleDetails }: Props) {
  const { posterior, plan, approach, policy } = state;
  const rooms = useMemo(() => roomIndex(building), [building]);

  // ---- most likely room and its floor band (p_floor is keyed by floor label) ----
  let topId: string | null = null;
  let topP = 0;
  for (const [id, p] of Object.entries(posterior.room_probs)) if (p > topP) [topId, topP] = [id, p];
  const foundRoom = state.found_room_id ? rooms.get(state.found_room_id) : undefined;
  const topRoom = foundRoom ?? (topId ? rooms.get(topId) : undefined);
  const floors = Object.entries(posterior.p_floor).sort((a, b) => b[1] - a[1]);
  const band = floors.filter(([, p], i) => i === 0 || (i === 1 && p >= SECOND_FLOOR_MIN)).slice(0, 2);
  const floorName = (label: string) => `Floor ${label}`;
  const topOutlined = topRoom !== undefined && (outlineRoomIds?.includes(topRoom.id) ?? false);

  // ---- outside the mapped rooms, with its trend (kept on the client: the server sends one value) ----
  const pOut = plan.p_out ?? posterior.p_not_in_model;
  const [lastPOut, setLastPOut] = useState(pOut);
  const [pOutRose, setPOutRose] = useState(false);
  if (pOut !== lastPOut) {
    setPOutRose(pOut > lastPOut + 1e-4);
    setLastPOut(pOut);
  }

  const searchedRooms = new Set(state.searched.map((s) => s.room_id)).size;
  const routeOn = policy.route_enabled;

  return (
    <div className="border-b border-line">
      <div className="grid grid-cols-2 gap-px bg-line lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_auto]">
        {!routeOn && !foundRoom ? (
          // Route planning off (active threat, confirmed or suggested): no single room is put
          // forward as the place to go. The floor band stays, as information for command.
          <Answer
            label="Reports concentrate on"
            icon={<MapPin size={14} aria-hidden />}
            title="Floors that hold most of the probability. Information only: route planning is off for this incident."
          >
            <span className="truncate text-[17px] font-semibold leading-tight tabular-nums text-ink">
              {band.length > 0 ? band.map(([label, p]) => `${floorName(label)} (${pct(p)})`).join(" / ") : "No floor stands out yet"}
            </span>
            <span className="truncate text-[13px] text-ink-dim">
              {policy.show_incident_board
                ? "Information only - caller reports are on the incident board"
                : "Information only - route planning is off"}
            </span>
          </Answer>
        ) : (
        <Answer
          label={foundRoom ? "Found" : "Most likely"}
          icon={<MapPin size={14} aria-hidden />}
          highlight={topOutlined}
          onHover={(on) => onHoverRooms(on && topRoom ? [topRoom.id] : null)}
          title="Most probable room, and the floors that hold most of the probability"
        >
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="truncate text-[17px] font-semibold leading-tight text-ink">{topRoom?.name ?? "-"}</span>
            {topRoom && !foundRoom ? <span className="num shrink-0 text-[13px] text-ink-dim">{pct(topP)}</span> : null}
          </div>
          <div className="truncate text-[13px] tabular-nums text-ink-dim">
            {foundRoom
              ? `${floorName(foundRoom.floor_label)} - recorded by the dispatcher`
              : band.length > 0
                ? band.map(([label, p]) => `${floorName(label)} (${pct(p)})`).join(" or ")
                : "No floor stands out yet"}
          </div>
        </Answer>
        )}

        <Answer label="Go in by" icon={<DoorOpen size={14} aria-hidden />} title="Recommended entrance and stairs">
          {routeOn && approach.entrance_name ? (
            <>
              <span className="truncate text-[17px] font-semibold leading-tight text-ink">{approach.entrance_name}</span>
              <span className="truncate text-[13px] text-ink-dim">
                {approach.stairwell_name ? `then ${approach.stairwell_name}` : "No stair recommendation"}
                {approach.target_floor_label ? ` to Floor ${approach.target_floor_label}` : ""}
              </span>
            </>
          ) : (
            <>
              <span className="text-[17px] font-semibold leading-tight text-ink-mute" aria-label="No entrance shown">
                -
              </span>
              <span className="truncate text-[13px] text-ink-dim">
                {routeOn ? "No entrance computed" : "Route planning is off for this incident"}
              </span>
            </>
          )}
        </Answer>

        <Answer
          label={foundRoom ? "Searched before found" : "Searched so far"}
          icon={<Search size={14} aria-hidden />}
          title="Rooms searched so far, and the chance the current plan finds the caller once it is followed to the end"
        >
          <span className="text-[17px] font-semibold leading-tight tabular-nums text-ink">
            {searchedRooms === 0 ? "None yet" : `${searchedRooms} room${searchedRooms === 1 ? "" : "s"}`}
          </span>
          <span className="truncate text-[13px] text-ink-dim">
            {foundRoom ? (
              "Search ended - caller found"
            ) : plan.cumulative_success !== null ? (
              <>
                Plan reaches <span className="num font-semibold text-ink">{pct(plan.cumulative_success)}</span> chance of
                finding
              </>
            ) : routeOn ? (
              "Plan coverage not reported"
            ) : (
              "No search plan for this incident"
            )}
          </span>
        </Answer>

        <Answer
          label="May not be in mapped rooms"
          icon={<Waypoints size={14} aria-hidden />}
          title="Chance the caller is somewhere the building model does not cover (after the searches so far)"
        >
          <span className="flex items-baseline gap-1.5">
            <span className="num text-[17px] font-semibold leading-tight text-ink">{pct(pOut)}</span>
            {pOutRose ? (
              <span className="flex items-center text-[12px] font-semibold text-warn" title="Higher than before the last update">
                <ArrowUp size={14} strokeWidth={2.5} aria-hidden />
                rising
              </span>
            ) : null}
          </span>
          <span className="truncate text-[13px] text-ink-dim">chance the caller is elsewhere</span>
        </Answer>

        <button
          onClick={onToggleDetails}
          aria-expanded={showDetails}
          aria-controls="answer-details"
          className="col-span-2 flex items-center justify-center gap-1 bg-panel px-3 py-1.5 text-xs text-ink-dim hover:bg-raised hover:text-ink lg:col-span-1"
        >
          Details
          <ChevronDown size={14} className={`transition-transform ${showDetails ? "rotate-180" : ""}`} aria-hidden />
        </button>
      </div>

      {showDetails ? (
        <dl
          id="answer-details"
          className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-line bg-bg px-3 py-1.5 text-xs"
        >
          <Detail term="Entropy" title="Shannon entropy of the room probabilities. Lower = more certain.">
            {posterior.entropy_bits.toFixed(2)} bits
          </Detail>
          <Detail term="Peaks" title="Connected clusters of high-probability rooms">
            {posterior.peak_count}
          </Detail>
          <Detail term="90% set" title="Smallest set of rooms holding 90% of the in-model probability">
            {posterior.credible_set_90.length} of {building.rooms.length} rooms
          </Detail>
          {routeOn ? (
            <>
              <Detail term="Expected time to find" title="Expected time to find the caller following the plan">
                {mmss(plan.expected_time_to_find_s)}
              </Detail>
              {plan.time_to_cover_all_s !== null ? (
                <Detail term="Time to cover every room" title="When the last first look in the plan finishes">
                  {mmss(plan.time_to_cover_all_s)}
                </Detail>
              ) : null}
              <Detail term="Plan" title="Planner objective and solver">
                {plan.objective === "survival" ? "survival-weighted" : "shortest expected time"}, {plan.solver}
              </Detail>
            </>
          ) : null}
          <Detail term="Prior hour" title="Hour of day the occupancy prior was computed for">
            {String(state.prior_hour).padStart(2, "0")}:00
          </Detail>
        </dl>
      ) : null}
    </div>
  );
}

function Answer({
  label,
  icon,
  title,
  highlight = false,
  onHover,
  children,
}: {
  label: string;
  icon: ReactNode;
  title?: string;
  highlight?: boolean;
  onHover?: (on: boolean) => void;
  children: ReactNode;
}) {
  return (
    <div
      className={`flex min-w-0 flex-col justify-center gap-0.5 px-3 py-1.5 ${highlight ? "bg-raised" : "bg-panel"}`}
      title={title}
      onMouseEnter={onHover ? () => onHover(true) : undefined}
      onMouseLeave={onHover ? () => onHover(false) : undefined}
    >
      <span className="eyebrow flex items-center gap-1.5 !text-[10px] leading-none">
        <span className="text-ink-mute">{icon}</span>
        {label}
      </span>
      {children}
    </div>
  );
}

function Detail({ term, title, children }: { term: string; title?: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-1.5" title={title}>
      <dt className="text-ink-mute">{term}</dt>
      <dd className="num m-0 text-ink">{children}</dd>
    </div>
  );
}

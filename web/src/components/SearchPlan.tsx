import { useMemo, useState } from "react";
import { Check, ListOrdered } from "lucide-react";
import type { Building, Posterior, SearchPlan as SearchPlanData, Stop } from "../types";
import { mmss, pct, roomIndex } from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  plan: SearchPlanData;
  posterior: Posterior;
  clearedRoomIds: string[];
  /** room id currently being cleared (request in flight) */
  clearingRoomId: string | null;
  /** true while any request is in flight */
  busy: boolean;
  onClear: (roomId: string) => void;
}

const VISIBLE_STOPS = 8;

export function SearchPlan({ building, plan, posterior, clearedRoomIds, clearingRoomId, busy, onClear }: Props) {
  const rooms = useMemo(() => roomIndex(building), [building]);
  const multiTeam = plan.teams.length > 1;
  const totalStops = plan.teams.reduce((n, t) => n + t.length, 0);

  return (
    <Panel
      title="Search plan"
      icon={<ListOrdered size={15} aria-hidden />}
      actions={
        <span className="num text-xs text-ink-mute">
          {totalStops} stops - E[T] <span className="text-ink">{mmss(plan.expected_time_to_find_s)}</span>
        </span>
      }
      className="h-full"
      bodyClassName="scroll-thin overflow-y-auto"
    >
      {totalStops === 0 ? (
        <p className="p-3 text-xs text-ink-mute">No rooms left to search in the current plan.</p>
      ) : (
        plan.teams.map((stops, team) => (
          <TeamList
            key={team}
            team={team}
            stops={stops}
            multiTeam={multiTeam}
            posterior={posterior}
            clearingRoomId={clearingRoomId}
            busy={busy}
            onClear={onClear}
          />
        ))
      )}

      {clearedRoomIds.length > 0 ? (
        <div className="border-t border-line p-3">
          <div className="eyebrow mb-1.5">Cleared ({clearedRoomIds.length})</div>
          <ul className="space-y-1">
            {clearedRoomIds.map((id) => {
              const room = rooms.get(id);
              return (
                <li key={id} className="flex items-center gap-2 text-xs text-ink-mute">
                  <Check size={12} className="shrink-0" aria-hidden />
                  <span className="line-through">{room?.name ?? id}</span>
                  {room ? <span className="num">F{room.floor_label}</span> : null}
                  <span className="num ml-auto" title="Residual probability after a 90%-effective search">
                    {pct(posterior.room_probs[id] ?? 0)} left
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </Panel>
  );
}

function TeamList({
  team,
  stops,
  multiTeam,
  posterior,
  clearingRoomId,
  busy,
  onClear,
}: {
  team: number;
  stops: Stop[];
  multiTeam: boolean;
  posterior: Posterior;
  clearingRoomId: string | null;
  busy: boolean;
  onClear: (roomId: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? stops : stops.slice(0, VISIBLE_STOPS);
  const hidden = stops.length - visible.length;
  const teamName = `Team ${String.fromCharCode(65 + team)}`;

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
            const isNext = i === 0;
            const p = posterior.room_probs[stop.room_id] ?? 0;
            const clearing = clearingRoomId === stop.room_id;
            return (
              <li key={stop.room_id} className={`px-3 py-2 ${isNext ? "bg-accent/[0.06]" : ""}`}>
                <div className="flex items-center gap-2.5">
                  <span
                    className={`num flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
                      isNext ? "bg-accent text-accent-ink" : "border border-line-strong text-ink-dim"
                    }`}
                    aria-label={`Stop ${i + 1}`}
                  >
                    {i + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="truncate text-[13px] font-medium text-ink">{stop.room_name}</span>
                      <span className="num shrink-0 rounded-sm border border-line px-1 text-[10px] text-ink-dim">
                        F{stop.floor_label}
                      </span>
                      {isNext ? <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-accent">Next</span> : null}
                    </div>
                    <div className="num mt-0.5 flex gap-3 text-[11px] text-ink-mute">
                      <span title="Arrival time from now">
                        ETA <span className="text-ink-dim">{mmss(stop.eta_s)}</span>
                      </span>
                      <span title="Posterior probability the caller is in this room">
                        P <span className="text-ink-dim">{pct(p)}</span>
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => onClear(stop.room_id)}
                    disabled={busy}
                    aria-label={`Mark ${stop.room_name} cleared`}
                    className={`shrink-0 rounded border px-2 py-1 text-[11px] font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
                      isNext
                        ? "border-accent/70 text-accent hover:bg-accent hover:text-accent-ink"
                        : "border-line-strong text-ink-dim hover:border-ink-dim hover:text-ink"
                    }`}
                  >
                    {clearing ? "Clearing…" : "Mark Cleared"}
                  </button>
                </div>
                <div className="mt-1.5 flex items-center gap-2 pl-[34px]" title="Cumulative probability searched once this stop is complete">
                  <span className="block h-1 flex-1 overflow-hidden rounded-full bg-line" aria-hidden>
                    <span
                      className="block h-full rounded-full bg-prob transition-[width] duration-300"
                      style={{ width: `${Math.min(100, stop.cumulative_prob * 100)}%` }}
                    />
                  </span>
                  <span className="num w-12 text-right text-[10px] text-ink-mute">Σ {pct(stop.cumulative_prob, 0)}</span>
                </div>
              </li>
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

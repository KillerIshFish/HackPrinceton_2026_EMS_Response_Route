import { Clock, Footprints, MoveVertical } from "lucide-react";
import type { Approach, IncidentType, Policy, Stake } from "../types";
import { mmss } from "../lib";

interface Props {
  policy: Policy;
  approach: Approach;
  incidentType: IncidentType;
}

const MEDICAL: ReadonlySet<IncidentType> = new Set([
  "cardiac_arrest",
  "stroke",
  "internal_bleeding",
  "external_bleeding",
  "overdose",
  "other_medical",
]);

/**
 * One figure for the stake: "under 0.1" below 0.1, two decimals (trailing zeros trimmed)
 * below 1, one decimal otherwise. Never "0": zero and anything below 0.1 read "under 0.1".
 */
export function formatStake(v: number): string {
  if (!Number.isFinite(v) || v < 0.1) return "under 0.1";
  if (v < 1) return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return v.toFixed(1);
}

/** "1.8 to 2.3", or one figure when both ends agree; null when the stake has no figures. */
export function stakeRange(stake: Stake): string | null {
  if (!stake.quantified) return null;
  const { per_min_low: lo, per_min_high: hi } = stake;
  if (lo === null && hi === null) return null;
  const a = formatStake(lo ?? hi!);
  const b = formatStake(hi ?? lo!);
  return a === b ? a : `${a} to ${b}`;
}

/**
 * What a minute of delay costs, from policy.stake, directly under the answer strip. A label alone
 * when the stake is not quantified; nothing at all when there is no stake (active threat). For a
 * medical call with an elevator on offer, stair and elevator times to the target floor sit side by
 * side as the crew's choice: the console does not pick between them.
 */
export function StakeCard({ policy, approach, incidentType }: Props) {
  const stake = policy.stake;
  const stairs = approach.stairs_s_to_target_floor ?? null;
  const elevator = approach.elevator_s_to_target_floor ?? null;
  const showLift =
    policy.route_enabled && policy.elevator_allowed && MEDICAL.has(incidentType) && stairs !== null && elevator !== null;
  if (!stake && !showLift) return null;
  const range = stake ? stakeRange(stake) : null;

  return (
    <div
      role="region"
      aria-label="Time cost"
      className="flex flex-wrap items-stretch gap-px border-b border-line bg-line"
    >
      {stake ? (
        <div className="flex min-w-0 flex-[1_1_520px] items-center gap-3 bg-panel px-3 py-1.5">
          <span className="eyebrow flex shrink-0 items-center gap-1.5 !text-[10px]">
            <Clock size={14} className="text-ink-mute" aria-hidden />
            Time cost
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-[14px] font-semibold text-ink">{stake.label}</span>
              {range ? (
                <span className="text-[13px] text-ink">
                  <span className="num font-semibold">{range}</span>
                  {stake.unit ? <span className="text-ink-dim"> {stake.unit}</span> : null}
                </span>
              ) : null}
            </div>
            {range && (stake.origin || stake.note) ? (
              <p className="truncate text-[11px] leading-snug text-ink-mute" title={[stake.origin, stake.note].filter(Boolean).join(". ")}>
                {stake.origin ? <span className="text-ink-dim">{stake.origin}.</span> : null}
                {stake.note ? <span> {stake.note}</span> : null}
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      {showLift ? (
        <div className="flex flex-[0_1_auto] items-center gap-3 bg-panel px-3 py-1.5" aria-label="Stairs or elevator">
          <span className="eyebrow shrink-0 !text-[10px]">
            To Floor <span className="num">{approach.target_floor_label ?? "?"}</span>
          </span>
          <span className="flex items-baseline gap-1.5 text-[13px] text-ink" title="Stair time from the entrance floor to the target floor">
            <Footprints size={14} className="self-center text-ink-mute" aria-hidden />
            Stairs <span className="num font-semibold">{mmss(stairs!)}</span>
          </span>
          <span className="flex items-baseline gap-1.5 text-[13px] text-ink" title="Elevator time to the target floor, including the wait">
            <MoveVertical size={14} className="self-center text-ink-mute" aria-hidden />
            Elevator <span className="num font-semibold">{mmss(elevator!)}</span>
          </span>
          <span className="rounded border border-line-strong px-1.5 py-0.5 text-[11px] font-semibold text-ink-dim">Crew's choice</span>
        </div>
      ) : null}
    </div>
  );
}

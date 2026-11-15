import type { SearchPlan } from "../types";
import { pct } from "../lib";

/** Under the plan header: the chance the plan finds the caller by its last stop. */
export function PlanSuccess({ plan }: { plan: SearchPlan }) {
  const value = plan.cumulative_success;
  const w = value === null ? 0 : Math.min(100, Math.max(0, value * 100));
  return (
    <div className="shrink-0 border-b border-line px-3 py-1.5">
      <div className="min-w-0" title="Chance the caller is found once every stop in this plan has been searched as listed">
        <div className="flex items-baseline gap-2">
          <span className="truncate text-[11px] text-ink-dim">Found by end of plan</span>
          <span className="num ml-auto shrink-0 text-[13px] font-semibold text-ink">{value === null ? "-" : pct(value)}</span>
        </div>
        <div className="relative mt-0.5 h-1.5 rounded-full bg-line" aria-hidden>
          <div className="h-full rounded-full bg-prob transition-[width] duration-300" style={{ width: `${w}%` }} />
        </div>
      </div>
    </div>
  );
}

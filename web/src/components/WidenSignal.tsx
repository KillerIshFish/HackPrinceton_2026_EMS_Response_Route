import { useState } from "react";
import { TriangleAlert } from "lucide-react";
import type { Posterior, SearchPlan } from "../types";
import { pct } from "../lib";

/**
 * Agency-set: the share of probability outside the mapped rooms at which the console suggests
 * widening the search. A setting for the agency to choose, not a published figure. Nothing happens
 * automatically when it is passed.
 */
export const WIDEN_P_OUT_THRESHOLD = 0.3;

const HISTORY_MAX = 40;

interface Props {
  plan: SearchPlan;
  posterior: Posterior;
  /** history restarts when this changes (new building or scenario) */
  historyKey: string;
}

/**
 * Two numbers under the plan header: the chance the caller is outside the mapped rooms (with its
 * history this session) and the chance the plan finds the caller by its last stop.
 */
export function WidenSignal({ plan, posterior, historyKey }: Props) {
  const pOut = plan.p_out ?? posterior.p_not_in_model;
  const success = plan.cumulative_success;

  // The server sends one value per state, so the history is kept here, one point per change.
  const [hist, setHist] = useState<{ key: string; values: number[] }>({ key: historyKey, values: [pOut] });
  if (hist.key !== historyKey) {
    setHist({ key: historyKey, values: [pOut] });
  } else if (Math.abs((hist.values[hist.values.length - 1] ?? -1) - pOut) > 1e-6) {
    setHist({ key: historyKey, values: [...hist.values, pOut].slice(-HISTORY_MAX) });
  }
  const values = hist.key === historyKey ? hist.values : [pOut];
  const prev = values.length > 1 ? values[values.length - 2]! : null;
  const over = pOut > WIDEN_P_OUT_THRESHOLD;

  return (
    <div className="shrink-0 border-b border-line px-3 py-1.5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-x-3 gap-y-1">
        <Meter
          label="May not be in mapped rooms"
          value={pOut}
          tone={over ? "warn" : "prob"}
          threshold={WIDEN_P_OUT_THRESHOLD}
          title={`Chance the caller is somewhere the building model does not cover, after the searches so far. The tick marks the agency-set ${pct(WIDEN_P_OUT_THRESHOLD, 0)} threshold.`}
          trend={prev === null ? null : pOut > prev + 1e-4 ? "up" : pOut < prev - 1e-4 ? "down" : null}
        />
        <div className="flex flex-col items-end">
          <span
            className="text-[10px] leading-none text-ink-mute"
            title="Agency-set threshold for suggesting a wider search (dashed line and bar tick)"
          >
            widen at {pct(WIDEN_P_OUT_THRESHOLD, 0)}
          </span>
          <Sparkline values={values} threshold={WIDEN_P_OUT_THRESHOLD} warn={over} />
        </div>
        <div className="col-span-2">
          <Meter
            label="Found by end of plan"
            value={success}
            tone="prob"
            title="Chance the caller is found once every stop in this plan has been searched as listed"
          />
        </div>
      </div>
      {over ? (
        <div role="status" className="mt-2 flex gap-2 rounded border border-warn/60 bg-warn/10 px-2 py-1.5 text-[12px] leading-snug text-ink">
          <TriangleAlert size={14} className="mt-0.5 shrink-0 text-warn" aria-hidden />
          <span>
            Consider widening beyond the mapped rooms. Ending a building search is command's decision.
          </span>
        </div>
      ) : null}
    </div>
  );
}

function Meter({
  label,
  value,
  tone,
  threshold,
  title,
  trend = null,
}: {
  label: string;
  value: number | null;
  tone: "prob" | "warn";
  threshold?: number;
  title: string;
  trend?: "up" | "down" | null;
}) {
  const w = value === null ? 0 : Math.min(100, Math.max(0, value * 100));
  return (
    <div className="min-w-0" title={title}>
      <div className="flex items-baseline gap-2">
        <span className="truncate text-[11px] text-ink-dim">{label}</span>
        <span className="num ml-auto shrink-0 text-[13px] font-semibold text-ink">{value === null ? "-" : pct(value)}</span>
        {trend ? (
          <span className={`shrink-0 text-[10.5px] font-semibold ${trend === "up" ? "text-warn" : "text-ink-mute"}`}>
            {trend === "up" ? "▲ rising" : "▼ falling"}
          </span>
        ) : null}
      </div>
      <div className="relative mt-0.5 h-1.5 rounded-full bg-line" aria-hidden>
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${tone === "warn" ? "bg-warn" : "bg-prob"}`}
          style={{ width: `${w}%` }}
        />
        {threshold !== undefined ? (
          <div className="absolute -top-0.5 h-2.5 w-0.5 bg-ink-dim" style={{ left: `calc(${threshold * 100}% - 1px)` }} />
        ) : null}
      </div>
    </div>
  );
}

const SW = 72;
const SH = 22;

/** p_out after each change this session. The y-scale shows the threshold whenever it is in reach. */
function Sparkline({ values, threshold, warn }: { values: number[]; threshold: number; warn: boolean }) {
  const top = Math.max(...values, 0.05) * 1.25;
  const yMax = top >= threshold * 0.6 ? Math.max(top, threshold * 1.15) : top;
  const n = values.length;
  const x = (i: number) => (n === 1 ? SW - 3 : 3 + (i * (SW - 6)) / (n - 1));
  const y = (v: number) => SH - 2 - (v / yMax) * (SH - 4);
  const d = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const last = values[n - 1] ?? 0;
  return (
    <svg
      width={SW}
      height={SH}
      role="img"
      aria-label={`Outside-the-mapped-rooms chance this session: ${values.map((v) => pct(v)).join(", ")}`}
      className="shrink-0 overflow-visible"
    >
      <title>{`This session: ${values.map((v) => pct(v)).join(" → ")}`}</title>
      <line x1={0} x2={SW} y1={SH - 2} y2={SH - 2} stroke="var(--color-line)" strokeWidth={1} />
      {threshold <= yMax ? (
        <line x1={0} x2={SW} y1={y(threshold)} y2={y(threshold)} stroke="var(--color-ink-mute)" strokeWidth={1} strokeDasharray="2 2" />
      ) : null}
      {n > 1 ? <path d={d} fill="none" stroke={warn ? "var(--color-warn)" : "var(--color-prob)"} strokeWidth={1.75} strokeLinejoin="round" /> : null}
      <circle cx={x(n - 1)} cy={y(last)} r={2.5} fill={warn ? "var(--color-warn)" : "var(--color-prob)"} />
    </svg>
  );
}

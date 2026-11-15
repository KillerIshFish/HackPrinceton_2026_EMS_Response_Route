import { ArrowUpDown, Smartphone } from "lucide-react";
import type { Building, PhoneFix, Posterior } from "../types";
import { Z_BAND_M, floorLabelMap, floorZ, pct } from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  posterior: Posterior;
  phoneFix: PhoneFix | null;
  selectedFloor: number;
  onSelectFloor: (floor: number) => void;
}

/**
 * More floors than DENSE_FROM: one-line rows; more than DENSE_SCROLL: the list scrolls instead of
 * stretching the floor plan beside it. Layout thresholds, tuned by eye; no external source.
 */
const DENSE_FROM = 8;
const DENSE_SCROLL = 16;

/** Elevator-style vertical stack: top floor at the top, one cell per floor. */
export function FloorProbBar({ building, posterior, phoneFix, selectedFloor, onSelectFloor }: Props) {
  const labels = floorLabelMap(building);
  const floors = [...building.floors].sort((a, b) => b - a);
  const z = phoneFix?.z_m ?? null;

  // Floors whose expected phone height lies inside the +/-3 m (80%) z band (Z_BAND_M, lib.ts), and
  // the single nearest floor, which carries the phone marker. Display only: the posterior already
  // weighs the fix.
  const inBand = new Set<number>();
  let nearest: number | null = null;
  if (z !== null) {
    let best = Infinity;
    for (const f of floors) {
      const d = Math.abs(z - floorZ(building, f));
      if (d <= Z_BAND_M) inBand.add(f);
      if (d < best) [best, nearest] = [d, f];
    }
  }

  const maxFloorP = Math.max(0, ...floors.map((f) => posterior.p_floor[labels.get(f) ?? String(f)] ?? 0));

  const dense = floors.length > DENSE_FROM;
  const scrolls = floors.length > DENSE_SCROLL;

  return (
    <Panel title="Floor" icon={<ArrowUpDown size={15} aria-hidden />} bodyClassName="flex min-h-0 flex-col p-2">
      <ol
        className={`flex flex-1 flex-col ${dense ? "gap-0.5" : "gap-1"} ${scrolls ? "max-h-[340px] overflow-y-auto pr-1" : ""}`}
        aria-label="Probability by floor, top floor first"
      >
        {floors.map((f) => {
          const label = labels.get(f) ?? String(f);
          const p = posterior.p_floor[label] ?? 0;
          const selected = f === selectedFloor;
          // ties all highlight; a floor with no mass is never "top"
          const isTop = p > 0 && p === maxFloorP;
          const band = inBand.has(f);
          return (
            <li key={f} className={`flex ${scrolls ? "shrink-0" : "flex-1"}`}>
              <button
                ref={
                  selected && scrolls
                    ? (el) => {
                        // keep the selected floor visible by scrolling the list only, never the page
                        const list = el?.closest("ol");
                        if (!el || !list) return;
                        const top = el.offsetTop - list.offsetTop;
                        if (top < list.scrollTop) list.scrollTop = top;
                        else if (top + el.offsetHeight > list.scrollTop + list.clientHeight)
                          list.scrollTop = top + el.offsetHeight - list.clientHeight;
                      }
                    : undefined
                }
                onClick={() => onSelectFloor(f)}
                aria-pressed={selected}
                aria-label={`Floor ${label}, probability ${pct(p)}${band ? ", inside phone z-axis band" : ""}`}
                className={`relative flex ${dense ? "min-h-7" : "min-h-14"} w-full items-stretch overflow-hidden rounded border text-left ${
                  selected ? "border-ink bg-raised" : "border-line hover:border-line-strong hover:bg-raised/60"
                }`}
              >
                {/* z-fix band marker on the shaft edge */}
                <span
                  className={`w-1 shrink-0 ${band ? "bg-accent" : "bg-transparent"}`}
                  aria-hidden
                />
                <span className="flex w-9 shrink-0 items-center justify-center border-r border-line">
                  <span className={`num ${dense ? "text-[13px]" : "text-lg"} font-semibold ${isTop ? "text-ink" : "text-ink-dim"}`}>{label}</span>
                </span>
                <span
                  className={`relative flex min-w-0 flex-1 ${
                    dense ? "flex-row-reverse items-center gap-2 px-2" : "flex-col justify-center gap-1 px-2 py-1.5"
                  }`}
                >
                  <span className="flex items-baseline justify-between gap-1">
                    <span className={`num text-[13px] font-semibold ${isTop ? "text-ink" : "text-ink-dim"}`}>{pct(p)}</span>
                    {nearest === f && z !== null ? (
                      <span className="flex items-center gap-1 text-[10px] font-medium text-accent" title={`Barometric z-axis fix: ${z.toFixed(1)} m above ground datum`}>
                        <Smartphone size={10} aria-hidden />
                        {dense ? null : <span className="num">{z.toFixed(1)} m</span>}
                      </span>
                    ) : null}
                  </span>
                  <span className={`block h-1.5 overflow-hidden rounded-full bg-line ${dense ? "min-w-0 flex-1" : ""}`} aria-hidden>
                    <span
                      className="block h-full rounded-full bg-prob transition-[width] duration-300"
                      style={{ width: `${Math.min(100, p * 100)}%`, opacity: isTop ? 1 : 0.65 }}
                    />
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>

      <p className="mt-2 flex items-start gap-1.5 text-[11px] leading-snug text-ink-mute">
        <span className="mt-0.5 block h-2.5 w-1 shrink-0 bg-accent" aria-hidden />
        {z !== null ? <>Within ±3 m of the phone z-axis fix</> : <>No z-axis fix yet</>}
      </p>
    </Panel>
  );
}

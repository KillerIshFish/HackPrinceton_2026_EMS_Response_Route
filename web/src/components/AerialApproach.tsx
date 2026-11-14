import { useMemo } from "react";
import { Navigation } from "lucide-react";
import type { Approach, Building, SearchPlan, XY } from "../types";
import { buildingBounds, mmss, titleCase } from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  plan: SearchPlan;
  approach: Approach;
}

const S = 6; // svg units per metre
const SITE_MARGIN_M = 6; // context drawn around the outermost feature
const BAND_M = 14; // minimum depth of the parking-lot / street bands

/** Schematic site plan + approach recommendation. Stands in for the Vexcel aerial layer. */
export function AerialApproach({ building, plan, approach }: Props) {
  const geo = useMemo(() => {
    const fp = buildingBounds(building);
    const pts: XY[] = [
      ...building.entrances.map((e) => e.xy),
      ...building.staging_spots.map((s) => s.xy),
    ];
    const minX = Math.min(fp.minX, ...pts.map((p) => p[0])) - SITE_MARGIN_M;
    const maxX = Math.max(fp.maxX, ...pts.map((p) => p[0])) + SITE_MARGIN_M;
    const minY = Math.min(fp.minY - BAND_M, ...pts.map((p) => p[1] - SITE_MARGIN_M));
    const maxY = Math.max(fp.maxY + BAND_M, ...pts.map((p) => p[1] + SITE_MARGIN_M));
    return { fp, minX, maxX, minY, maxY };
  }, [building]);

  const { fp, minX, maxX, minY, maxY } = geo;
  const W = (maxX - minX) * S;
  const H = (maxY - minY) * S;
  const X = (x: number) => (x - minX) * S;
  const Y = (y: number) => (maxY - y) * S; // north up

  const recEntrance = building.entrances.find((e) => e.id === approach.entrance_id);
  const recStaging = building.staging_spots.find((s) => s.id === approach.staging_id);

  // parking stall ticks across the north band
  const stallY0 = Y(fp.maxY + 5);
  const stallY1 = Y(fp.maxY + 10);
  const stalls: number[] = [];
  for (let x = Math.ceil(minX / 3) * 3 + 3; x < maxX - 2; x += 3) stalls.push(X(x));

  const best = Math.min(plan.baseline_nearest_s, plan.baseline_highest_prob_s);
  const saved = best > 0 ? (best - plan.expected_time_to_find_s) / best : 0;
  const bars = [
    { key: "plan", label: "This plan", value: plan.expected_time_to_find_s, primary: true },
    { key: "nearest", label: "Nearest-first", value: plan.baseline_nearest_s, primary: false },
    { key: "highest", label: "Highest-prob-first", value: plan.baseline_highest_prob_s, primary: false },
  ];
  const barMax = Math.max(1, ...bars.map((b) => b.value));

  const otherEntrances = Object.entries(approach.entrance_expected_travel_s).sort((a, b) => a[1] - b[1]);

  return (
    <Panel
      title="Approach"
      icon={<Navigation size={15} aria-hidden />}
      actions={<span className="text-[11px] text-ink-mute">Schematic site plan - aerial imagery not loaded</span>}
      bodyClassName="grid gap-0 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]"
    >
      {/* site schematic */}
      <div className="border-b border-line p-2 lg:border-b-0 lg:border-r">
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block h-auto max-h-[340px] w-full" role="img" aria-label="Schematic site plan: parking lot to the north, street to the south, entrances and staging spots">
          {/* ground */}
          <rect width={W} height={H} fill="#0e151d" />
          {/* parking lot (north) */}
          <rect x={0} y={0} width={W} height={Y(fp.maxY + 2)} fill="#131c27" />
          {stalls.map((sx) => (
            <line key={sx} x1={sx} y1={stallY0} x2={sx} y2={stallY1} stroke="#2b3a4d" strokeWidth={1} />
          ))}
          <line x1={X(minX + 3)} y1={stallY0} x2={X(maxX - 3)} y2={stallY0} stroke="#2b3a4d" strokeWidth={1} />
          <text x={8} y={14} fontSize="9.5" fontWeight="600" letterSpacing="1" fill="#8394a6">
            N - PARKING LOT
          </text>
          {/* street (south) */}
          <rect x={0} y={Y(fp.minY - 5)} width={W} height={H - Y(fp.minY - 5)} fill="#161f2a" />
          <line
            x1={0}
            y1={(Y(fp.minY - 5) + H) / 2}
            x2={W}
            y2={(Y(fp.minY - 5) + H) / 2}
            stroke="#3a4a5e"
            strokeWidth={1.2}
            strokeDasharray="10 8"
          />
          <text x={8} y={H - 6} fontSize="9.5" fontWeight="600" letterSpacing="1" fill="#8394a6">
            S - STREET
          </text>

          {/* building footprint */}
          <rect
            x={X(fp.minX)}
            y={Y(fp.maxY)}
            width={(fp.maxX - fp.minX) * S}
            height={(fp.maxY - fp.minY) * S}
            fill="#1b2633"
            stroke="#4a5d73"
            strokeWidth={1.5}
          />
          <text x={X((fp.minX + fp.maxX) / 2)} y={Y((fp.minY + fp.maxY) / 2) - 2} textAnchor="middle" fontSize="11" fontWeight="600" fill="#a7b6c5">
            {building.name}
          </text>
          <text x={X((fp.minX + fp.maxX) / 2)} y={Y((fp.minY + fp.maxY) / 2) + 12} textAnchor="middle" fontSize="9.5" fill="#8394a6">
            {building.floors.length} floors - {titleCase(building.use_type).toLowerCase()}
          </text>

          {/* approach path: staging -> entrance */}
          {recEntrance && recStaging ? (
            <line
              x1={X(recStaging.xy[0])}
              y1={Y(recStaging.xy[1])}
              x2={X(recEntrance.xy[0])}
              y2={Y(recEntrance.xy[1])}
              stroke="#f2a93b"
              strokeWidth={2}
              strokeDasharray="5 4"
            />
          ) : null}

          {/* staging spots */}
          {building.staging_spots.map((s) => {
            const rec = s.id === approach.staging_id;
            const cx = X(s.xy[0]);
            const cy = Y(s.xy[1]);
            return (
              <g key={s.id}>
                <title>{`${s.name}${rec ? " (recommended staging)" : ""}`}</title>
                <rect
                  x={cx - 9}
                  y={cy - 6}
                  width={18}
                  height={12}
                  rx={2}
                  fill={rec ? "#f2a93b" : "#0e151d"}
                  stroke={rec ? "#f2a93b" : "#62758a"}
                  strokeWidth={1.3}
                />
                <text x={cx} y={cy + 3.5} textAnchor="middle" fontSize="8.5" fontWeight="700" fill={rec ? "#1a1204" : "#a7b6c5"}>
                  STG
                </text>
              </g>
            );
          })}

          {/* entrances */}
          {building.entrances.map((e) => {
            const rec = e.id === approach.entrance_id;
            const cx = X(e.xy[0]);
            const cy = Y(e.xy[1]);
            const north = e.xy[1] > (fp.minY + fp.maxY) / 2;
            return (
              <g key={e.id}>
                <title>{`${e.name}${rec ? " (recommended entrance)" : ""}`}</title>
                <rect
                  x={cx - 8}
                  y={cy - 3.5}
                  width={16}
                  height={7}
                  fill={rec ? "#f2a93b" : "#62758a"}
                  stroke="#0b1016"
                  strokeWidth={1}
                />
                <text
                  x={cx}
                  y={north ? cy + 16 : cy - 8}
                  textAnchor="middle"
                  fontSize="9"
                  fontWeight={rec ? 700 : 500}
                  fill={rec ? "#f2a93b" : "#a7b6c5"}
                >
                  {e.name}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-ink-mute">
          <span className="flex items-center gap-1.5">
            <span className="block h-2 w-4 bg-accent" aria-hidden /> Recommended entrance / staging
          </span>
          <span className="flex items-center gap-1.5">
            <span className="block h-2 w-4 bg-[#62758a]" aria-hidden /> Other entrance
          </span>
        </div>
      </div>

      {/* recommendation + E[T] comparison */}
      <div className="flex flex-col gap-3 p-3">
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-[13px]">
          <dt className="text-ink-mute">Stage at</dt>
          <dd className="text-ink">
            <span className="font-medium">{approach.staging_name}</span>
            <span className="num ml-2 text-[11px] text-ink-dim">
              {approach.staging_latlng[0].toFixed(5)}, {approach.staging_latlng[1].toFixed(5)}
            </span>
          </dd>
          <dt className="text-ink-mute">Enter via</dt>
          <dd className="font-medium text-accent">{approach.entrance_name}</dd>
          <dt className="text-ink-mute">Go up</dt>
          <dd className="text-ink">
            {approach.stairwell_name ? (
              <span className="font-medium">{approach.stairwell_name}</span>
            ) : (
              <span className="text-ink-dim">No stairs needed - target is on the entry floor</span>
            )}
            {approach.target_floor_label ? (
              <span className="text-ink-dim"> to Floor <span className="num">{approach.target_floor_label}</span></span>
            ) : null}
          </dd>
          {otherEntrances.length > 1 ? (
            <>
              <dt className="text-ink-mute">Entrances</dt>
              <dd className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-ink-dim">
                {otherEntrances.map(([id, s]) => {
                  const name = building.entrances.find((e) => e.id === id)?.name ?? id;
                  return (
                    <span key={id} className={id === approach.entrance_id ? "text-ink" : ""} title="Probability-weighted walking time from this entrance">
                      {name} <span className="num">{mmss(s)}</span>
                    </span>
                  );
                })}
              </dd>
            </>
          ) : null}
        </dl>

        <div className="border-t border-line pt-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="eyebrow">Expected time to find, E[T]</span>
            <span className={`num text-xs font-semibold ${saved > 0.005 ? "text-prob" : "text-ink-mute"}`}>
              {saved > 0.005
                ? `${Math.round(saved * 100)}% faster than best baseline`
                : saved < -0.005
                  ? `${Math.round(-saved * 100)}% slower than best baseline`
                  : "On par with best baseline"}
            </span>
          </div>
          <ul className="space-y-1.5">
            {bars.map((b) => (
              <li key={b.key} className="grid grid-cols-[7.5rem_1fr_3.25rem] items-center gap-2 text-xs">
                <span className={b.primary ? "font-medium text-ink" : "text-ink-dim"}>{b.label}</span>
                <span className="block h-2.5 overflow-hidden rounded-sm bg-line" aria-hidden>
                  <span
                    className={`block h-full rounded-sm transition-[width] duration-300 ${b.primary ? "bg-accent" : "bg-ink-mute/70"}`}
                    style={{ width: `${(b.value / barMax) * 100}%` }}
                  />
                </span>
                <span className={`num text-right ${b.primary ? "font-semibold text-ink" : "text-ink-dim"}`} title={`${b.value.toFixed(0)} s`}>
                  {b.value.toFixed(0)} s
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  );
}

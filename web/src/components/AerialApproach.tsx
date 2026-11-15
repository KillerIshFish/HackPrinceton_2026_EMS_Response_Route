import { useMemo, useState } from "react";
import { Ban, Footprints, MoveVertical, Navigation, TriangleAlert } from "lucide-react";
import type { Approach, Building, BuildingMeta, Policy, SearchPlan, XY } from "../types";
import { buildingBounds, mmss, titleCase } from "../lib";
import { Panel } from "./Panel";
import { AerialSite, STAGING_COLOR } from "./AerialSite";

interface Props {
  building: Building;
  plan: SearchPlan;
  approach: Approach;
  /** provenance + site annotation; aerial imagery is shown when it carries a tile window */
  meta: BuildingMeta | null;
  /**
   * What the console may show. route_enabled false (active threat, confirmed or only suggested):
   * the aerial image and building outline only, with no entrance, staging spot, stairs or path.
   * elevator_allowed false (fire, smoke, alarm): no elevator option.
   */
  policy: Policy;
}

// Schematic site plan (no aerial imagery), in metres of the building's own coordinates. Display values, not data:
// the parking lot to the north and the street to the south are drawn as context, not taken from a map.
const S = 6; // svg units per metre
const SITE_MARGIN_M = 6; // context drawn around the outermost feature (m)
const BAND_M = 14; // minimum depth of the parking-lot / street bands (m)

/** entrances listed by walking time; the rest are summarised */
const MAX_LISTED_ENTRANCES = 8;

/** Site view (Vexcel aerial imagery when available, otherwise a schematic) + approach recommendation. */
export function AerialApproach({ building, plan, approach, meta, policy }: Props) {
  const tiles = meta?.site.tiles ?? null;
  /** false: nothing below may draw or name an entrance, staging spot, stairs or route */
  const routeOn = policy.route_enabled;
  const hasMap = !!tiles && (meta?.site.surroundings?.features.length ?? 0) > 0;
  const [siteView, setSiteView] = useState<"aerial" | "map">("aerial");
  const mode = hasMap ? siteView : "aerial";
  const geo = useMemo(() => {
    const fp = buildingBounds(building);
    const pts: XY[] = routeOn
      ? [...building.entrances.map((e) => e.xy), ...building.staging_spots.map((s) => s.xy)]
      : [];
    const minX = Math.min(fp.minX, ...pts.map((p) => p[0])) - SITE_MARGIN_M;
    const maxX = Math.max(fp.maxX, ...pts.map((p) => p[0])) + SITE_MARGIN_M;
    const minY = Math.min(fp.minY - BAND_M, ...pts.map((p) => p[1] - SITE_MARGIN_M));
    const maxY = Math.max(fp.maxY + BAND_M, ...pts.map((p) => p[1] + SITE_MARGIN_M));
    return { fp, minX, maxX, minY, maxY };
  }, [building, routeOn]);

  const { fp, minX, maxX, minY, maxY } = geo;
  const W = (maxX - minX) * S;
  const H = (maxY - minY) * S;
  const X = (x: number) => (x - minX) * S;
  const Y = (y: number) => (maxY - y) * S; // north up

  const recEntrance = routeOn ? building.entrances.find((e) => e.id === approach.entrance_id) : undefined;
  const recStaging = routeOn ? building.staging_spots.find((s) => s.id === approach.staging_id) : undefined;

  // parking stall ticks across the north band: a row 5 to 10 m north of the building, a tick every 3 m (schematic)
  const stallY0 = Y(fp.maxY + 5);
  const stallY1 = Y(fp.maxY + 10);
  const stalls: number[] = [];
  for (let x = Math.ceil(minX / 3) * 3 + 3; x < maxX - 2; x += 3) stalls.push(X(x));

  // E[T] gain over the better of the two baselines, as a share of it. Within +/-0.5% the headline says "on par":
  // that is where the rounded percentage would read 0%.
  const best = Math.min(plan.baseline_nearest_s, plan.baseline_highest_prob_s);
  const saved = best > 0 ? (best - plan.expected_time_to_find_s) / best : 0;
  const bars = [
    { key: "plan", label: "This plan", value: plan.expected_time_to_find_s, primary: true },
    { key: "nearest", label: "Nearest-first", value: plan.baseline_nearest_s, primary: false },
    { key: "highest", label: "Highest-prob-first", value: plan.baseline_highest_prob_s, primary: false },
  ];
  const barMax = Math.max(1, ...bars.map((b) => b.value));

  /** a property whose buildings could not be placed on the map: only the address point is real */
  const unplaced = meta?.site.layout_placed === false;
  const otherEntrances = Object.entries(approach.entrance_expected_travel_s).sort((a, b) => a[1] - b[1]);
  const stairsS = approach.stairs_s_to_target_floor ?? null;
  const elevatorS = approach.elevator_s_to_target_floor ?? null;
  /** an elevator is offered only when policy allows it and the backend computed a time */
  const elevatorOption = policy.elevator_allowed && policy.travel.elevator_allowed && elevatorS !== null;
  const entranceWarnings = approach.entrance_warnings?.[approach.entrance_id] ?? [];

  return (
    <Panel
      title="Site and approach"
      icon={<Navigation size={15} aria-hidden />}
      actions={
        hasMap ? (
          <div className="flex overflow-hidden rounded-sm border border-line text-[11px]" role="group" aria-label="Site view">
            {(["aerial", "map"] as const).map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={mode === v}
                onClick={() => setSiteView(v)}
                className={`px-2 py-0.5 ${mode === v ? "bg-accent font-semibold text-[#1a1204]" : "text-ink-dim hover:text-ink"}`}
              >
                {v === "aerial" ? "Aerial" : "Map"}
              </button>
            ))}
          </div>
        ) : (
          <span className="text-[11px] text-ink-mute">{tiles ? "Vexcel aerial imagery" : "Schematic site plan - no aerial imagery for this building"}</span>
        )
      }
      bodyClassName="grid gap-0"
    >
      {/* site schematic */}
      <div className="border-b border-line p-2">
        {tiles && meta ? (
          <AerialSite building={building} meta={meta} tiles={tiles} approach={approach} mode={mode} showApproach={routeOn} />
        ) : (<>
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block h-auto max-h-[340px] w-full xl:max-h-[clamp(300px,calc(100vh_-_500px),340px)]" role="img" aria-label={routeOn ? "Schematic site plan: parking lot to the north, street to the south, entrances and staging spots" : "Schematic site plan: building outline only, route planning is off"}>
          {/* ground */}
          <rect width={W} height={H} fill="#0e151d" />
          {/* parking lot (north): from 2 m north of the building to the top edge */}
          <rect x={0} y={0} width={W} height={Y(fp.maxY + 2)} fill="#131c27" />
          {stalls.map((sx) => (
            <line key={sx} x1={sx} y1={stallY0} x2={sx} y2={stallY1} stroke="#2b3a4d" strokeWidth={1} />
          ))}
          <line x1={X(minX + 3)} y1={stallY0} x2={X(maxX - 3)} y2={stallY0} stroke="#2b3a4d" strokeWidth={1} />
          <text x={8} y={14} fontSize="9.5" fontWeight="600" letterSpacing="1" fill="#8394a6">
            N - PARKING LOT
          </text>
          {/* street (south): from 5 m south of the building to the bottom edge, centre line dashed */}
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

          {/* staging spots (none while route planning is off) */}
          {(routeOn ? building.staging_spots : []).map((s) => {
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
                  fill={rec ? STAGING_COLOR : "#0e151d"}
                  stroke={rec ? STAGING_COLOR : "#62758a"}
                  strokeWidth={1.3}
                />
                <text x={cx} y={cy + 3.5} textAnchor="middle" fontSize="8.5" fontWeight="700" fill={rec ? "#052e16" : "#a7b6c5"}>
                  STG
                </text>
              </g>
            );
          })}

          {/* entrances (none while route planning is off) */}
          {(routeOn ? building.entrances : []).map((e) => {
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
          {routeOn ? (
            <>
              <span className="flex items-center gap-1.5">
                <span className="block h-2 w-4 bg-accent" aria-hidden /> Recommended entrance
              </span>
              <span className="flex items-center gap-1.5">
                <span className="block h-2 w-4" style={{ background: STAGING_COLOR }} aria-hidden /> Recommended staging
              </span>
              <span className="flex items-center gap-1.5">
                <span className="block h-2 w-4 bg-[#62758a]" aria-hidden /> Other entrance
              </span>
            </>
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="block h-2 w-4 border border-[#4a5d73] bg-[#1b2633]" aria-hidden /> Building outline only
            </span>
          )}
        </div>
        </>)}
        {/* a generated building: where its outline comes from and what about it is generated or assumed */}
        {meta?.kind === "generated" && (meta.caveats ?? []).length > 0 ? (
          <ul className="mt-1.5 list-disc space-y-0.5 pl-5 pr-1 text-[11px] leading-snug text-ink-dim" aria-label="What this model is based on">
            {meta.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        ) : null}
      </div>

      {/* recommendation + E[T] comparison; while route planning is off, only why there is none */}
      {!routeOn ? (
        <RouteOff boardShown={policy.show_incident_board} />
      ) : (
      <div className="grid gap-3 p-3 md:grid-cols-2">
        <dl className="grid grid-cols-[auto_1fr] content-start gap-x-3 gap-y-1.5 text-[13px]">
          <dt className="text-ink-mute">Stage at</dt>
          <dd className="text-ink">
            <span className="font-medium">{unplaced ? "The address point" : approach.staging_name}</span>
            <span className="num ml-2 text-[11px] text-ink-dim">
              {approach.staging_latlng[0].toFixed(5)}, {approach.staging_latlng[1].toFixed(5)}
            </span>
          </dd>
          <dt className="text-ink-mute">Enter via</dt>
          <dd className="font-medium text-accent">
            {approach.entrance_name}
            {unplaced ? <span className="ml-2 text-[11px] font-normal text-ink-mute">its position on the site is not known</span> : null}
          </dd>
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
          {approach.stairwell_name && stairsS !== null ? (
            <>
              <dt className="text-ink-mute">Stairs</dt>
              <dd className="flex items-center gap-1.5 text-ink">
                <Footprints size={13} className="text-ink-mute" aria-hidden />
                <span className="num">{mmss(stairsS)}</span>
                <span className="text-[11px] text-ink-dim">entrance floor to Floor {approach.target_floor_label}</span>
              </dd>
            </>
          ) : null}
          {elevatorOption ? (
            <>
              <dt className="text-ink-mute">Elevator</dt>
              <dd className="flex items-center gap-1.5 text-ink">
                <MoveVertical size={13} className="text-ink-mute" aria-hidden />
                <span className="num">{mmss(elevatorS!)}</span>
                <span className="rounded border border-line-strong px-1 text-[11px] text-ink-dim">Crew's choice</span>
              </dd>
            </>
          ) : !policy.elevator_allowed ? (
            <>
              <dt className="text-ink-mute">Stairs only</dt>
              <dd className="flex items-center gap-1.5 text-[12px] text-ink-dim">
                <Ban size={13} className="shrink-0 text-ink-mute" aria-hidden />
                Elevator routes are not offered while fire, smoke or an alarm is reported
              </dd>
            </>
          ) : null}
          {entranceWarnings.length > 0 ? (
            <>
              <dt className="text-ink-mute">Entrance</dt>
              <dd className="space-y-0.5">
                {entranceWarnings.map((w) => (
                  <span key={w} className="flex items-center gap-1.5 text-[12px] text-ink">
                    <TriangleAlert size={13} className="shrink-0 text-warn" aria-hidden />
                    {w}
                  </span>
                ))}
              </dd>
            </>
          ) : null}
          {otherEntrances.length > 1 ? (
            <>
              <dt className="text-ink-mute">Entrances</dt>
              <dd className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-ink-dim">
                {otherEntrances.slice(0, MAX_LISTED_ENTRANCES).map(([id, s]) => {
                  const name = building.entrances.find((e) => e.id === id)?.name ?? id;
                  return (
                    <span key={id} className={id === approach.entrance_id ? "text-ink" : ""} title="Probability-weighted walking time from this entrance">
                      {name} <span className="num">{mmss(s)}</span>
                    </span>
                  );
                })}
                {otherEntrances.length > MAX_LISTED_ENTRANCES ? (
                  <span className="text-ink-mute">and {otherEntrances.length - MAX_LISTED_ENTRANCES} more, slower</span>
                ) : null}
              </dd>
            </>
          ) : null}
        </dl>

        <div className="border-t border-line pt-3 md:border-l md:border-t-0 md:pl-3 md:pt-0">
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
        {(approach.warnings ?? []).length > 0 ? (
          <ul className="space-y-0.5 border-t border-line pt-2">
            {(approach.warnings ?? []).map((w) => (
              <li key={w} className="flex items-start gap-1.5 text-[12px] text-ink-dim">
                <TriangleAlert size={13} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                {w}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      )}
    </Panel>
  );
}

/** Right-hand side while route planning is off: why nothing is suggested, and where reports go. */
function RouteOff({ boardShown }: { boardShown: boolean }) {
  return (
    <div className="flex flex-col gap-2 p-3 text-[13px]">
      <p className="flex items-center gap-2 font-semibold text-ink">
        <Ban size={15} className="shrink-0 text-alert" aria-hidden />
        Route planning is off for this incident
      </p>
      <p className="text-ink-dim">
        No entrance, staging spot, stairs or path is suggested. The view above shows the site and the building outline
        as information.
      </p>
      <p className="text-ink-dim">
        {boardShown
          ? "Caller reports are on the incident board, for relay to unified command."
          : "An active threat has been suggested from the call. Confirm or dismiss it in the situation bar."}
      </p>
    </div>
  );
}

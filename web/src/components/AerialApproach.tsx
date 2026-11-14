import { useMemo } from "react";
import { Ban, Footprints, MoveVertical, Navigation, TriangleAlert } from "lucide-react";
import { api } from "../api";
import type { Approach, Building, BuildingMeta, LatLng, Policy, SearchPlan, SiteTiles, XY } from "../types";
import { buildingBounds, latLngToTile, mmss, titleCase } from "../lib";
import { Panel } from "./Panel";
import { AerialSite } from "./AerialSite";

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

const S = 6; // svg units per metre
const SITE_MARGIN_M = 6; // context drawn around the outermost feature
const BAND_M = 14; // minimum depth of the parking-lot / street bands

/** Site view (Vexcel aerial imagery when available, otherwise a schematic) + approach recommendation. */
/** entrances listed by walking time; the rest are summarised */
const MAX_LISTED_ENTRANCES = 8;

export function AerialApproach({ building, plan, approach, meta, policy }: Props) {
  const tiles = meta?.site.tiles ?? null;
  /** false: nothing below may draw or name an entrance, staging spot, stairs or route */
  const routeOn = policy.route_enabled;
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
      title="Approach"
      icon={<Navigation size={15} aria-hidden />}
      actions={<span className="text-[11px] text-ink-mute">{tiles ? "Vexcel aerial imagery" : "Schematic site plan - no aerial imagery for this building"}</span>}
      bodyClassName="grid gap-0 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]"
    >
      {/* site schematic */}
      <div className="border-b border-line p-2 lg:border-b-0 lg:border-r">
        {tiles && meta ? (
          routeOn ? (
            <AerialSite building={building} meta={meta} tiles={tiles} approach={approach} />
          ) : (
            <AerialOutline building={building} meta={meta} tiles={tiles} />
          )
        ) : (<>
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block h-auto max-h-[340px] w-full" role="img" aria-label={routeOn ? "Schematic site plan: parking lot to the north, street to the south, entrances and staging spots" : "Schematic site plan: building outline only, route planning is off"}>
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
                <span className="block h-2 w-4 bg-accent" aria-hidden /> Recommended entrance / staging
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
      </div>

      {/* recommendation + E[T] comparison; while route planning is off, only why there is none */}
      {!routeOn ? (
        <RouteOff boardShown={policy.show_incident_board} />
      ) : (
      <div className="flex flex-col gap-3 p-3">
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-[13px]">
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
        No entrance, staging spot, stairs or path is suggested. The image shows the site and the building outline as
        information.
      </p>
      <p className="text-ink-dim">
        {boardShown
          ? "Caller reports are on the incident board, for relay to unified command."
          : "An active threat has been suggested from the call. Confirm or dismiss it in the situation bar."}
      </p>
    </div>
  );
}

/**
 * The aerial image with the building outline and nothing else: the route-off counterpart of
 * AerialSite, which always draws entrances and staging spots.
 */
function AerialOutline({ building, meta, tiles }: { building: Building; meta: BuildingMeta; tiles: SiteTiles }) {
  const px = ([lat, lng]: LatLng): [number, number] => {
    const [tx, ty] = latLngToTile(lat, lng, tiles.zoom);
    return [(tx - tiles.x_min) * tiles.tile_px, (ty - tiles.y_min) * tiles.tile_px];
  };
  const W = (tiles.x_max - tiles.x_min + 1) * tiles.tile_px;
  const H = (tiles.y_max - tiles.y_min + 1) * tiles.tile_px;
  const given = meta.site.outlines ?? [];
  const outlines = (given.length > 0 ? given : [{ kind: "annotated", latlng: meta.site.footprint_latlng }])
    .filter((o) => o.latlng.length >= 3)
    .map((o) => ({ kind: o.kind, points: o.latlng.map(px) }));
  const pts = [...outlines.flatMap((o) => o.points), ...(meta.site.address_latlng ? [px(meta.site.address_latlng)] : [])];
  let view = { x: 0, y: 0, w: W, h: H };
  if (pts.length > 0) {
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const side = Math.min(Math.max(extent * 1.7, extent + 30 / tiles.m_per_px), W, H);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    view = { x: Math.min(Math.max(cx - side / 2, 0), W - side), y: Math.min(Math.max(cy - side / 2, 0), H - side), w: side, h: side };
  }
  const tilesInView: { key: string; src: string; x: number; y: number }[] = [];
  for (let ty = tiles.y_min; ty <= tiles.y_max; ty++)
    for (let tx = tiles.x_min; tx <= tiles.x_max; tx++) {
      const x = (tx - tiles.x_min) * tiles.tile_px;
      const y = (ty - tiles.y_min) * tiles.tile_px;
      if (x + tiles.tile_px < view.x || x > view.x + view.w || y + tiles.tile_px < view.y || y > view.y + view.h) continue;
      tilesInView.push({ key: `${tx}-${ty}`, src: api.tileUrl(tiles.layer, tiles.zoom, tx, ty), x, y });
    }
  const u = view.w / 560;
  return (
    <div>
      <div className="mx-auto max-h-[420px] overflow-hidden rounded-sm bg-bg" style={{ aspectRatio: `${view.w} / ${view.h}` }}>
        <svg
          viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
          className="block h-full w-full"
          role="img"
          aria-label={`Aerial image of ${meta.name ?? building.name} with the building outline only. Route planning is off. North is up.`}
        >
          <rect x={view.x} y={view.y} width={view.w} height={view.h} fill="#0e151d" />
          {tilesInView.map((t) => (
            <image key={t.key} href={t.src} x={t.x} y={t.y} width={tiles.tile_px + 0.5} height={tiles.tile_px + 0.5} preserveAspectRatio="none" />
          ))}
          {outlines.map((o, i) => (
            <polygon
              key={i}
              points={o.points.map((p) => p.join(",")).join(" ")}
              fill="none"
              stroke="#e6edf3"
              strokeWidth={(o.kind === "model" ? 1.6 : 3) * u}
              strokeDasharray={o.kind === "model" ? `${7 * u} ${5 * u}` : undefined}
              strokeLinejoin="round"
            />
          ))}
          <text x={view.x + view.w - 26 * u} y={view.y + 30 * u} textAnchor="middle" fontSize={13 * u} fontWeight="700" fill="#e6edf3" stroke="#0b1016" strokeWidth={3 * u} paintOrder="stroke">
            N
          </text>
        </svg>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-ink-mute">
        <span className="flex items-center gap-1.5">
          <span className="block h-2 w-4 border-2 border-ink" aria-hidden /> Building outline only
        </span>
        <span className="ml-auto">Imagery: Vexcel</span>
      </div>
    </div>
  );
}

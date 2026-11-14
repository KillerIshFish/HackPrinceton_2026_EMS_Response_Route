import { useMemo } from "react";
import { api } from "../api";
import type { Approach, Building, BuildingMeta, LatLng, SiteTiles } from "../types";
import { latLngToTile } from "../lib";

interface Props {
  building: Building;
  meta: BuildingMeta;
  tiles: SiteTiles;
  approach: Approach;
}

/**
 * Vexcel ortho imagery of the site (tiles proxied and cached by the backend) with the
 * annotated roof outline, entrances, staging spots and the recommended approach drawn on top.
 */
export function AerialSite({ building, meta, tiles, approach }: Props) {
  const cols = tiles.x_max - tiles.x_min + 1;
  const rows = tiles.y_max - tiles.y_min + 1;
  const W = cols * tiles.tile_px;
  const H = rows * tiles.tile_px;

  const px = useMemo(
    () =>
      ([lat, lng]: LatLng): [number, number] => {
        const [tx, ty] = latLngToTile(lat, lng, tiles.zoom);
        return [(tx - tiles.x_min) * tiles.tile_px, (ty - tiles.y_min) * tiles.tile_px];
      },
    [tiles],
  );

  const tileList = useMemo(() => {
    const out: { key: string; src: string }[] = [];
    for (let y = tiles.y_min; y <= tiles.y_max; y++)
      for (let x = tiles.x_min; x <= tiles.x_max; x++)
        out.push({ key: `${x}-${y}`, src: api.tileUrl(tiles.layer, tiles.zoom, x, y) });
    return out;
  }, [tiles]);

  const footprint = meta.site.footprint_latlng.map((ll) => px(ll).join(",")).join(" ");
  const recEntrance = building.entrances.find((e) => e.id === approach.entrance_id);
  const recStaging = building.staging_spots.find((s) => s.id === approach.staging_id);
  const scale = W / 560; // overlay sizes are tuned for a ~560 px wide rendering
  const scaleBarM = 25;
  const scaleBarPx = scaleBarM / tiles.m_per_px;

  return (
    <div>
      <div className="relative mx-auto aspect-square max-h-[420px] overflow-hidden rounded-sm bg-bg">
        <div
          className="absolute inset-0 grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)` }}
          aria-hidden
        >
          {tileList.map((t) => (
            <img key={t.key} src={t.src} alt="" loading="lazy" draggable={false} className="block h-full w-full" />
          ))}
        </div>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="absolute inset-0 h-full w-full"
          role="img"
          aria-label={`Aerial image of ${meta.name ?? building.name} with entrances, staging spots and the recommended approach. North is up.`}
        >
          {footprint ? (
            <polygon points={footprint} fill="#f2a93b" fillOpacity={0.1} stroke="#f2a93b" strokeWidth={3 * scale} strokeLinejoin="round" />
          ) : null}

          {recEntrance && recStaging ? (
            <line
              x1={px(recStaging.latlng)[0]}
              y1={px(recStaging.latlng)[1]}
              x2={px(recEntrance.latlng)[0]}
              y2={px(recEntrance.latlng)[1]}
              stroke="#f2a93b"
              strokeWidth={4 * scale}
              strokeDasharray={`${10 * scale} ${7 * scale}`}
              strokeLinecap="round"
            />
          ) : null}

          {building.staging_spots.map((s) => {
            const rec = s.id === approach.staging_id;
            const [cx, cy] = px(s.latlng);
            return (
              <g key={s.id}>
                <title>{`${s.name}${rec ? " (recommended staging)" : ""}`}</title>
                <rect
                  x={cx - 17 * scale}
                  y={cy - 10 * scale}
                  width={34 * scale}
                  height={20 * scale}
                  rx={3 * scale}
                  fill={rec ? "#f2a93b" : "#0b1016"}
                  stroke={rec ? "#0b1016" : "#e6edf3"}
                  strokeWidth={1.5 * scale}
                />
                <text x={cx} y={cy + 5 * scale} textAnchor="middle" fontSize={13 * scale} fontWeight="700" fill={rec ? "#1a1204" : "#e6edf3"}>
                  STG
                </text>
              </g>
            );
          })}

          {building.entrances.map((e) => {
            const rec = e.id === approach.entrance_id;
            const [cx, cy] = px(e.latlng);
            return (
              <g key={e.id}>
                <title>{`${e.name}${rec ? " (recommended entrance)" : ""}`}</title>
                <circle cx={cx} cy={cy} r={(rec ? 9 : 7) * scale} fill={rec ? "#f2a93b" : "#e6edf3"} stroke="#0b1016" strokeWidth={2 * scale} />
                <text
                  x={cx + 13 * scale}
                  y={cy + 5 * scale}
                  fontSize={13.5 * scale}
                  fontWeight={rec ? 700 : 600}
                  fill={rec ? "#f2a93b" : "#e6edf3"}
                  stroke="#0b1016"
                  strokeWidth={3.5 * scale}
                  paintOrder="stroke"
                >
                  {e.name}
                </text>
              </g>
            );
          })}

          {/* north arrow + scale bar */}
          <g fill="#e6edf3" stroke="#0b1016" strokeWidth={3 * scale} paintOrder="stroke" fontSize={13 * scale} fontWeight="700">
            <text x={W - 26 * scale} y={30 * scale} textAnchor="middle">
              N
            </text>
            <text x={18 * scale} y={H - 26 * scale}>
              {scaleBarM} m
            </text>
          </g>
          <path
            d={`M ${W - 26 * scale} ${36 * scale} l ${-7 * scale} ${16 * scale} l ${7 * scale} ${-5 * scale} l ${7 * scale} ${5 * scale} z`}
            fill="#e6edf3"
            stroke="#0b1016"
            strokeWidth={1.2 * scale}
          />
          <rect x={18 * scale} y={H - 20 * scale} width={scaleBarPx} height={5 * scale} fill="#e6edf3" stroke="#0b1016" strokeWidth={1.2 * scale} />
        </svg>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-ink-mute">
        <span className="flex items-center gap-1.5">
          <span className="block h-2 w-4 bg-accent" aria-hidden /> Recommended entrance / staging
        </span>
        <span className="flex items-center gap-1.5">
          <span className="block h-2 w-4 border-2 border-accent" aria-hidden /> Roof outline
        </span>
        <span className="ml-auto">Imagery: Vexcel - positions annotated by hand, approximate</span>
      </div>
    </div>
  );
}

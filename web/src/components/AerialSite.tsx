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

type Box = [number, number, number, number]; // x0, y0, x1, y1 in mosaic pixels

/** with more entrances than this, only the recommended one is named on the image */
const MAX_NAMED_ENTRANCES = 6;

const overlaps = (a: Box, b: Box) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];

/** A 1-2-5 length in metres whose bar is about a fifth of the view wide. */
function niceScale(viewMetres: number): number {
  const target = viewMetres / 5;
  const pow = 10 ** Math.floor(Math.log10(target));
  const step = [1, 2, 5, 10].find((s) => s * pow >= target) ?? 10;
  return step * pow;
}

/**
 * Vexcel ortho imagery of the site with the building outline, entrances, staging spots and
 * the recommended approach drawn on top.
 *
 * Imagery and overlay live in one SVG and one coordinate system (mosaic pixels), so they
 * cannot drift apart whatever size or shape the panel is given. The view is a window on the
 * mosaic fitted to what is drawn, and every overlay size is a fraction of that window.
 */
export function AerialSite({ building, meta, tiles, approach }: Props) {
  const W = (tiles.x_max - tiles.x_min + 1) * tiles.tile_px;
  const H = (tiles.y_max - tiles.y_min + 1) * tiles.tile_px;

  const px = useMemo(
    () =>
      ([lat, lng]: LatLng): [number, number] => {
        const [tx, ty] = latLngToTile(lat, lng, tiles.zoom);
        return [(tx - tiles.x_min) * tiles.tile_px, (ty - tiles.y_min) * tiles.tile_px];
      },
    [tiles],
  );

  // What to outline: the real building (OpenStreetMap) and the model's extent when the
  // backend supplies them, otherwise the single annotated footprint.
  const outlines = useMemo(() => {
    const given = meta.site.outlines ?? [];
    const list = given.length > 0 ? given : [{ kind: "annotated", latlng: meta.site.footprint_latlng }];
    return list
      .filter((o) => o.latlng.length >= 3)
      .map((o) => ({ kind: o.kind, source: "source" in o ? o.source : undefined, points: o.latlng.map(px) }));
  }, [meta, px]);
  const real = outlines.find((o) => o.kind === "building");
  const hasReal = !!real;

  // Where the layout's positions on the ground are invented (several buildings on one
  // property), entrances and staging are not drawn: only the address point is real.
  const placed = meta.site.layout_placed !== false;
  const entranceList = useMemo(() => (placed ? building.entrances : []), [placed, building]);
  const stagingList = useMemo(() => (placed ? building.staging_spots : []), [placed, building]);
  const address = !placed && meta.site.address_latlng ? px(meta.site.address_latlng) : null;
  const recEntrance = entranceList.find((e) => e.id === approach.entrance_id);
  const recStaging = stagingList.find((s) => s.id === approach.staging_id);

  // ---- view window: fitted to everything drawn, square, kept inside the mosaic ----------
  const view = useMemo(() => {
    const pts: [number, number][] = [
      ...outlines.flatMap((o) => o.points),
      ...entranceList.map((e) => px(e.latlng)),
      ...stagingList.map((s) => px(s.latlng)),
      ...(address ? [address] : []),
    ];
    if (pts.length === 0) return { x: 0, y: 0, w: W, h: H };
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    // room around the building for the street and the neighbours, never more than the mosaic holds
    const side = Math.min(Math.max(extent * 1.7, extent + 30 / tiles.m_per_px), W, H);
    const x = Math.min(Math.max(cx - side / 2, 0), W - side);
    const y = Math.min(Math.max(cy - side / 2, 0), H - side);
    return { x, y, w: side, h: side };
  }, [outlines, entranceList, stagingList, address, px, W, H, tiles.m_per_px]);

  const tileList = useMemo(() => {
    const out: { key: string; src: string; x: number; y: number }[] = [];
    for (let ty = tiles.y_min; ty <= tiles.y_max; ty++)
      for (let tx = tiles.x_min; tx <= tiles.x_max; tx++) {
        const x = (tx - tiles.x_min) * tiles.tile_px;
        const y = (ty - tiles.y_min) * tiles.tile_px;
        // only the tiles the window can see are requested
        if (x + tiles.tile_px < view.x || x > view.x + view.w || y + tiles.tile_px < view.y || y > view.y + view.h) continue;
        out.push({ key: `${tx}-${ty}`, src: api.tileUrl(tiles.layer, tiles.zoom, tx, ty), x, y });
      }
    return out;
  }, [tiles, view]);

  const u = view.w / 560; // one overlay unit: sizes below are in "pixels of a 560 px wide view"
  const fontPx = 13 * u;

  // ---- markers and labels, placed so that none covers another ---------------------------
  const layout = useMemo(() => {
    const taken: Box[] = [];
    const inView = (b: Box) => b[0] >= view.x && b[1] >= view.y && b[2] <= view.x + view.w && b[3] <= view.y + view.h;
    const entrances = entranceList.map((e) => {
      const [cx, cy] = px(e.latlng);
      const rec = e.id === approach.entrance_id;
      return { e, cx, cy, rec, r: (rec ? 9 : 6) * u };
    });
    for (const m of entrances) taken.push([m.cx - m.r, m.cy - m.r, m.cx + m.r, m.cy + m.r]);

    // staging badge: at its own spot unless that sits on the entrance marker, then pushed clear of it
    let badge: { cx: number; cy: number; w: number; h: number } | null = null;
    if (recStaging) {
      const [sx, sy] = px(recStaging.latlng);
      const w = 34 * u;
      const h = 20 * u;
      let cx = sx;
      let cy = sy;
      const rec = entrances.find((m) => m.rec);
      if (rec) {
        const dx = sx - rec.cx;
        const dy = sy - rec.cy;
        const d = Math.hypot(dx, dy);
        const clear = rec.r + Math.hypot(w, h) / 2 + 3 * u;
        if (d < clear) {
          const ux = d > 1e-6 ? dx / d : 0;
          const uy = d > 1e-6 ? dy / d : 1;
          cx = rec.cx + ux * clear;
          cy = rec.cy + uy * clear;
        }
      }
      badge = { cx, cy, w, h };
      taken.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    }

    // labels: the recommended entrance first, then the others while there is room
    const labels: { key: string; x: number; y: number; anchor: "start" | "end" | "middle"; text: string; rec: boolean }[] = [];
    const ordered = [...entrances].sort((a, b) => Number(b.rec) - Number(a.rec));
    const nameAll = entrances.length <= MAX_NAMED_ENTRANCES;
    for (const m of ordered) {
      if (!m.rec && !nameAll) continue; // many entrances: markers only, names are in the list beside the map
      const text = m.e.name;
      const tw = text.length * fontPx * 0.56;
      const th = fontPx * 1.25;
      const gap = m.r + 4 * u;
      const options: { box: Box; x: number; y: number; anchor: "start" | "end" | "middle" }[] = [
        { box: [m.cx + gap, m.cy - th / 2, m.cx + gap + tw, m.cy + th / 2], x: m.cx + gap, y: m.cy + fontPx * 0.36, anchor: "start" },
        { box: [m.cx - gap - tw, m.cy - th / 2, m.cx - gap, m.cy + th / 2], x: m.cx - gap, y: m.cy + fontPx * 0.36, anchor: "end" },
        { box: [m.cx - tw / 2, m.cy - gap - th, m.cx + tw / 2, m.cy - gap], x: m.cx, y: m.cy - gap - th * 0.22, anchor: "middle" },
        { box: [m.cx - tw / 2, m.cy + gap, m.cx + tw / 2, m.cy + gap + th], x: m.cx, y: m.cy + gap + th * 0.78, anchor: "middle" },
      ];
      const fit = options.find((o) => inView(o.box) && !taken.some((t) => overlaps(o.box, t)));
      if (!fit) continue; // no free spot: the marker keeps its tooltip, the name stays in the list beside the map
      taken.push(fit.box);
      labels.push({ key: m.e.id, x: fit.x, y: fit.y, anchor: fit.anchor, text, rec: m.rec });
    }
    return { entrances, badge, labels };
  }, [entranceList, approach, recStaging, px, u, fontPx, view]);

  const scaleM = niceScale(view.w * tiles.m_per_px);
  const scalePx = scaleM / tiles.m_per_px;
  const placement = meta.site.placement_source;
  const generated = meta.site.source === "generated";

  return (
    <div>
      <div className="mx-auto max-h-[420px] overflow-hidden rounded-sm bg-bg" style={{ aspectRatio: `${view.w} / ${view.h}` }}>
        <svg
          viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
          className="block h-full w-full"
          role="img"
          aria-label={`Aerial image of ${meta.name ?? building.name} with entrances, staging spots and the recommended approach. North is up.`}
        >
          <rect x={view.x} y={view.y} width={view.w} height={view.h} fill="#0e151d" />
          {tileList.map((t) => (
            <image key={t.key} href={t.src} x={t.x} y={t.y} width={tiles.tile_px + 0.5} height={tiles.tile_px + 0.5} preserveAspectRatio="none" />
          ))}

          {outlines.map((o, i) => {
            const points = o.points.map((p) => p.join(",")).join(" ");
            // the model's own extent is secondary once the real outline is known
            const secondary = o.kind === "model" && hasReal;
            if (secondary)
              return (
                <g key={i} fill="none" strokeLinejoin="round">
                  <polygon points={points} stroke="#0b1016" strokeOpacity={0.7} strokeWidth={4 * u} />
                  <polygon points={points} stroke="#e6edf3" strokeWidth={1.8 * u} strokeDasharray={`${7 * u} ${5 * u}`} />
                </g>
              );
            return (
              <polygon
                key={i}
                points={points}
                fill="#f2a93b"
                fillOpacity={secondary ? 0 : o.kind === "model" && generated ? 0.06 : 0.12}
                stroke={secondary ? "#e6edf3" : "#f2a93b"}
                strokeOpacity={secondary ? 0.75 : 1}
                strokeWidth={(secondary ? 1.6 : 3) * u}
                strokeDasharray={o.kind === "model" && generated ? `${7 * u} ${5 * u}` : undefined}
                strokeLinejoin="round"
              />
            );
          })}

          {recEntrance && recStaging ? (
            <line
              x1={px(recStaging.latlng)[0]}
              y1={px(recStaging.latlng)[1]}
              x2={px(recEntrance.latlng)[0]}
              y2={px(recEntrance.latlng)[1]}
              stroke="#f2a93b"
              strokeWidth={4 * u}
              strokeDasharray={`${10 * u} ${7 * u}`}
              strokeLinecap="round"
            />
          ) : null}

          {stagingList
            .filter((s) => s.id !== approach.staging_id)
            .map((s) => {
              const [cx, cy] = px(s.latlng);
              return (
                <rect key={s.id} x={cx - 4 * u} y={cy - 4 * u} width={8 * u} height={8 * u} fill="#0b1016" stroke="#e6edf3" strokeWidth={1.4 * u}>
                  <title>{s.name}</title>
                </rect>
              );
            })}

          {layout.entrances.map((m) => (
            <circle key={m.e.id} cx={m.cx} cy={m.cy} r={m.r} fill={m.rec ? "#f2a93b" : "#e6edf3"} stroke="#0b1016" strokeWidth={2 * u}>
              <title>{`${m.e.name}${m.rec ? " (recommended entrance)" : ""}`}</title>
            </circle>
          ))}

          {layout.badge && recStaging ? (
            <g>
              <title>{`${recStaging.name} (recommended staging)`}</title>
              <rect
                x={layout.badge.cx - layout.badge.w / 2}
                y={layout.badge.cy - layout.badge.h / 2}
                width={layout.badge.w}
                height={layout.badge.h}
                rx={3 * u}
                fill="#f2a93b"
                stroke="#0b1016"
                strokeWidth={1.5 * u}
              />
              <text x={layout.badge.cx} y={layout.badge.cy + fontPx * 0.36} textAnchor="middle" fontSize={fontPx} fontWeight="700" fill="#1a1204">
                STG
              </text>
            </g>
          ) : null}

          {layout.labels.map((l) => (
            <text
              key={l.key}
              x={l.x}
              y={l.y}
              textAnchor={l.anchor}
              fontSize={fontPx}
              fontWeight={l.rec ? 700 : 600}
              fill={l.rec ? "#f2a93b" : "#e6edf3"}
              stroke="#0b1016"
              strokeWidth={3.5 * u}
              paintOrder="stroke"
            >
              {l.text}
            </text>
          ))}

          {address ? (
            <g>
              <title>Address point of the property</title>
              <circle cx={address[0]} cy={address[1]} r={9 * u} fill="#f2a93b" stroke="#0b1016" strokeWidth={2 * u} />
              <circle cx={address[0]} cy={address[1]} r={3 * u} fill="#0b1016" />
              <text
                x={address[0] + 14 * u}
                y={address[1] + fontPx * 0.36}
                fontSize={fontPx}
                fontWeight="700"
                fill="#f2a93b"
                stroke="#0b1016"
                strokeWidth={3.5 * u}
                paintOrder="stroke"
              >
                Address point
              </text>
            </g>
          ) : null}

          {/* north arrow + scale bar, pinned to the view's corners */}
          <g fill="#e6edf3" stroke="#0b1016" strokeWidth={3 * u} paintOrder="stroke" fontSize={fontPx} fontWeight="700">
            <text x={view.x + view.w - 26 * u} y={view.y + 30 * u} textAnchor="middle">
              N
            </text>
            <text x={view.x + 18 * u} y={view.y + view.h - 26 * u}>
              {scaleM} m
            </text>
          </g>
          <path
            d={`M ${view.x + view.w - 26 * u} ${view.y + 36 * u} l ${-7 * u} ${16 * u} l ${7 * u} ${-5 * u} l ${7 * u} ${5 * u} z`}
            fill="#e6edf3"
            stroke="#0b1016"
            strokeWidth={1.2 * u}
          />
          <rect x={view.x + 18 * u} y={view.y + view.h - 20 * u} width={scalePx} height={5 * u} fill="#e6edf3" stroke="#0b1016" strokeWidth={1.2 * u} />
        </svg>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-ink-mute">
        <span className="flex items-center gap-1.5">
          <span className="block h-2 w-4 bg-accent" aria-hidden /> {placed ? "Recommended entrance / staging" : "Address point"}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="block h-2 w-4 border-2 border-accent" aria-hidden />
          {hasReal ? `Building outline (${real?.source ?? "surveyed"})` : generated ? "Generated layout extent" : "Building outline"}
        </span>
        {hasReal && outlines.some((o) => o.kind === "model") ? (
          <span className="flex items-center gap-1.5">
            <span className="block h-2 w-4 border border-dashed border-ink" aria-hidden /> {generated ? "Generated layout extent" : "Scan extent"}
          </span>
        ) : null}
        {!placed ? (
          <span className="basis-full text-ink-dim">
            {meta.site.buildings && meta.site.buildings > 1 ? `${meta.site.buildings} buildings on this property. ` : ""}
            Their positions are not in the data, so entrances and staging are not placed on the image: stage at the address point.
          </span>
        ) : null}
        <span className="ml-auto">
          Imagery: Vexcel ·{" "}
          {placement
            ? placed
              ? `placed by ${placement}${generated ? "; size and entrances are generated" : ""}`
              : placement
            : "positions annotated by hand, approximate"}
        </span>
      </div>
    </div>
  );
}

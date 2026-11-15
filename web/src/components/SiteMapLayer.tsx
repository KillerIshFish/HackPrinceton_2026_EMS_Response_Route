import { useMemo } from "react";
import type { LatLng, SiteSurroundings } from "../types";

type Pt = [number, number];
type Box = [number, number, number, number]; // x0, y0, x1, y1 in the parent SVG's coordinates

interface Props {
  surroundings: SiteSurroundings;
  /** lat/lng -> the parent SVG's coordinates (mosaic pixels) */
  px: (p: LatLng) => [number, number];
  view: { x: number; y: number; w: number; h: number };
  /** one overlay unit of the parent view */
  u: number;
  mPerPx: number;
  fontPx: number;
  /** what the overlay on top occupies (markers and their names): no place name is put under these */
  avoid?: Box[];
}

export const MAP_COLORS = {
  ground: "#111922",
  parking: "#27384b",
  residential: "#3f5a4a",
  building: "#33414f",
  road: "#5b6f86",
  roadCasing: "#0b1016",
  place: "#9fd0ff",
} as const;

/** drawn width of every street (m). Assumed: the features are OpenStreetMap centrelines with no width, so one
 * width of about a two-lane street is used for all of them. */
const ROAD_WIDTH_M = 9;
/** at most this many place names are written, nearest the building first (UI choice); the rest keep dot and tooltip */
const MAX_PLACE_LABELS = 8;

const overlaps = (a: Box, b: Box) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];

/** The stretches of a line that lie inside a box (each segment clipped, Liang-Barsky). */
function clipToBox(pts: Pt[], [x0, y0, x1, y1]: Box): Pt[][] {
  const runs: Pt[][] = [];
  let run: Pt[] = [];
  const close = () => {
    if (run.length > 1) runs.push(run);
    run = [];
  };
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1] as Pt;
    const b = pts[k] as Pt;
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    let t0 = 0;
    let t1 = 1;
    let inside = true;
    const edges: [number, number][] = [
      [-dx, a[0] - x0],
      [dx, x1 - a[0]],
      [-dy, a[1] - y0],
      [dy, y1 - a[1]],
    ];
    for (const [p, q] of edges) {
      if (p === 0) {
        if (q < 0) inside = false;
      } else if (p < 0) t0 = Math.max(t0, q / p);
      else t1 = Math.min(t1, q / p);
    }
    if (!inside || t0 > t1) {
      close();
      continue;
    }
    if (t0 > 0 || run.length === 0) {
      close();
      run.push([a[0] + t0 * dx, a[1] + t0 * dy]);
    }
    run.push([a[0] + t1 * dx, a[1] + t1 * dy]);
    if (t1 < 1) close();
  }
  close();
  return runs;
}

const pathOf = (pts: Pt[]) => "M " + pts.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" L ");

/**
 * The street map under the site overlay, drawn from the same OpenStreetMap features the
 * backend uses to decide what each window looks at: road centrelines, car parks, houses,
 * other buildings and named places. It shares the aerial image's coordinate system, so the
 * building outline, entrances and staging spots sit on it exactly as they do on the imagery.
 */
export function SiteMapLayer({ surroundings, px, view, u, mPerPx, fontPx, avoid }: Props) {
  const shapes = useMemo(() => {
    // a feature is kept when any point is in the view or within 20% of its size outside it (a road that crosses
    // the view between two far-off points can still be dropped; acceptable for a context map)
    const inView = (pts: [number, number][]) =>
      pts.some((p) => p[0] >= view.x - view.w * 0.2 && p[0] <= view.x + view.w * 1.2 && p[1] >= view.y - view.h * 0.2 && p[1] <= view.y + view.h * 1.2);
    const areas: { key: string; points: string; fill: string; title: string; cx: number; cy: number; cls: string }[] = [];
    const roads: { key: string; d: string; name: string }[] = [];
    // one label per street name, on its longest stretch inside the view, so that a name is
    // neither cut off at the edge nor centred on a part of the road that is out of sight
    const named = new Map<string, { key: string; name: string; labelD: string; length: number }>();
    const pad = view.w * 0.02; // street names stay 2% of the view in from its edges
    const inner: Box = [view.x + pad, view.y + pad, view.x + view.w - pad, view.y + view.h - pad];
    surroundings.features.forEach((f, i) => {
      if (f.own) return; // the modelled building is drawn by the outline on top
      const pts = f.latlng.map(px);
      if (pts.length < 2 || !inView(pts)) return;
      if (f.shape === "line") {
        roads.push({ key: `r${i}`, d: pathOf(pts), name: f.name });
        if (!f.name) return;
        for (const run of clipToBox(pts, inner)) {
          let length = 0;
          run.forEach((p, k) => {
            const q = run[k - 1];
            if (q) length += Math.hypot(p[0] - q[0], p[1] - q[1]);
          });
          if ((named.get(f.name)?.length ?? 0) >= length) continue;
          // names read left to right whichever way the road was drawn
          const first = run[0] as Pt;
          const last = run[run.length - 1] as Pt;
          named.set(f.name, { key: `r${i}`, name: f.name, labelD: pathOf(last[0] >= first[0] ? run : [...run].reverse()), length });
        }
        return;
      }
      const fill = f.class === "parking_lot" ? MAP_COLORS.parking : f.class === "residential" ? MAP_COLORS.residential : MAP_COLORS.building;
      const what = f.class === "parking_lot" ? "Car park" : f.class === "residential" ? "House" : "Building";
      areas.push({
        key: `a${i}`,
        points: pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
        fill,
        title: f.name ? `${f.name} (${what.toLowerCase()})` : what,
        cx: pts.reduce((s, p) => s + p[0], 0) / pts.length,
        cy: pts.reduce((s, p) => s + p[1], 0) / pts.length,
        cls: f.class,
      });
    });
    // car parks under buildings, so a building standing in a car park stays visible
    areas.sort((a, b) => Number(b.cls === "parking_lot") - Number(a.cls === "parking_lot"));

    const cx = view.x + view.w / 2;
    const cy = view.y + view.h / 2;
    const places = (surroundings.places ?? [])
      .map((p, i) => ({ key: `p${i}`, name: p.name, kind: p.kind.replace(/_/g, " "), xy: px(p.latlng) }))
      .filter((p) => p.xy[0] >= view.x && p.xy[0] <= view.x + view.w && p.xy[1] >= view.y && p.xy[1] <= view.y + view.h)
      .sort((a, b) => Math.hypot(a.xy[0] - cx, a.xy[1] - cy) - Math.hypot(b.xy[0] - cx, b.xy[1] - cy));
    return { areas, roads, roadLabels: [...named.values()], places };
  }, [surroundings, px, view]);

  const roadW = ROAD_WIDTH_M / mPerPx; // metres -> mosaic pixels
  const small = fontPx * 0.82; // map names a size under the overlay's own labels

  // Place names, nearest the building first: to the right of the dot, or to its left when the
  // right side runs out of the view or under a marker or name drawn on top. A place with no
  // free side keeps its dot and tooltip.
  const placeLabels = useMemo(() => {
    const out = new Map<string, { x: number; anchor: "start" | "end" }>();
    const taken: Box[] = [...(avoid ?? [])];
    // estimated label box, as in AerialSite: 1.25 em tall, 0.56 em per character
    const tall = small * 1.25;
    const gap = 6 * u;
    for (const p of shapes.places) {
      if (out.size >= MAX_PLACE_LABELS) break;
      const wide = p.name.length * small * 0.56;
      const y0 = p.xy[1] - tall / 2;
      const right: Box = [p.xy[0] + gap, y0, p.xy[0] + gap + wide, y0 + tall];
      const left: Box = [p.xy[0] - gap - wide, y0, p.xy[0] - gap, y0 + tall];
      const fit = [right, left].find((b) => b[0] >= view.x && b[2] <= view.x + view.w && !taken.some((t) => overlaps(b, t)));
      if (!fit) continue;
      taken.push(fit);
      out.set(p.key, fit === right ? { x: right[0], anchor: "start" } : { x: left[2], anchor: "end" });
    }
    return out;
  }, [shapes.places, avoid, small, u, view]);

  return (
    <g>
      <rect x={view.x} y={view.y} width={view.w} height={view.h} fill={MAP_COLORS.ground} />
      {shapes.areas.map((a) => (
        <polygon key={a.key} points={a.points} fill={a.fill} stroke="#0b1016" strokeWidth={0.8 * u} strokeLinejoin="round">
          <title>{a.title}</title>
        </polygon>
      ))}
      {shapes.areas
        .filter((a) => a.cls === "parking_lot")
        .map((a) => (
          <text key={`${a.key}-p`} x={a.cx} y={a.cy + small * 0.36} textAnchor="middle" fontSize={small} fontWeight="700" fill="#8fa6bf" pointerEvents="none">
            P
          </text>
        ))}
      <g fill="none" strokeLinecap="round" strokeLinejoin="round">
        {shapes.roads.map((r) => (
          <path key={`${r.key}-c`} d={r.d} stroke={MAP_COLORS.roadCasing} strokeWidth={roadW + 2.5 * u} />
        ))}
        {shapes.roads.map((r) => (
          <path key={r.key} d={r.d} stroke={MAP_COLORS.road} strokeWidth={roadW}>
            <title>{r.name || "Street"}</title>
          </path>
        ))}
      </g>
      <defs>
        {shapes.roadLabels.map((r) => (
          <path key={r.key} id={`road-${r.key}`} d={r.labelD} />
        ))}
      </defs>
      {shapes.roadLabels
        // only where the stretch in view is longer than the name (about 0.6 em per character)
        .filter((r) => r.length > r.name.length * small * 0.6)
        .map((r) => (
          <text key={`${r.key}-t`} fontSize={small} fontWeight="600" fill="#e6edf3" dy={small * 0.34} pointerEvents="none">
            <textPath href={`#road-${r.key}`} startOffset="50%" textAnchor="middle">
              {r.name}
            </textPath>
          </text>
        ))}
      {shapes.places.map((p) => {
        const label = placeLabels.get(p.key);
        return (
          <g key={p.key}>
            <title>{`${p.name} (${p.kind})`}</title>
            <circle cx={p.xy[0]} cy={p.xy[1]} r={3.2 * u} fill={MAP_COLORS.place} stroke="#0b1016" strokeWidth={1.2 * u} />
            {label ? (
              <text
                x={label.x}
                y={p.xy[1] + small * 0.36}
                textAnchor={label.anchor}
                fontSize={small}
                fontWeight="600"
                fill={MAP_COLORS.place}
                stroke="#0b1016"
                strokeWidth={3 * u}
                paintOrder="stroke"
              >
                {p.name}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

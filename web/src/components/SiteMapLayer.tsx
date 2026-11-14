import { useMemo } from "react";
import type { LatLng, SiteSurroundings } from "../types";

interface Props {
  surroundings: SiteSurroundings;
  /** lat/lng -> the parent SVG's coordinates (mosaic pixels) */
  px: (p: LatLng) => [number, number];
  view: { x: number; y: number; w: number; h: number };
  /** one overlay unit of the parent view */
  u: number;
  mPerPx: number;
  fontPx: number;
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

const ROAD_WIDTH_M = 9;
const MAX_PLACE_LABELS = 8;

/**
 * The street map under the site overlay, drawn from the same OpenStreetMap features the
 * backend uses to decide what each window looks at: road centrelines, car parks, houses,
 * other buildings and named places. It shares the aerial image's coordinate system, so the
 * building outline, entrances and staging spots sit on it exactly as they do on the imagery.
 */
export function SiteMapLayer({ surroundings, px, view, u, mPerPx, fontPx }: Props) {
  const shapes = useMemo(() => {
    const inView = (pts: [number, number][]) =>
      pts.some((p) => p[0] >= view.x - view.w * 0.2 && p[0] <= view.x + view.w * 1.2 && p[1] >= view.y - view.h * 0.2 && p[1] <= view.y + view.h * 1.2);
    const areas: { key: string; points: string; fill: string; title: string; cx: number; cy: number; cls: string }[] = [];
    const roads: { key: string; d: string; labelD: string; name: string; length: number }[] = [];
    surroundings.features.forEach((f, i) => {
      if (f.own) return; // the modelled building is drawn by the outline on top
      const pts = f.latlng.map(px);
      if (pts.length < 2 || !inView(pts)) return;
      if (f.shape === "line") {
        const d = "M " + pts.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" L ");
        // names read left to right whichever way the road was drawn
        const first = pts[0] as [number, number];
        const last = pts[pts.length - 1] as [number, number];
        const ltr = last[0] >= first[0] ? pts : [...pts].reverse();
        const labelD = "M " + ltr.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" L ");
        let length = 0;
        pts.forEach((p, k) => {
          const q = pts[k - 1];
          if (q) length += Math.hypot(p[0] - q[0], p[1] - q[1]);
        });
        roads.push({ key: `r${i}`, d, labelD, name: f.name, length });
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

    // one label per street name: on its longest stretch in view
    const named = new Map<string, (typeof roads)[number]>();
    for (const r of roads) {
      if (!r.name) continue;
      const cur = named.get(r.name);
      if (!cur || r.length > cur.length) named.set(r.name, r);
    }

    const cx = view.x + view.w / 2;
    const cy = view.y + view.h / 2;
    const places = (surroundings.places ?? [])
      .map((p, i) => ({ key: `p${i}`, name: p.name, kind: p.kind.replace(/_/g, " "), xy: px(p.latlng) }))
      .filter((p) => p.xy[0] >= view.x && p.xy[0] <= view.x + view.w && p.xy[1] >= view.y && p.xy[1] <= view.y + view.h)
      .sort((a, b) => Math.hypot(a.xy[0] - cx, a.xy[1] - cy) - Math.hypot(b.xy[0] - cx, b.xy[1] - cy));
    return { areas, roads, roadLabels: [...named.values()], places };
  }, [surroundings, px, view]);

  const roadW = ROAD_WIDTH_M / mPerPx;
  const small = fontPx * 0.82;

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
        .filter((r) => r.length > r.name.length * small * 0.6)
        .map((r) => (
          <text key={`${r.key}-t`} fontSize={small} fontWeight="600" fill="#e6edf3" dy={small * 0.34} pointerEvents="none">
            <textPath href={`#road-${r.key}`} startOffset="50%" textAnchor="middle">
              {r.name}
            </textPath>
          </text>
        ))}
      {shapes.places.map((p, i) => (
        <g key={p.key}>
          <title>{`${p.name} (${p.kind})`}</title>
          <circle cx={p.xy[0]} cy={p.xy[1]} r={3.2 * u} fill={MAP_COLORS.place} stroke="#0b1016" strokeWidth={1.2 * u} />
          {i < MAX_PLACE_LABELS ? (
            <text
              x={p.xy[0] + 6 * u}
              y={p.xy[1] + small * 0.36}
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
      ))}
    </g>
  );
}

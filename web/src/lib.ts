import type { Building, ClueType, Posterior, Room, XY } from "./types";

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** 0.0734 -> "7.3%", 0.42 -> "42%", tiny -> "<0.1%" */
export function pct(p: number, digits?: number): string {
  if (!Number.isFinite(p)) return "-";
  const v = p * 100;
  if (v > 0 && v < 0.1) return "<0.1%";
  const d = digits ?? (v >= 10 ? 0 : 1);
  return `${v.toFixed(d)}%`;
}

/** seconds -> "m:ss" */
export function mmss(seconds: number): string {
  if (!Number.isFinite(seconds)) return "-";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Probability ramp: one perceptual sequential scale, dark (low) -> light (high).
// Lightness rises monotonically so it reads correctly on the dark surface.
// ---------------------------------------------------------------------------

const RAMP: [number, number, number][] = [
  [20, 29, 41], // #141d29  ~ panel surface: zero-probability rooms recede
  [27, 63, 99], // #1b3f63
  [34, 120, 154], // #22789a
  [63, 184, 174], // #3fb8ae
  [168, 230, 193], // #a8e6c1
  [241, 251, 227], // #f1fbe3
];

export function rampColor(t: number): string {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const pos = x * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = RAMP[i]!;
  const b = RAMP[i + 1]!;
  const c = a.map((v, k) => Math.round(v + (b[k]! - v) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Ink colour that stays legible on top of rampColor(t). */
export function rampInk(t: number): string {
  return t > 0.55 ? "#0b1016" : "#e6edf3";
}

export const RAMP_CSS_GRADIENT = `linear-gradient(to right, ${[0, 0.2, 0.4, 0.6, 0.8, 1]
  .map((t) => rampColor(t))
  .join(", ")})`;

/**
 * Heatmap scale position for probability p given the current max.
 * Square-root easing keeps secondary peaks visible next to a dominant room.
 */
export function heatT(p: number, maxP: number): number {
  if (maxP <= 0) return 0;
  return Math.sqrt(Math.min(1, Math.max(0, p / maxP)));
}

// ---------------------------------------------------------------------------
// Clue type styling (muted categorical accents; never red - red is for alerts)
// ---------------------------------------------------------------------------

export const CLUE_STYLE: Record<ClueType, { label: string; color: string }> = {
  room_type: { label: "Room", color: "#7fb4ff" },
  floor: { label: "Floor", color: "#b79cff" },
  view: { label: "View", color: "#5fd0b5" },
  landmark: { label: "Landmark", color: "#e3c56b" },
  sound: { label: "Sound", color: "#8fd2e8" },
  smell: { label: "Smell", color: "#c9a27e" },
  route: { label: "Route", color: "#e0a56a" },
  negative: { label: "Negative", color: "#a9b4c0" },
};

// ---------------------------------------------------------------------------
// Building helpers
// ---------------------------------------------------------------------------

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function boundsOf(points: XY[]): Bounds {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}

export function buildingBounds(building: Building): Bounds {
  return boundsOf(building.rooms.flatMap((r) => r.polygon));
}

export function floorLabelMap(building: Building): Map<number, string> {
  const m = new Map<number, string>();
  for (const r of building.rooms) if (!m.has(r.floor)) m.set(r.floor, r.floor_label);
  for (const f of building.floors) if (!m.has(f)) m.set(f, String(f));
  return m;
}

/** Floor index with the highest marginal probability (null if no mass anywhere). */
export function mostProbableFloor(building: Building, posterior: Posterior): number | null {
  const labels = floorLabelMap(building);
  let best: number | null = null;
  let bestP = 0;
  for (const f of building.floors) {
    const p = posterior.p_floor[labels.get(f) ?? String(f)] ?? 0;
    if (p > bestP) {
      bestP = p;
      best = f;
    }
  }
  return best;
}

export function roomIndex(building: Building): Map<string, Room> {
  return new Map(building.rooms.map((r) => [r.id, r]));
}

// ---------------------------------------------------------------------------
// Web-Mercator tile maths (mirrors ingest/vexcel.py) for the aerial overlay
// ---------------------------------------------------------------------------

/** Fractional tile column / row of a WGS84 point at `zoom`. */
export function latLngToTile(lat: number, lng: number, zoom: number): [number, number] {
  const n = 2 ** zoom;
  const x = ((lng + 180) / 360) * n;
  const latR = (lat * Math.PI) / 180;
  const y = ((1 - Math.asinh(Math.tan(latR)) / Math.PI) / 2) * n;
  return [x, y];
}

/** Expected phone height on a floor: finished floor + 1.2 m (held phone). */
export const PHONE_HEIGHT_M = 1.2;
/** +/-3 m is the 80% z-axis bound. */
export const Z_BAND_M = 3.0;

export function floorZ(building: Building, floor: number): number {
  return (building.floor_heights_m[String(floor)] ?? 0) + PHONE_HEIGHT_M;
}

/** "02:14:07 PM" for an ISO timestamp, or null when there is none */
export function clockTime(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

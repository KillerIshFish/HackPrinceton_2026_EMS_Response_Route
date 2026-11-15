import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import type { ReactNode } from "react";
import { Ambulance, Boxes, DoorClosed, DoorOpen, Flame, Info, LogIn, RotateCw } from "lucide-react";
import { api } from "../api";
import type { Building, Edge, HazardClass, Model3DPayload, Policy, Posterior, SearchPlan, TravelProfile } from "../types";
import { heatT, pct, rampColor, roomIndex } from "../lib";
import { Panel } from "./Panel";

interface Props {
  building: Building;
  posterior: Posterior;
  plan: SearchPlan;
  /** floor shown in the 2D plan; the 3D view cuts the building away down to it */
  floor: number;
  onSelectFloor: (floor: number) => void;
  onSelectRoom: (roomId: string) => void;
  /**
   * What the console may show. route_enabled false: no stop numbers, next-stop marker, route,
   * entrance or staging (the probability colouring stays, as information). show_hazard_class:
   * hazard chips over the rooms the plan's stops classify.
   */
  policy: Policy;
  /** rooms the officer closed: the drawn route goes around them, as the plan's travel times do */
  closedRoomIds?: string[];
}

/** Hazard class in words, with an icon: never colour alone, and no style that reads as "fine to leave". */
export const HAZARD_STYLE: Record<HazardClass, { label: string; icon: ReactNode; className: string }> = {
  fire_room: {
    label: "Fire room",
    icon: <Flame size={11} aria-hidden />,
    className: "border-alert bg-[#2a1214] text-[#ffb3b3]",
  },
  open_path: {
    label: "Open to the fire",
    icon: <DoorOpen size={11} aria-hidden />,
    className: "border-warn bg-[#262010] text-[#f3dd8a]",
  },
  closed_door: {
    label: "Behind a closed door",
    icon: <DoorClosed size={11} aria-hidden />,
    className: "border-line-strong bg-raised text-ink",
  },
};

export const HAZARD_LEGEND = "Shown for information. A closed door stops protecting once it is opened.";

/** A hazard chip: icon and words. */
export function HazardChip({ hazard, compact = false, iconOnly = false }: { hazard: HazardClass; compact?: boolean; iconOnly?: boolean }) {
  const st = HAZARD_STYLE[hazard];
  return (
    <span
      title={iconOnly ? st.label : undefined}
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-sm border font-semibold leading-none ${st.className} ${
        compact ? "px-1 py-0.5 text-[10px]" : "px-1.5 py-1 text-[11px]"
      }`}
    >
      {st.icon}
      {iconOnly ? <span className="sr-only">{st.label}</span> : st.label}
    </span>
  );
}

/** circulation spaces open to the fire: counted in the legend, not marked on the model (too many to read) */
const CIRCULATION = new Set(["corridor", "stairwell", "elevator"]);

type ViewMode = "cutaway" | "floor" | "all";
const VIEW_MODES: { key: ViewMode; label: string; title: string }[] = [
  { key: "cutaway", label: "Cutaway", title: "The selected floor and everything below it" },
  { key: "floor", label: "This floor", title: "Only the selected floor" },
  { key: "all", label: "Whole building", title: "Every floor, upper walls see-through" },
];

const FIDELITY_LABEL: Record<string, string> = {
  surveyed: "measured by a scan",
  traced: "traced from the published floor plan",
  stamped: "from a scan of the unit type",
  cluster: "scan positions only",
  template: "generated layout",
  roster: "floor known, position unknown",
};

/** stops per team that the drawn route runs through; each one gets a floating label with its room's name */
const ROUTE_STOPS = 8;
/**
 * Where a stop's label may stand, nearest first: the label's centre relative to the room's point,
 * in pixels, for a label `lw` by `lh`. Above the point at four heights (centred, or shifted to
 * either side), beside it at two distances, and under it as a last resort (never under the cone
 * that marks the next stop). The first place that covers no other label or stop wins.
 */
function labelSlots(lw: number, lh: number, coned: boolean): [number, number][] {
  const out: [number, number][] = [];
  const above = (lift: number) => {
    for (const shift of [0, 0.36, -0.36]) out.push([shift * lw, -(lift + lh / 2)]);
  };
  const beside = (rise: number, gap: number) => {
    for (const side of [1, -1]) out.push([side * (lw / 2 + gap), -rise]);
  };
  above(26);
  beside(5, 16);
  above(50);
  beside(30, 16);
  beside(5, 48);
  above(74);
  beside(-14, 16);
  if (!coned) for (const shift of [0, 0.36, -0.36]) out.push([shift * lw, 20 + lh / 2]);
  above(98);
  beside(30, 48);
  return out;
}

/** A room name short enough for a label. The start and the end are kept: the end often tells rooms apart ("... (Level 1 north)"). */
function shortName(name: string, max = 32): string {
  if (name.length <= max) return name;
  // a closing "(...)" is kept whole when it is short: it is usually the part that differs
  const bracket = /\([^()]*\)?$/.exec(name)?.[0] ?? "";
  let tail = bracket.length >= 3 && bracket.length <= max - 12 ? bracket : name.slice(name.length - Math.floor((max - 3) * 0.45));
  if (tail !== bracket) {
    const firstSpace = tail.indexOf(" ");
    if (firstSpace >= 0 && firstSpace < 8 && firstSpace < tail.length - 4) tail = tail.slice(firstSpace + 1);
  }
  let head = name.slice(0, Math.max(6, max - 3 - tail.length));
  // cut between words when a word boundary is close
  const lastSpace = head.lastIndexOf(" ");
  if (lastSpace >= head.length - 8 && lastSpace > 4) head = head.slice(0, lastSpace);
  return `${head.trimEnd()} … ${tail.trimStart()}`;
}
/**
 * Where a stop's number may stand when an earlier stop's number or name is already on its point
 * (rooms one above the other, or a view from far away): on the point, else the nearest place a
 * whole number of badges away. In badge widths and heights, nearest first. The first BADGE_NEAR
 * (within about four badges) are tried for a place clear of everything; all of them, up to six
 * badges away, for a place that is at least clear of every name.
 */
const BADGE_NEAR = 81;
/** side of the squares the placed numbers are filed under, so a crowded floor's numbers are not all compared with each other */
const BADGE_CELL = 40;
const BADGE_NUDGES: [number, number][] = (() => {
  const out: [number, number][] = [];
  for (let nx = -6; nx <= 6; nx++) for (let ny = -6; ny <= 6; ny++) out.push([nx, ny]);
  // a badge is wider than it is tall, and up is preferred to down between equals
  const far = ([nx, ny]: [number, number]) => Math.hypot(nx * 1.3, ny) + (ny > 0 ? 0.01 : 0);
  return out.sort((a, c) => far(a) - far(c));
})();
/** thickness of the floor slab drawn under each level's rooms */
const SLAB_M = 0.3;
/** air added between storeys, as a share of the building's typical storey height: 1 = one floor's height
 * of clear space above each floor, so every level stands apart and can be seen into from the side */
const FLOOR_GAP_STOREYS = 1.0;
/** the gap used when the model has no storey heights */
const FLOOR_GAP_FALLBACK_M = 3.0;

/** the gap between storeys for this model: its median storey height times FLOOR_GAP_STOREYS */
function floorGap(payload: Model3DPayload): number {
  const hs = payload.levels.map((l) => l.h).filter((h) => h > 0).sort((a, b) => a - b);
  const typical = hs.length ? hs[Math.floor(hs.length / 2)]! : FLOOR_GAP_FALLBACK_M;
  return typical * FLOOR_GAP_STOREYS;
}
/** how far past the building's outline (as a share of its longest side) the view can be panned */
const PAN_MARGIN = 0.1;
/** a double-click that comes this soon after a drag ended is the end of that drag, not a pick */
const DRAG_CLICK_MS = 600;
/** pointer travel, in pixels, past which a press is a drag rather than a click */
const DRAG_PX = 5;
/** height of the route line above the floor it runs on */
const ROUTE_LIFT_M = 1.0;
/** route and stop colour per team (the stop numbers also carry the team letter) */
const TEAM_COLOR = [0xf2a93b, 0x79a8d8];

/** longest dimension of the model, for sizing things that must stay readable on any building */
function modelSpan(payload: Model3DPayload): number {
  const b = payload.bounds;
  return Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1, 8);
}
/**
 * The model with `gap` metres of air (``floorGap``) added under each storey above the lowest: every level's floor
 * height and its walls' and windows' heights move up together, so everything drawn per floor (slabs,
 * room fills, walls, glass, labels, route, marker, framing) follows from the one change, and a route
 * leg between floors runs from one level's height to the other's, across the gap.
 */
function separateFloors(payload: Model3DPayload, gap: number): Model3DPayload {
  const order = [...payload.levels].sort((a, b) => a.z0 - b.z0);
  const lift = new Map(order.map((l, k) => [l.i, k * gap]));
  const up = (level: number) => lift.get(level) ?? 0;
  type Span = [number, number, number, number];
  const shift = (spans: Span[], d: number) => spans.map(([t0, t1, z0, z1]): Span => [t0, t1, z0 + d, z1 + d]);
  return {
    ...payload,
    levels: payload.levels.map((l) => ({ ...l, z0: l.z0 + up(l.i) })),
    walls: payload.walls.map((w) => {
      const d = up(w.l);
      return d ? { ...w, p: shift(w.p, d), g: shift(w.g, d) } : w;
    }),
    bounds: { ...payload.bounds, z1: payload.bounds.z1 + Math.max(0, order.length - 1) * gap },
  };
}

/** height of the cone that points down at the next stop */
const markerHeight = (span: number) => Math.max(1.4, span / 28);

/** One level's meshes plus what is needed to recolour and pick its floor plates. */
interface LevelMeshes {
  index: number;
  group: THREE.Group;
  plates: THREE.Mesh;
  /** per plate triangle: index into payload.spaces */
  triSpace: Int32Array;
  /** per space drawn on this level: [space index, first vertex, vertex count] */
  ranges: [number, number, number][];
  walls: WallPart[];
}

/** One merged wall mesh of a level: the partitions, the window glass, or the shell walls that face one way. */
interface WallPart {
  mesh: THREE.Mesh;
  material: THREE.Material;
  kind: "shell" | "inner" | "glass";
  /** opacity on the floor in focus */
  base: number;
  /** which way the outside of these shell walls faces, in scene axes; null when unknown or not shell */
  outward: THREE.Vector3 | null;
  /** the material's own colour (walls only), dimmed on the storeys that are not in focus */
  color: THREE.Color | null;
  /** set by the view mode: opacity before the view-dependent fade, depth writing, and whether the fade applies */
  opacity: number;
  write: boolean;
  fade: boolean;
}

/** Wall look by where the geometry comes from and whether it is part of the outer shell. */
const WALL_STYLE: Record<string, { color: number; base: number }> = {
  "measured:shell": { color: 0xc9d3df, base: 1 },
  "measured:inner": { color: 0xd7dee8, base: 0.9 },
  "generated:shell": { color: 0x9ba8b9, base: 0.85 },
  "generated:inner": { color: 0x8793a3, base: 0.5 },
};
/** walls (not glass) are drawn at this share of their style's strength, so the rooms and the route behind them show */
const WALL_OPACITY = 0.67;
const SLAB_COLOR = 0x303c4b;
const GLASS_OPACITY = 0.35;
/** how much of a shell wall's opacity is left when its outside faces the viewer */
const NEAR_SHELL = 0.16;
/** wall brightness on the storeys that are not in focus */
const OFF_FOCUS_TINT = 0.55;
const NO_PROB = new THREE.Color(0x1b2531);

/** data (x east, y north, z up) -> three (x, y up, z south) */
function v3(x: number, y: number, z: number): [number, number, number] {
  return [x, z, -y];
}

function pushBox(out: number[], a: [number, number], b: [number, number], t0: number, t1: number, z0: number, z1: number, th: number) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = (-dy / len) * (th / 2);
  const ny = (dx / len) * (th / 2);
  const p0: [number, number] = [a[0] + dx * t0, a[1] + dy * t0];
  const p1: [number, number] = [a[0] + dx * t1, a[1] + dy * t1];
  // 8 corners: bottom then top, around the footprint
  const foot: [number, number][] = [
    [p0[0] + nx, p0[1] + ny],
    [p1[0] + nx, p1[1] + ny],
    [p1[0] - nx, p1[1] - ny],
    [p0[0] - nx, p0[1] - ny],
  ];
  const c = [...foot.map((p) => v3(p[0], p[1], z0)), ...foot.map((p) => v3(p[0], p[1], z1))];
  const quads: [number, number, number, number][] = [
    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7], [3, 2, 1, 0],
  ];
  for (const [i, j, k, l] of quads) {
    for (const idx of [i, j, k, i, k, l]) out.push(...c[idx]!);
  }
}

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!;
    const [xj, yj] = poly[j]!;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Convex hull (monotone chain) of 2D points, counter-clockwise. */
function hull2d(points: [number, number][]): [number, number][] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: [number, number], a: [number, number], b: [number, number]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: [number, number][]) => {
    const h: [number, number][] = [];
    for (const p of list) {
      while (h.length >= 2 && cross(h[h.length - 2]!, h[h.length - 1]!, p) <= 0) h.pop();
      h.push(p);
    }
    h.pop();
    return h;
  };
  return [...half(pts), ...half(pts.reverse())];
}

/**
 * Which walls belong to the outer shell. A whole-building scan's own flag is trusted. For generated
 * and outlined buildings the flag only means "listed by one room", which is also true of a corridor
 * wall that several rooms share piecewise and of a stamped unit's party walls, so those are checked
 * against the floor: a shell wall has a room on one side only.
 */
function shellWalls(payload: Model3DPayload): { shell: Uint8Array; outward: Float32Array } {
  const shell = new Uint8Array(payload.walls.length);
  const outward = new Float32Array(payload.walls.length * 2); // unit vector out of the building; (0, 0) when unknown
  const measured = payload.archetype === "whole_scan";
  const byLevel = new Map<number, { p: [number, number][]; x0: number; y0: number; x1: number; y1: number }[]>();
  for (const s of payload.spaces) {
    if (s.par || s.p.length < 3) continue;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of s.p) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
    let list = byLevel.get(s.l);
    if (!list) byLevel.set(s.l, (list = []));
    list.push({ p: s.p, x0, y0, x1, y1 });
  }
  payload.walls.forEach((w, wi) => {
    if (!w.x) return;
    const floor = byLevel.get(w.l) ?? [];
    const inside = (x: number, y: number) => floor.some((s) => x >= s.x0 && x <= s.x1 && y >= s.y0 && y <= s.y1 && pointInPolygon(x, y, s.p));
    const dx = w.b[0] - w.a[0];
    const dy = w.b[1] - w.a[1];
    const len = Math.hypot(dx, dy) || 1;
    const off = (w.th || 0.12) / 2 + 0.3;
    const nx = (-dy / len) * off;
    const ny = (dx / len) * off;
    let open = 0;
    let plus = 0;
    let minus = 0;
    for (const t of [0.2, 0.5, 0.8]) {
      const x = w.a[0] + dx * t;
      const y = w.a[1] + dy * t;
      const a = inside(x + nx, y + ny);
      const b = inside(x - nx, y - ny);
      if (!(a && b)) open += 1;
      plus += Number(a);
      minus += Number(b);
    }
    shell[wi] = measured || open >= 2 ? 1 : 0;
    // the outside is the side with no room on it
    const sign = plus > minus ? -1 : minus > plus ? 1 : 0;
    outward[wi * 2] = (sign * nx) / off;
    outward[wi * 2 + 1] = (sign * ny) / off;
  });
  return { shell, outward };
}

/** Today's three-quarter view: about 32 degrees above the horizon, 30 degrees to one side of the front. */
const VIEW_ELEVATION = Math.atan2(0.62, Math.hypot(0.5, 0.86));
const VIEW_SIDE = Math.PI / 6;
/** used instead when the front is the short end of a long building, so its long side still shows */
const VIEW_SIDE_END_ON = (58 * Math.PI) / 180;

/**
 * Direction from the model toward the default camera: outside the front of the building, a little
 * to one side and above, so the street side faces the viewer. The front is the side the
 * street entrance opens onto (the model's first entrance when none is marked as the street one).
 * A generated layout puts the street on -y, which is also the fallback for a model with no door
 * to the outside. This is fixed building data: it does not depend on the call or the plan.
 */
function viewDirection(payload: Model3DPayload, building: Building): THREE.Vector3 {
  const doors = (payload.entrances ?? []).filter((e) => e.o && Math.hypot(e.o[0], e.o[1]) > 0.5);
  const street = building.entrances.find((e) => e.facing === "street");
  const door = (street && doors.find((e) => e.id === street.id)) ?? doors[0];
  let fx = 0;
  let fy = -1;
  if (door?.o) {
    const n = Math.hypot(door.o[0], door.o[1]);
    fx = door.o[0] / n;
    fy = door.o[1] / n;
    // a door at the corner of its room gives a slanted vector: square it up when it is close to an axis
    if (Math.abs(fx) > 0.82) [fx, fy] = [Math.sign(fx), 0];
    else if (Math.abs(fy) > 0.82) [fx, fy] = [0, Math.sign(fy)];
  }
  const b = payload.bounds;
  const dx = b.x1 - b.x0;
  const dy = b.y1 - b.y0;
  const depth = Math.abs(fx) * dx + Math.abs(fy) * dy; // extent along the front direction
  const width = Math.abs(fy) * dx + Math.abs(fx) * dy; // extent across it
  const endOn = Math.min(1, Math.max(0, (depth / Math.max(width, 1e-6) - 1) / 1.5));
  const angle = VIEW_SIDE + (VIEW_SIDE_END_ON - VIEW_SIDE) * endOn;
  // stand on the side of the front where the door is; to the left (as before) when it is near the middle
  let side = -1;
  if (door) {
    const along = (door.xy[0] - (b.x0 + b.x1) / 2) * -fy + (door.xy[1] - (b.y0 + b.y1) / 2) * fx;
    if (Math.abs(along) > 0.1 * width) side = Math.sign(along);
  }
  const theta = angle * side;
  const sx = fx; // the front direction in scene axes (x, z) is (fx, -fy)
  const sz = -fy;
  const hx = sx * Math.cos(theta) + sz * Math.sin(theta);
  const hz = -sx * Math.sin(theta) + sz * Math.cos(theta);
  return new THREE.Vector3(hx * Math.cos(VIEW_ELEVATION), Math.sin(VIEW_ELEVATION), hz * Math.cos(VIEW_ELEVATION)).normalize();
}

/**
 * Put the camera where all the points are in frame with a small margin and their outline on
 * screen is centred, whatever the canvas aspect ratio and however long, flat or tall the model
 * is. The distance is found by bisection on the projected points; the aim point is then moved in
 * the view plane until the projected outline sits in the middle (a perspective view of a box is
 * not centred on the box's centre), so nothing is cropped and nothing sits off to one side.
 */
function fitCamera(camera: THREE.PerspectiveCamera, controls: OrbitControls, points: THREE.Vector3[], dir: THREE.Vector3) {
  if (points.length === 0) return;
  const box = new THREE.Box3().setFromPoints(points);
  const target = box.getCenter(new THREE.Vector3());
  const reach = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
  const margin = 0.9;
  const p = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const place = (d: number) => {
    camera.position.copy(target).addScaledVector(dir, d);
    camera.near = Math.max(d / 500, 0.05);
    camera.far = d * 40 + reach * 4;
    camera.lookAt(target);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  };
  const extent = () => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, front = true;
    for (const c of points) {
      p.copy(c).project(camera);
      if (!(p.z < 1)) front = false;
      x0 = Math.min(x0, p.x);
      x1 = Math.max(x1, p.x);
      y0 = Math.min(y0, p.y);
      y1 = Math.max(y1, p.y);
    }
    return { x0, x1, y0, y1, front };
  };
  let distance = reach;
  for (let pass = 0; pass < 6; pass++) {
    let lo = reach * 0.05;
    let hi = reach * 40;
    for (let i = 0; i < 28; i++) {
      const mid = (lo + hi) / 2;
      place(mid);
      const e = extent();
      if (e.front && e.x0 >= -margin && e.x1 <= margin && e.y0 >= -margin && e.y1 <= margin) hi = mid;
      else lo = mid;
    }
    distance = hi;
    place(distance);
    const e = extent();
    const cx = (e.x0 + e.x1) / 2;
    const cy = (e.y0 + e.y1) / 2;
    if (pass === 5 || (Math.abs(cx) < 0.003 && Math.abs(cy) < 0.003)) break;
    const halfH = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * distance;
    right.setFromMatrixColumn(camera.matrixWorld, 0);
    up.setFromMatrixColumn(camera.matrixWorld, 1);
    target.addScaledVector(right, cx * halfH * camera.aspect).addScaledVector(up, cy * halfH);
  }
  controls.target.copy(target);
  controls.update();
}

function buildLevels(payload: Model3DPayload): LevelMeshes[] {
  const parents = new Set(payload.spaces.map((s) => s.par).filter((p): p is string => !!p));
  const { shell, outward } = shellWalls(payload);
  const out: LevelMeshes[] = [];
  for (const lv of payload.levels) {
    const group = new THREE.Group();
    // ---- floor plates: leaf spaces only (a unit with child rooms is drawn as its rooms),
    //      each standing on a slab so a storey reads as a solid floor from the side
    const pos: number[] = [];
    const tri: number[] = [];
    const slab: number[] = [];
    const ranges: [number, number, number][] = [];
    payload.spaces.forEach((s, si) => {
      if (s.l !== lv.i || parents.has(s.id) || s.p.length < 3) return;
      const contour = s.p.map((p) => new THREE.Vector2(p[0], p[1]));
      const faces = THREE.ShapeUtils.triangulateShape(contour, []);
      const start = pos.length / 3;
      // walkways and unknown plates sit a little lower so real rooms read on top
      const lift = s.k === "walkway" ? 0.01 : 0.04;
      for (const f of faces) {
        for (const vi of f) {
          const p = s.p[vi]!;
          pos.push(...v3(p[0], p[1], lv.z0 + lift));
          slab.push(...v3(p[0], p[1], lv.z0 - SLAB_M));
        }
        tri.push(si);
      }
      ranges.push([si, start, pos.length / 3 - start]);
      for (let i = 0; i < s.p.length; i++) {
        const a = s.p[i]!;
        const b = s.p[(i + 1) % s.p.length]!;
        const lo0 = v3(a[0], a[1], lv.z0 - SLAB_M);
        const lo1 = v3(b[0], b[1], lv.z0 - SLAB_M);
        const hi0 = v3(a[0], a[1], lv.z0);
        const hi1 = v3(b[0], b[1], lv.z0);
        slab.push(...lo0, ...lo1, ...hi1, ...lo0, ...hi1, ...hi0);
      }
    });
    const plateGeo = new THREE.BufferGeometry();
    plateGeo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    plateGeo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
    plateGeo.computeVertexNormals();
    const plates = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    group.add(plates);
    if (slab.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(slab, 3));
      geo.computeVertexNormals();
      group.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: SLAB_COLOR, side: THREE.DoubleSide })));
    }

    // ---- walls (pre-cut panels) and window glass. Partitions are grouped by source; shell walls
    //      and their glass also by the way their outside faces (eight directions), so the ones
    //      between the viewer and the rooms can be faded as the camera moves
    const buckets = new Map<string, { kind: WallPart["kind"]; style: string; dir: number; pos: number[] }>();
    const bucket = (kind: WallPart["kind"], style: string, dir: number) => {
      const key = `${kind}:${style}:${dir}`;
      let b = buckets.get(key);
      if (!b) buckets.set(key, (b = { kind, style, dir, pos: [] }));
      return b.pos;
    };
    payload.walls.forEach((w, wi) => {
      if (w.l !== lv.i) return;
      const ox = outward[wi * 2]!;
      const oy = outward[wi * 2 + 1]!;
      const isShell = shell[wi] === 1;
      // -1: not a shell wall, or its outside is not known
      const dir = isShell && (ox || oy) ? (Math.round(Math.atan2(oy, ox) / (Math.PI / 4)) + 8) % 8 : -1;
      const source = w.f === "template" || w.f === "roster" ? "generated" : "measured";
      const th = Math.max(0.06, w.th || 0.12);
      const solid = bucket(isShell ? "shell" : "inner", `${source}:${isShell ? "shell" : "inner"}`, dir);
      for (const [t0, t1, z0, z1] of w.p) pushBox(solid, w.a, w.b, t0, t1, z0, z1, th);
      if (w.g.length) {
        const glass = bucket("glass", "glass", dir);
        for (const [t0, t1, z0, z1] of w.g) pushBox(glass, w.a, w.b, t0, t1, z0, z1, 0.04);
      }
    });
    const walls: WallPart[] = [];
    for (const b of buckets.values()) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(b.pos, 3));
      geo.computeVertexNormals();
      const style = WALL_STYLE[b.style];
      const material = style
        ? new THREE.MeshLambertMaterial({ color: style.color, transparent: true, opacity: style.base * WALL_OPACITY })
        : new THREE.MeshBasicMaterial({ color: 0x7fc4ff, transparent: true, opacity: GLASS_OPACITY, depthWrite: false });
      const mesh = new THREE.Mesh(geo, material);
      const angle = (b.dir * Math.PI) / 4;
      group.add(mesh);
      const base = style ? style.base * WALL_OPACITY : GLASS_OPACITY;
      walls.push({
        mesh,
        material,
        kind: b.kind,
        base,
        outward: b.dir < 0 ? null : new THREE.Vector3(Math.cos(angle), 0, -Math.sin(angle)),
        color: style ? new THREE.Color(style.color) : null,
        opacity: base,
        write: b.kind !== "glass",
        fade: false,
      });
    }
    out.push({ index: lv.i, group, plates, triSpace: Int32Array.from(tri), ranges, walls });
  }
  return out;
}

// ---------------------------------------------------------------------------
// The route between the marked stops, on the same travel graph the plan's times use
// ---------------------------------------------------------------------------

/** a point of the route: model x / y on a level */
interface RoutePoint {
  x: number;
  y: number;
  level: number;
}

/** A stop of the plan as the model marks it: one per room, at the room's first stop. */
interface RouteStop {
  team: number;
  /** position in its team's list, from 0 */
  index: number;
  roomId: string;
  /** a second (or later) look at a room that was already searched */
  again: boolean;
  /** later stops at the same room, as they are numbered ("6", "B2") */
  also: string[];
}

interface RouteData {
  /** straight pieces of the way in and between the stops on the route */
  segments: { team: number; a: RoutePoint; b: RoutePoint }[];
  /** the rooms the route runs through (each team's next ROUTE_STOPS stops), in the order labels claim space: by number */
  stops: RouteStop[];
  entrance: { name: string; at: RoutePoint } | null;
  staging: { name: string; at: RoutePoint } | null;
}

type Hop = { to: string; t: number; edge: Edge };

/**
 * Room-to-room travel times as the search plan counts them (core/building_graph.py): walk each
 * edge at the profile's speed, add the stair rate per floor climbed, and nothing through a room
 * the officer closed. When the profile offers an elevator (medical calls, no fire or smoke),
 * every pair of stops of one shaft is a single ride (fixed wait + seconds per floor), so the
 * route takes the elevator where that is faster than the stairs.
 */
function travelGraph(building: Building, travel: TravelProfile, closed: Set<string>): Map<string, Hop[]> {
  const rooms = new Map(building.rooms.map((r) => [r.id, r]));
  const adj = new Map<string, Hop[]>();
  const link = (e: Edge, t: number) => {
    for (const [from, to] of [[e.u, e.v], [e.v, e.u]] as const) {
      let list = adj.get(from);
      if (!list) adj.set(from, (list = []));
      list.push({ to, t, edge: e });
    }
  };
  const elevatorsOn = travel.elevator_allowed && travel.elevator_fixed_s != null;
  /** elevator stops joined by elevator edges (union-find), grouped into shafts below */
  const shaftParent = new Map<string, string>();
  const shaftRoot = (id: string): string => {
    let r = id;
    while (shaftParent.get(r) !== r) r = shaftParent.get(r)!;
    shaftParent.set(id, r);
    return r;
  };
  /** the elevator edge each pair of adjacent stops came with (keeps its length for a one-floor ride) */
  const shaftEdges = new Map<string, Edge>();
  for (const e of building.edges) {
    if (closed.has(e.u) || closed.has(e.v)) continue;
    const u = rooms.get(e.u);
    const v = rooms.get(e.v);
    if (!u || !v) continue;
    const length =
      e.length_m ?? Math.hypot(u.centroid[0] - v.centroid[0], u.centroid[1] - v.centroid[1], u.centroid[2] - v.centroid[2]);
    if (e.kind === "elevator") {
      if (!elevatorsOn) continue;
      for (const id of [e.u, e.v]) if (!shaftParent.has(id)) shaftParent.set(id, id);
      shaftParent.set(shaftRoot(e.u), shaftRoot(e.v));
      shaftEdges.set([e.u, e.v].sort().join("|"), { ...e, length_m: length });
      continue;
    }
    let t = length / Math.max(travel.walk_mps, 0.1);
    if (e.kind === "stairs") {
      const lower = Math.min(u.floor, v.floor);
      const floors = Math.abs(Math.trunc(e.floors || 0)) || Math.abs(u.floor - v.floor);
      for (let f = lower + 1; f <= lower + floors; f++) {
        t += travel.high_floor_from != null && f > travel.high_floor_from ? travel.stair_s_per_floor_high : travel.stair_s_per_floor;
      }
    }
    link(e, t);
  }
  // One ride between every pair of stops in a shaft: the fixed wait is paid once per ride.
  const shafts = new Map<string, string[]>();
  for (const id of shaftParent.keys()) {
    const root = shaftRoot(id);
    shafts.set(root, [...(shafts.get(root) ?? []), id]);
  }
  for (const stops of shafts.values()) {
    stops.sort();
    for (let i = 0; i < stops.length; i++) {
      for (let j = i + 1; j < stops.length; j++) {
        const u = rooms.get(stops[i]!)!;
        const v = rooms.get(stops[j]!)!;
        const floors = Math.abs(u.floor - v.floor);
        if (floors === 0) continue;
        const t = travel.elevator_fixed_s! + (travel.elevator_s_per_floor ?? 0) * floors;
        const edge: Edge = shaftEdges.get(`${u.id}|${v.id}`) ?? { u: u.id, v: v.id, kind: "elevator", length_m: 0, floors };
        link(edge, t);
      }
    }
  }
  return adj;
}

/** Fastest way from one room to another (Dijkstra): the rooms in order, each with the edge that led to it. */
function fastestPath(adj: Map<string, Hop[]>, from: string, to: string): { id: string; via: Edge | null }[] | null {
  if (from === to) return [{ id: from, via: null }];
  const dist = new Map<string, number>([[from, 0]]);
  const prev = new Map<string, { id: string; via: Edge }>();
  const done = new Set<string>();
  const heap: [number, string][] = [[0, from]];
  const pop = () => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      for (let i = 0; ; ) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heap[l]![0] < heap[m]![0]) m = l;
        if (r < heap.length && heap[r]![0] < heap[m]![0]) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m]!, heap[i]!];
        i = m;
      }
    }
    return top;
  };
  const push = (item: [number, string]) => {
    heap.push(item);
    for (let i = heap.length - 1; i > 0; ) {
      const parent = (i - 1) >> 1;
      if (heap[parent]![0] <= heap[i]![0]) break;
      [heap[i], heap[parent]] = [heap[parent]!, heap[i]!];
      i = parent;
    }
  };
  while (heap.length) {
    const [d, id] = pop();
    if (done.has(id)) continue;
    done.add(id);
    if (id === to) break;
    for (const hop of adj.get(id) ?? []) {
      const nd = d + hop.t;
      if (nd < (dist.get(hop.to) ?? Infinity)) {
        dist.set(hop.to, nd);
        prev.set(hop.to, { id, via: hop.edge });
        push([nd, hop.to]);
      }
    }
  }
  if (!prev.has(to)) return null;
  const path: { id: string; via: Edge | null }[] = [];
  for (let id = to; ; ) {
    const step = prev.get(id);
    path.push({ id, via: step?.via ?? null });
    if (!step) break;
    id = step.id;
  }
  return path.reverse();
}

/** The way in (staging spot, entrance) and the legs between each team's first stops. */
function planRoute(
  building: Building,
  plan: SearchPlan,
  travel: TravelProfile,
  closed: Set<string>,
  anchors: Map<string, RoutePoint>,
  lowestLevel: number,
): RouteData {
  const out: RouteData = { segments: [], stops: [], entrance: null, staging: null };
  const adj = travelGraph(building, travel, closed);
  const ent = building.entrances.find((e) => e.id === plan.entrance_id);
  const line = (team: number, pts: RoutePoint[]) => {
    for (let i = 1; i < pts.length; i++) out.segments.push({ team, a: pts[i - 1]!, b: pts[i]! });
  };
  if (ent) {
    const inside = anchors.get(ent.room_id);
    const level = inside?.level ?? lowestLevel;
    const door: RoutePoint = { x: ent.xy[0], y: ent.xy[1], level };
    out.entrance = { name: ent.name, at: door };
    const way: RoutePoint[] = [door];
    const spot = building.staging_spots.find((s) => s.near_entrance_id === ent.id);
    // an assumed entrance has its staging spot on the same point: nothing to draw between them
    if (spot && Math.hypot(spot.xy[0] - ent.xy[0], spot.xy[1] - ent.xy[1]) > 1) {
      const at: RoutePoint = { x: spot.xy[0], y: spot.xy[1], level };
      out.staging = { name: spot.name, at };
      way.unshift(at);
    }
    if (inside) way.push(inside);
    line(0, way);
  }
  const multiTeam = plan.teams.length > 1;
  const every = plan.teams.flatMap((stops, team) => stops.map((stop, index) => ({ team, index, stop })));
  every.sort((a, b) => a.index - b.index || a.team - b.team);
  const byRoom = new Map<string, RouteStop>();
  for (const { team, index, stop } of every) {
    const first = byRoom.get(stop.room_id);
    if (first) {
      first.also.push(`${multiTeam ? String.fromCharCode(65 + team) : ""}${index + 1}`);
      continue;
    }
    // a room the route never reaches is not marked at all (a later visit to a room on the route is noted on its label)
    if (index >= ROUTE_STOPS) continue;
    const entry: RouteStop = { team, index, roomId: stop.room_id, again: stop.kind === "re_search", also: [] };
    byRoom.set(stop.room_id, entry);
    out.stops.push(entry);
  }
  plan.teams.forEach((stops, team) => {
    let from = ent?.room_id ?? null;
    stops.slice(0, ROUTE_STOPS).forEach((stop) => {
      const path = from ? fastestPath(adj, from, stop.room_id) : null;
      from = stop.room_id;
      if (!path) return;
      let cur = anchors.get(path[0]!.id);
      for (const node of path.slice(1)) {
        const next = anchors.get(node.id);
        const e = node.via;
        if (cur && next && e) {
          const pts: RoutePoint[] = [cur];
          if (cur.level === next.level) {
            if (e.xy) pts.push({ x: e.xy[0], y: e.xy[1], level: cur.level }); // through the door, not the wall
          } else if (e.bottom_xy && e.top_xy) {
            const climbing = next.level > cur.level;
            const [start, end] = climbing ? [e.bottom_xy, e.top_xy] : [e.top_xy, e.bottom_xy];
            pts.push({ x: start[0], y: start[1], level: cur.level }, { x: end[0], y: end[1], level: next.level });
          }
          pts.push(next);
          line(team, pts);
        }
        cur = next;
      }
    });
  });
  return out;
}

/** Something drawn in HTML over the model at a point of the scene. */
interface Label {
  key: string;
  kind: "hazard" | "stop" | "entrance" | "staging";
  roomId?: string;
  pos: THREE.Vector3;
  title: string;
  /** "above": the label stands on its point; "centre": it sits on it */
  anchor: "above" | "centre";
  node: ReactNode;
  /** labels must not cover it (a fire-room chip, the entrance): its box is kept free */
  keepFree?: boolean;
  /** stops: the order in which they claim space, lowest first */
  rank?: number;
  /** stops: what the pointer shows over the number or the label */
  hover?: string;
  /** stops on the route: the floating name label, tied by a leader to `foot` (the room, or the cone over the next stop) */
  full?: { node: ReactNode; foot: THREE.Vector3; other: boolean; coned: boolean };
}

/** A box on screen, in CSS pixels inside the mount. */
interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const overlaps = (a: Box, b: Box, gap: number) => a.x0 < b.x1 + gap && a.x1 > b.x0 - gap && a.y0 < b.y1 + gap && a.y1 > b.y0 - gap;

/** A leader on screen: from the room's point up to its label. */
interface Leader {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Does the segment pass through the box (grown by `gap`)? Clipping the segment to the box leaves something. */
function crosses(l: Leader, b: Box, gap: number): boolean {
  let t0 = 0;
  let t1 = 1;
  const dx = l.x1 - l.x0;
  const dy = l.y1 - l.y0;
  for (const [d, lo, hi] of [
    [dx, b.x0 - gap - l.x0, b.x1 + gap - l.x0],
    [dy, b.y0 - gap - l.y0, b.y1 + gap - l.y0],
  ] as const) {
    if (Math.abs(d) < 1e-9) {
      if (lo > 0 || hi < 0) return false;
      continue;
    }
    const a = lo / d;
    const c = hi / d;
    t0 = Math.max(t0, Math.min(a, c));
    t1 = Math.min(t1, Math.max(a, c));
    if (t0 > t1) return false;
  }
  return true;
}

/** How one stop is drawn right now. */
export interface StopMark {
  key: string;
  roomId: string;
  /** "label": number and room name on a leader; "badge": number only; "hidden": its point is out of view */
  form: "label" | "badge" | "hidden";
  /** the label's or the badge's box: [left, top, right, bottom] */
  box: [number, number, number, number] | null;
  /** where the leader ends on the room */
  foot: [number, number] | null;
  text: string;
}

/** What a browser check can ask the view (set on the mount element as `view3d`). */
export interface View3DProbe {
  /** model coordinates (x east, y north, z up) -> CSS pixels inside the mount */
  project: (x: number, y: number, z: number) => [number, number];
  /** the framed model's outline on screen, in CSS pixels inside the mount: [left, top, right, bottom] */
  frame: () => [number, number, number, number] | null;
  /** what the scene is drawing besides the building */
  drawn: () => { routeLines: number; marker: boolean };
  /** every stop mark as laid out on the last frame */
  stops: () => StopMark[];
  /** where a room's marks stand: model x / y and the level, or null for a room the model does not have */
  anchor: (roomId: string) => { x: number; y: number; level: number } | null;
}

/**
 * The building model in 3D, coloured by the posterior. The geometry comes from the backend's
 * unified model: surveyed walls with their door and window openings where a scan measured
 * them, a generated layout elsewhere. Each storey stands on a slab, the outer shell is drawn
 * more solid than the partitions, and by default the street side faces the viewer. While route
 * planning is on, the way in and the route between the next stops are drawn through the model.
 * Double-click a room to select it.
 */
export function Building3D({ building, posterior, plan, floor, onSelectFloor, onSelectRoom, policy, closedRoomIds }: Props) {
  const [rawPayload, setPayload] = useState<Model3DPayload | null>(null);
  /** the model as drawn: storeys pulled apart by about one storey's height (floorGap) */
  const payload = useMemo(() => (rawPayload ? separateFloors(rawPayload, floorGap(rawPayload)) : null), [rawPayload]);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("cutaway");
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const three = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    levels: LevelMeshes[];
    marker: THREE.Mesh;
    render: () => void;
    /** frame the given levels; skipped once the dispatcher has moved the camera, unless forced */
    fit: (visible: (index: number) => boolean, force?: boolean) => void;
  } | null>(null);
  /** runs after every frame: moves the HTML labels (hazard chips, stop numbers and names, entrance) over their points */
  const afterRender = useRef<() => void>(() => undefined);
  /** the stop marks as the last frame laid them out: read by the pointer (hover text) and the test probe */
  const stopMarks = useRef<StopMark[]>([]);
  const anchorsRef = useRef(new Map<string, RoutePoint>());
  /** the press in progress on the model (pointer start, and whether it has moved far enough to be a drag) */
  const press = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  /** when the last orbit or pan drag ended, so its closing click is not taken for half of a double-click */
  const dragEndedAt = useRef(-Infinity);
  /** which levels the view currently shows, kept for the resize handler and the reset button */
  const visibleRef = useRef<(index: number) => boolean>(() => true);
  /** direction from the model to the default camera */
  const viewDirRef = useRef(new THREE.Vector3(-0.5, 0.62, 0.86).normalize());
  const rooms = useMemo(() => roomIndex(building), [building]);

  useEffect(() => {
    let cancelled = false;
    setPayload(null);
    setError(null);
    api
      .model3d(building.id)
      .then((p) => !cancelled && setPayload(p))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [building.id]);

  // Default camera side: the street front. Declared before the scene so a new scene is framed from it.
  const viewDir = useMemo(() => (payload ? viewDirection(payload, building) : null), [payload, building]);
  useEffect(() => {
    if (!viewDir) return;
    viewDirRef.current = viewDir;
    three.current?.fit(visibleRef.current);
  }, [viewDir]);

  // Scene: built once per payload.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !payload) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    // The drawing buffer is the mount's size times the device pixel ratio; the element itself must
    // stay the mount's size in CSS pixels. Without this the canvas lays out at its buffer size on
    // any display scaled above 100% and the model ends up cropped in the bottom-right corner.
    const canvas = renderer.domElement;
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    mount.appendChild(canvas);
    const scene = new THREE.Scene();
    const b = payload.bounds;
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    const span = modelSpan(payload);
    const camera = new THREE.PerspectiveCamera(38, 1, span / 200, span * 20);
    scene.add(camera);
    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    // the key light rides with the camera (from over the viewer's left shoulder), so the faces
    // toward the viewer are lit whichever side of the building the view starts on or orbits to
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-0.6, 1, 0.5);
    sun.target.position.set(0, 0, -1);
    camera.add(sun, sun.target);
    const controls = new OrbitControls(camera, canvas);
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    // left-drag orbits, right-drag (or shift / ctrl + left-drag) pans, scroll zooms; on touch one
    // finger orbits and two pinch and pan. Panning slides the camera and the point it orbits round
    // together, in the plane of the screen, by a distance that scales with how far the camera is,
    // so a drag moves the model the same on screen zoomed in or out
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.panSpeed = 1;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    // "moved by hand" means the camera actually changed during a drag or a scroll: a plain click or
    // a double-click on a room starts an interaction too, and must not stop the view re-framing
    let userMoved = false;
    let interacting = false;
    controls.addEventListener("start", () => {
      interacting = true;
    });
    controls.addEventListener("end", () => {
      interacting = false;
    });
    // outline of each level in scene coordinates (the hull of its rooms and walls, at the slab and
    // at the ceiling), so the camera can frame just the levels on show and frame them tightly. Walls
    // count: a scan can have walls around space that has no room outline.
    const levelPoints = new Map<number, THREE.Vector3[]>();
    for (const lv of payload.levels) {
      const flat: [number, number][] = [];
      for (const s of payload.spaces) if (s.l === lv.i) for (const p of s.p) flat.push(p);
      for (const w of payload.walls) if (w.l === lv.i) flat.push(w.a, w.b);
      const pts: THREE.Vector3[] = [];
      for (const p of hull2d(flat)) {
        pts.push(new THREE.Vector3(...v3(p[0], p[1], lv.z0 - SLAB_M)), new THREE.Vector3(...v3(p[0], p[1], lv.z0 + lv.h)));
      }
      if (pts.length) levelPoints.set(lv.i, pts);
    }
    // where the orbit point may be panned to: the building's box plus a margin, and not below the
    // ground, so the model cannot be slid off screen and lost
    const panBox = new THREE.Box3();
    for (const lp of levelPoints.values()) for (const p of lp) panBox.expandByPoint(p);
    if (panBox.isEmpty()) panBox.setFromCenterAndSize(new THREE.Vector3(cx, 0, -cy), new THREE.Vector3(span, span, span));
    panBox.expandByScalar(Math.max(4, span * PAN_MARGIN));
    panBox.min.y = Math.max(panBox.min.y, -SLAB_M);
    const clamped = new THREE.Vector3();
    const keepInBounds = () => {
      clamped.copy(controls.target).clamp(panBox.min, panBox.max);
      if (clamped.equals(controls.target)) return;
      // move the camera by the same amount, so the view direction and the distance are kept
      camera.position.add(clamped.sub(controls.target));
      controls.target.add(clamped);
    };
    const framed = (visible: (index: number) => boolean) => {
      let pts: THREE.Vector3[] = [];
      for (const [index, lp] of levelPoints) if (visible(index)) pts = pts.concat(lp);
      if (pts.length === 0) for (const lp of levelPoints.values()) pts = pts.concat(lp);
      return pts;
    };
    const fit = (visible: (index: number) => boolean, force = false) => {
      if (userMoved && !force) return;
      if (force) userMoved = false;
      const pts = framed(visible);
      if (pts.length === 0) return;
      fitCamera(camera, controls, pts, viewDirRef.current);
      render();
    };
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(span * 4, span * 4),
      new THREE.MeshBasicMaterial({ color: 0x0b1118 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -SLAB_M - 0.05, -cy);
    scene.add(ground);
    const levels = buildLevels(payload);
    for (const lv of levels) scene.add(lv.group);
    const marker = new THREE.Mesh(
      new THREE.ConeGeometry(Math.max(0.5, span / 70), markerHeight(span), 16),
      new THREE.MeshBasicMaterial({ color: 0xffa62b }),
    );
    marker.rotation.x = Math.PI; // point down at the room
    marker.visible = false;
    scene.add(marker);

    // Shell walls on the floor in focus that stand between the viewer and the rooms are faded, so
    // the rooms and their colours stay in view from any side; the walls on the far side stay solid.
    const look = new THREE.Vector3();
    const shade = () => {
      camera.getWorldDirection(look);
      for (const lv of levels) {
        for (const part of lv.walls) {
          // > 0 when the outside of these walls faces the viewer
          const facing = part.fade && part.outward ? -part.outward.dot(look) : -1;
          const near = THREE.MathUtils.smoothstep(facing, -0.05, 0.3);
          part.material.opacity = part.opacity * (1 - (1 - NEAR_SHELL) * near);
          part.material.depthWrite = part.write && near < 0.5;
          // partitions first, then the solid shell, then glass, and the faded walls last, over everything behind them
          part.mesh.renderOrder = near >= 0.5 ? 4 : part.kind === "inner" ? 1 : part.kind === "shell" ? 2 : 3;
        }
      }
    };
    let queued = false;
    const render = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        shade();
        renderer.render(scene, camera);
        afterRender.current();
      });
    };
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return; // the tab is hidden: the observer calls again when it is shown
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      fit(visibleRef.current); // a panel that changes shape re-frames the model (until the camera is moved by hand)
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    // a window dragged to a display with another scaling, or browser zoom: same CSS size, new pixel ratio
    let ratioQuery: MediaQueryList | null = null;
    const onRatio = () => {
      watchRatio();
      resize();
    };
    const watchRatio = () => {
      ratioQuery?.removeEventListener("change", onRatio);
      ratioQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      ratioQuery.addEventListener("change", onRatio);
    };
    watchRatio();
    controls.addEventListener("change", () => {
      keepInBounds();
      if (interacting) userMoved = true;
      render();
    });
    resize();
    three.current = { renderer, scene, camera, controls, levels, marker, render, fit };
    const probe: View3DProbe = {
      project: (x, y, z) => {
        const p = new THREE.Vector3(...v3(x, y, z)).project(camera);
        return [((p.x + 1) / 2) * mount.clientWidth, ((1 - p.y) / 2) * mount.clientHeight];
      },
      frame: () => {
        const pts = framed(visibleRef.current);
        if (pts.length === 0) return null;
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        const p = new THREE.Vector3();
        for (const c of pts) {
          p.copy(c).project(camera);
          x0 = Math.min(x0, p.x);
          x1 = Math.max(x1, p.x);
          y0 = Math.min(y0, p.y);
          y1 = Math.max(y1, p.y);
        }
        const w = mount.clientWidth;
        const h = mount.clientHeight;
        return [((x0 + 1) / 2) * w, ((1 - y1) / 2) * h, ((x1 + 1) / 2) * w, ((1 - y0) / 2) * h];
      },
      drawn: () => {
        let routeLines = 0;
        scene.traverse((o) => {
          if (o.userData.route && o.visible) routeLines += 1;
        });
        return { routeLines, marker: marker.visible };
      },
      stops: () => stopMarks.current,
      anchor: (roomId) => anchorsRef.current.get(roomId) ?? null,
    };
    (mount as HTMLDivElement & { view3d?: View3DProbe }).view3d = probe;
    mount.dataset.ready = "true";
    return () => {
      three.current = null;
      observer.disconnect();
      ratioQuery?.removeEventListener("change", onRatio);
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
        else mat?.dispose();
      });
      renderer.dispose();
      canvas.remove();
      delete (mount as HTMLDivElement & { view3d?: View3DProbe }).view3d;
      delete mount.dataset.ready;
    };
  }, [payload]);

  // Which floors are visible, and how solid their walls are.
  const levelNumbers = useMemo(() => (payload ? payload.levels.map((l) => l.i) : []), [payload]);
  const shown = levelNumbers.includes(floor) ? floor : (levelNumbers[0] ?? 1);
  const isVisible = useCallback(
    (index: number) => (mode === "all" ? true : mode === "floor" ? index === shown : index <= shown),
    [mode, shown],
  );
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    visibleRef.current = isVisible;
    for (const lv of t.levels) {
      lv.group.visible = isVisible(lv.index);
      const focus = lv.index === shown;
      for (const part of lv.walls) {
        // The floor in focus as built, its near shell walls faded. Under a cutaway the shell of the
        // storeys below stays solid, so they read as the building's mass. In the whole-building
        // view every other storey is see-through.
        const shell = part.kind === "shell";
        if (focus) part.opacity = part.base;
        else if (part.kind === "glass") part.opacity = mode === "all" ? 0.12 : part.base;
        else if (mode === "all") part.opacity = (shell ? 0.3 : 0.14) * WALL_OPACITY;
        else part.opacity = shell ? WALL_OPACITY : part.base * 0.55;
        part.write = part.kind !== "glass" && (focus || (shell && mode !== "all"));
        part.fade = focus;
        // the floor in focus is the bright one; the storeys around it step back
        if (part.color) (part.material as THREE.MeshLambertMaterial).color.copy(part.color).multiplyScalar(focus ? 1 : OFF_FOCUS_TINT);
      }
    }
    t.fit(isVisible);
    t.render();
  }, [payload, mode, shown, isVisible]);

  // ---- where things stand: a room's point on its level, and each level's floor height ----
  const levelZ = useMemo(() => new Map((payload?.levels ?? []).map((l) => [l.i, l])), [payload]);
  const anchors = useMemo(() => {
    const m = new Map<string, RoutePoint>();
    if (!payload) return m;
    // over a scan's own room when the ids were mapped, else the app room
    for (const s of payload.spaces) {
      if (s.par || s.p.length < 3 || m.has(s.r)) continue;
      const c = interiorPoint(s.p);
      m.set(s.r, { x: c[0], y: c[1], level: s.l });
    }
    for (const r of building.rooms) if (!m.has(r.id)) m.set(r.id, { x: r.centroid[0], y: r.centroid[1], level: r.floor });
    return m;
  }, [payload, building]);
  anchorsRef.current = anchors;
  const place = useCallback(
    (pt: RoutePoint, lift: number): THREE.Vector3 | null => {
      const level = levelZ.get(pt.level);
      return level ? new THREE.Vector3(...v3(pt.x, pt.y, level.z0 + lift)) : null;
    },
    [levelZ],
  );

  // ---- the way in and the route between the next stops (nothing at all while route planning is off) ----
  const closedKey = (closedRoomIds ?? []).join("|");
  const route = useMemo(() => {
    if (!policy.route_enabled || !payload) return null;
    const lowest = Math.min(...payload.levels.map((l) => l.i));
    return planRoute(building, plan, policy.travel, new Set(closedKey ? closedKey.split("|") : []), anchors, lowest);
  }, [policy.route_enabled, policy.travel, payload, building, plan, closedKey, anchors]);
  useEffect(() => {
    const t = three.current;
    if (!t || !route) return;
    const byTeam = new Map<number, number[]>();
    for (const s of route.segments) {
      if (!isVisible(s.a.level) || !isVisible(s.b.level)) continue;
      const a = place(s.a, ROUTE_LIFT_M);
      const b = place(s.b, ROUTE_LIFT_M);
      if (!a || !b) continue;
      let arr = byTeam.get(s.team);
      if (!arr) byTeam.set(s.team, (arr = []));
      arr.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const lines: LineSegments2[] = [];
    for (const [team, arr] of byTeam) {
      // Two passes: the line where nothing stands in front of it, and a fainter one through floors
      // and walls, so a leg on a storey below the one in focus reads as being underneath.
      const color = TEAM_COLOR[team % TEAM_COLOR.length]!;
      for (const through of [true, false]) {
        const material = new LineMaterial({ color, linewidth: 3, transparent: true, opacity: through ? 0.4 : 1, depthTest: !through, depthWrite: false });
        const line = new LineSegments2(new LineSegmentsGeometry().setPositions(arr), material);
        line.renderOrder = through ? 9 : 10;
        line.userData.route = true;
        t.scene.add(line);
        lines.push(line);
      }
    }
    t.render();
    return () => {
      for (const line of lines) {
        t.scene.remove(line);
        line.geometry.dispose();
        line.material.dispose();
      }
      if (three.current === t) t.render();
    };
  }, [payload, route, isVisible, place]);

  // Colour the plates by probability and put the marker over the next stop (no marker while route planning is off).
  const nextStopId = route?.stops.find((s) => s.team === 0 && s.index === 0)?.roomId ?? null;
  useEffect(() => {
    const t = three.current;
    if (!t || !payload) return;
    const maxP = Math.max(...Object.values(posterior.room_probs), 1e-9);
    const color = new THREE.Color();
    for (const lv of t.levels) {
      const attr = lv.plates.geometry.getAttribute("color") as THREE.BufferAttribute;
      for (const [si, start, count] of lv.ranges) {
        const s = payload.spaces[si]!;
        const p = posterior.room_probs[s.r];
        if (p === undefined) color.copy(NO_PROB);
        else color.set(rampColor(heatT(p, maxP)));
        if (s.k === "walkway") color.multiplyScalar(0.6);
        for (let i = start; i < start + count; i++) attr.setXYZ(i, color.r, color.g, color.b);
      }
      attr.needsUpdate = true;
    }
    const at = nextStopId ? anchors.get(nextStopId) : undefined;
    const level = at ? levelZ.get(at.level) : undefined;
    t.marker.visible = !!at && !!level && isVisible(at.level);
    if (at && level) {
      const h = (t.marker.geometry as THREE.ConeGeometry).parameters.height;
      t.marker.position.set(at.x, level.z0 + level.h * 0.5 + h / 2, -at.y);
    }
    t.render();
  }, [payload, posterior, nextStopId, anchors, levelZ, isVisible]);

  // ---- hazard class (fire / smoke): from the plan's stops, display only ----
  const hazards = useMemo(() => {
    const out = new Map<string, HazardClass>();
    if (!policy.show_hazard_class) return out;
    for (const team of plan.teams) for (const st of team) if (st.hazard_class && !out.has(st.room_id)) out.set(st.room_id, st.hazard_class);
    return out;
  }, [plan, policy.show_hazard_class]);
  const hazardCounts = useMemo(() => {
    const c: Partial<Record<HazardClass, number>> = {};
    for (const h of hazards.values()) c[h] = (c[h] ?? 0) + 1;
    return c;
  }, [hazards]);

  /**
   * Labels drawn over the model. Hazard chips: rooms on the shown floor (circulation only when it
   * is the fire room or behind a closed door). The entrance, the staging spot and the stops: only
   * while route planning is on. A stop on the route (each team's next ROUTE_STOPS) has a floating
   * label with its number and room name on the levels in view. A room the route does not reach has
   * no label at all.
   */
  const labels = useMemo(() => {
    const out: Label[] = [];
    if (!payload) return out;
    const shownLevel = levelZ.get(shown);
    const hz: { open: boolean; label: Label }[] = [];
    for (const [roomId, hazard] of hazards) {
      const room = rooms.get(roomId);
      const at = anchors.get(roomId);
      if (!room || !at || !shownLevel || room.floor !== shown) continue;
      if (hazard === "open_path" && CIRCULATION.has(room.room_type)) continue;
      hz.push({
        open: hazard === "open_path",
        label: {
          key: `hazard:${roomId}`,
          kind: "hazard",
          roomId,
          pos: new THREE.Vector3(...v3(at.x, at.y, shownLevel.z0 + 0.3)),
          title: room.name,
          anchor: "centre",
          keepFree: hazard !== "open_path",
          // "Open to the fire" covers most of a smoke floor: an icon marker (named in the legend) keeps the model readable
          node: <HazardChip hazard={hazard} compact iconOnly={hazard === "open_path"} />,
        },
      });
    }
    // fire room and closed-door chips are drawn last, on top of the open-path markers
    out.push(...hz.sort((a, b) => Number(!a.open) - Number(!b.open)).map((h) => h.label));
    if (!route) return out;
    const multiTeam = plan.teams.length > 1;
    const coneTop = markerHeight(modelSpan(payload));
    const multiFloor = payload.levels.filter((l) => isVisible(l.i)).length > 1;
    route.stops.forEach((st, rank) => {
      const at = anchors.get(st.roomId);
      const level = at ? levelZ.get(at.level) : undefined;
      if (!at || !level || !isVisible(at.level)) return;
      const room = rooms.get(st.roomId);
      const next = st.index === 0;
      const team = multiTeam ? String.fromCharCode(65 + st.team) : "";
      const other = st.team % TEAM_COLOR.length === 1;
      const number = `${team}${st.index + 1}`;
      const name = room?.name ?? st.roomId;
      const floorLabel = room?.floor_label ?? String(at.level);
      const words = `${multiTeam ? `Team ${team} stop` : "Stop"} ${st.index + 1}: ${name}, floor ${floorLabel}${st.again ? ", second look" : ""}${
        st.also.length ? ` (also stop ${st.also.join(", ")})` : ""
      }`;
      // the next stop's number sits on top of the cone that points at its room
      const coned = next && st.team === 0;
      const pos = new THREE.Vector3(...v3(at.x, at.y, level.z0 + (coned ? level.h * 0.5 + coneTop : 1.4)));
      const again = st.again ? <RotateCw size={9} strokeWidth={3} aria-hidden /> : null;
      const offFloor = at.level !== shown ? <span className="text-[9px] font-semibold opacity-80">{floorTag(floorLabel)}</span> : null;
      out.push({
        key: `stop:${st.team}:${st.index}`,
        kind: "stop",
        roomId: st.roomId,
        pos,
        title: words,
        hover: words,
        rank,
        anchor: "above",
        node: (
          <span
            className={`num flex h-[18px] min-w-[18px] items-center justify-center gap-0.5 rounded-full border px-1 text-[11px] font-bold leading-none shadow shadow-black/50 ${
              next
                ? other
                  ? "border-info bg-info text-bg"
                  : "border-accent bg-accent text-accent-ink"
                : other
                  ? "border-info bg-bg/90 text-info"
                  : "border-accent bg-bg/90 text-accent"
            }`}
          >
            {number}
            {again}
            {offFloor}
          </span>
        ),
        full: {
          // the leader ends on the room's floor; over the next stop it ends on the cone that points at the room
          foot: coned ? pos : new THREE.Vector3(...v3(at.x, at.y, level.z0 + 0.06)),
          other,
          coned,
          node: (
            <span
              className={`flex items-center gap-1 whitespace-nowrap rounded border bg-bg/95 px-1.5 py-[3px] text-[11px] leading-none shadow shadow-black/60 ${
                other ? "border-info/80" : "border-accent/80"
              }`}
            >
              <span
                className={`num font-bold ${
                  next
                    ? other
                      ? "-my-px rounded-sm bg-info px-1 py-px text-bg"
                      : "-my-px rounded-sm bg-accent px-1 py-px text-accent-ink"
                    : other
                      ? "text-info"
                      : "text-accent"
                }`}
              >
                {number}
              </span>
              <span className="font-bold text-ink-dim">·</span>
              <span className="max-w-[220px] truncate font-medium leading-[1.25] text-ink">{shortName(name)}</span>
              {multiFloor ? <span className="num text-ink-dim">{floorTag(floorLabel)}</span> : null}
              {st.again ? (
                <span className="-my-px flex items-center gap-0.5 rounded-sm border border-warn/70 px-1 text-[10px] font-semibold leading-[13px] text-warn">
                  <RotateCw size={9} strokeWidth={3} aria-hidden />
                  second look
                </span>
              ) : null}
            </span>
          ),
        },
      });
    });
    const tag = (icon: ReactNode, text: string) => (
      <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-sm border border-accent bg-bg/90 px-1 py-0.5 text-[10px] font-semibold leading-none text-accent shadow shadow-black/50">
        {icon}
        {text}
      </span>
    );
    // an entrance the data only assumes (no door to the outside was found) says so
    const entranceText = route.entrance && /assumed/i.test(route.entrance.name) ? "Entrance (assumed)" : "Entrance";
    for (const [kind, spot, icon, text] of [
      ["staging", route.staging, <Ambulance size={11} aria-hidden />, "Staging"],
      ["entrance", route.entrance, <LogIn size={11} aria-hidden />, entranceText],
    ] as const) {
      const pos = spot && isVisible(spot.at.level) ? place(spot.at, ROUTE_LIFT_M) : null;
      if (spot && pos) out.push({ key: kind, kind, pos, title: spot.name, anchor: "above", keepFree: true, node: tag(icon, text) });
    }
    return out;
  }, [payload, hazards, rooms, anchors, levelZ, shown, route, plan.teams.length, isVisible, place]);
  /** label elements by key; a stop's name label is under `<key>:full` */
  const labelEls = useRef(new Map<string, HTMLDivElement>());
  const leaderEls = useRef(new Map<string, SVGGElement>());
  /** each label element's laid-out size in CSS pixels (it does not change with the camera), measured when first needed */
  const labelSizes = useRef(new Map<string, [number, number]>());
  useEffect(() => {
    labelSizes.current.clear();
    /**
     * Lay the labels out over the frame just drawn. Chips, the entrance and the staging tag stand on
     * their points. Stops then claim space in order, by number. A stop takes the nearest slot by
     * its room where its name label covers no label placed before it, no chip or tag, no earlier
     * leader and no number already placed, and where its own leader crosses none of those.
     * With no slot it shows its number only, over its room. A number that would lie on an earlier
     * stop's number, on a name, or on a chip or tag steps aside by whole badges. So no two labels
     * overlap, no label covers a number, and it is always the later stop that gives way.
     */
    afterRender.current = () => {
      const t = three.current;
      const mount = mountRef.current;
      if (!t || !mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      const p = new THREE.Vector3();
      const project = (v: THREE.Vector3) => {
        p.copy(v).project(t.camera);
        return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h, inView: p.z < 1 && Math.abs(p.x) <= 1.05 && Math.abs(p.y) <= 1.05 };
      };
      const size = (key: string): [number, number] => {
        const known = labelSizes.current.get(key);
        if (known) return known;
        const el = labelEls.current.get(key);
        // the fractional size, rounded up: text rarely ends on a whole pixel, and a box cut short would let two labels touch
        const r = el?.getBoundingClientRect();
        const measured: [number, number] = [Math.ceil(r?.width ?? 0), Math.ceil(r?.height ?? 0)];
        // a hidden tab lays nothing out: keep asking until the element has a size
        if (measured[0] > 0) labelSizes.current.set(key, measured);
        return measured;
      };
      const show = (el: HTMLElement | SVGElement | undefined, on: boolean) => {
        if (el) el.style.visibility = on ? "visible" : "hidden";
      };
      /** boxes a name label may not cover: chips and tags, then each name label as it is placed */
      const taken: Box[] = [];
      /** the name labels placed so far, and their leaders */
      const named: Box[] = [];
      const leaders: Leader[] = [];
      /** the numbers placed so far (stops that show no name), also filed by the screen squares they touch */
      const badges: Box[] = [];
      const badgeGrid = new Map<number, Box[]>();
      const squares = (b: Box, visit: (key: number) => boolean): boolean => {
        for (let cx = Math.floor(b.x0 / BADGE_CELL); cx <= Math.floor(b.x1 / BADGE_CELL); cx++) {
          for (let cy = Math.floor(b.y0 / BADGE_CELL); cy <= Math.floor(b.y1 / BADGE_CELL); cy++) {
            if (visit(cx * 100003 + cy)) return true;
          }
        }
        return false;
      };
      const addBadge = (b: Box) => {
        badges.push(b);
        squares(b, (key) => {
          const list = badgeGrid.get(key);
          if (list) list.push(b);
          else badgeGrid.set(key, [b]);
          return false;
        });
      };
      const onBadge = (b: Box) => squares(b, (key) => badgeGrid.get(key)?.some((o) => overlaps(b, o, 0)) ?? false);
      const stops: { label: Label; el: HTMLDivElement; at: ReturnType<typeof project>; spot: Box }[] = [];
      for (const label of labels) {
        const el = labelEls.current.get(label.key);
        if (!el) continue;
        const at = project(label.pos);
        const [bw, bh] = size(label.key);
        const top = label.anchor === "above" ? at.y - bh - 2 : at.y - bh / 2;
        const box: Box = { x0: at.x - bw / 2, y0: top, x1: at.x + bw / 2, y1: top + bh };
        if (label.kind === "stop") {
          stops.push({ label, el, at, spot: box });
          continue;
        }
        show(el, at.inView);
        el.style.transform = `translate(${box.x0}px, ${box.y0}px)`;
        if (at.inView && label.keepFree) taken.push(box);
      }
      const marks: StopMark[] = [];
      stops.forEach((s) => {
        const key = s.label.key;
        const full = s.label.full;
        const fullEl = labelEls.current.get(`${key}:full`);
        const leader = leaderEls.current.get(key);
        let box: Box | null = null;
        let line: Leader | null = null;
        let foot: { x: number; y: number } | null = null;
        if (full && fullEl && s.at.inView) {
          foot = project(full.foot);
          const [lw, lh] = size(`${key}:full`);
          if (lw > 0 && lw + 4 <= w) {
            for (const [dx, dy] of labelSlots(lw, lh, full.coned)) {
              // a label that would leave the panel sideways slides back in; one that would leave it at the top or bottom is not used
              const x0 = Math.min(Math.max(foot.x + dx - lw / 2, 2), w - lw - 2);
              const y0 = foot.y + dy - lh / 2;
              if (y0 < 2 || y0 + lh > h - 2) continue;
              const candidate: Box = { x0, y0, x1: x0 + lw, y1: y0 + lh };
              // not on top of its own room's point
              if (foot.x > x0 - 4 && foot.x < x0 + lw + 4 && foot.y > y0 - 4 && foot.y < y0 + lh + 4) continue;
              if (taken.some((o) => overlaps(candidate, o, 3))) continue;
              // not on a number already placed, nor on an earlier leader
              if (badges.some((o) => overlaps(candidate, o, 1))) continue;
              if (leaders.some((o) => crosses(o, candidate, 2))) continue;
              // the leader runs from the room's point to the nearest point of the label's edge
              const own: Leader = {
                x0: foot.x,
                y0: foot.y,
                x1: Math.min(Math.max(foot.x, x0 + 5), x0 + lw - 5),
                y1: Math.min(Math.max(foot.y, y0), y0 + lh),
              };
              if (named.some((o) => crosses(own, o, 1))) continue;
              if (badges.some((o) => crosses(own, o, 0))) continue;
              box = candidate;
              line = own;
              break;
            }
          }
        }
        if (box && line && foot && fullEl) {
          taken.push(box);
          // the point where its leader ends stays in view too
          taken.push({ x0: foot.x - 3, y0: foot.y - 3, x1: foot.x + 3, y1: foot.y + 3 });
          named.push(box);
          leaders.push(line);
          fullEl.style.transform = `translate(${box.x0}px, ${box.y0}px)`;
          if (leader) {
            const stroke = leader.firstElementChild as SVGLineElement;
            const dot = leader.lastElementChild as SVGCircleElement;
            stroke.setAttribute("x1", String(line.x0));
            stroke.setAttribute("y1", String(line.y0));
            stroke.setAttribute("x2", String(line.x1));
            stroke.setAttribute("y2", String(line.y1));
            dot.setAttribute("cx", String(foot.x));
            dot.setAttribute("cy", String(foot.y));
          }
        } else {
          if (s.at.inView) {
            const home = s.spot;
            const bw = home.x1 - home.x0 + 1;
            const bh = home.y1 - home.y0 + 1;
            const place = (k: number): Box => {
              const [nx, ny] = BADGE_NUDGES[k]!;
              return { x0: home.x0 + nx * bw, y0: home.y0 + ny * bh, x1: home.x1 + nx * bw, y1: home.y1 + ny * bh };
            };
            // off every name, chip and tag (and the points the leaders end on)
            const offLabels = (b: Box) => !taken.some((o) => overlaps(b, o, 1));
            const inPanel = (b: Box) => b.x0 >= 0 && b.y0 >= 0 && b.x1 <= w && b.y1 <= h;
            let spot: Box | null = null;
            // near its point, in the panel and clear of every name, chip, tag, number and leader
            for (let k = 0; k < BADGE_NEAR && !spot; k++) {
              const b = place(k);
              if (inPanel(b) && offLabels(b) && !onBadge(b) && !leaders.some((o) => crosses(o, b, 0))) spot = b;
            }
            // failing that (a crowd of numbers seen from far away, or a room at the panel's edge): clear of every name, chip and tag
            for (let k = 0; k < BADGE_NUDGES.length && !spot; k++) {
              const b = place(k);
              if (offLabels(b)) spot = b;
            }
            s.spot = spot ?? home;
            addBadge(s.spot);
          }
          s.el.style.transform = `translate(${s.spot.x0}px, ${s.spot.y0}px)`;
        }
        show(fullEl, !!box);
        show(leader, !!box);
        show(s.el, !box && s.at.inView);
        const drawn = box ?? (s.at.inView ? s.spot : null);
        marks.push({
          key,
          roomId: s.label.roomId ?? "",
          form: box ? "label" : s.at.inView ? "badge" : "hidden",
          box: drawn ? [drawn.x0, drawn.y0, drawn.x1, drawn.y1] : null,
          foot: box && foot ? [foot.x, foot.y] : null,
          text: s.label.hover ?? s.label.title,
        });
      });
      stopMarks.current = marks;
    };
    afterRender.current();
    three.current?.render();
    // a font that arrives late changes the labels' sizes
    let cancelled = false;
    void document.fonts?.ready.then(() => {
      if (cancelled) return;
      labelSizes.current.clear();
      three.current?.render();
    });
    return () => {
      cancelled = true;
    };
  }, [labels]);

  function pick(e: React.MouseEvent): { roomId: string; label: string } | null {
    const t = three.current;
    const mount = mountRef.current;
    if (!t || !mount || !payload) return null;
    const rect = mount.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, t.camera);
    const meshes = t.levels.filter((l) => l.group.visible).map((l) => l.plates);
    const hit = ray.intersectObjects(meshes, false)[0];
    if (!hit || hit.faceIndex == null) return null;
    const lv = t.levels.find((l) => l.plates === hit.object);
    const s = lv ? payload.spaces[lv.triSpace[hit.faceIndex]!] : undefined;
    if (!s) return null;
    const room = rooms.get(s.r);
    const p = posterior.room_probs[s.r];
    const part = s.par && s.n ? ` - ${s.n}` : "";
    const hz = hazards.get(s.r);
    return {
      roomId: s.r,
      label: `${room?.name ?? s.n}${part} - floor ${room?.floor_label ?? s.l}${p !== undefined ? ` - P ${pct(p)}` : ""}${
        hz ? ` - ${HAZARD_STYLE[hz].label}` : ""
      } (${FIDELITY_LABEL[s.f] ?? s.f})`,
    };
  }

  const fidelity = payload?.counts.fidelity ?? {};
  /** how many stops per team the route runs through (each has a name label) */
  const markedStops = route ? Math.max(0, ...plan.teams.map((team) => Math.min(team.length, ROUTE_STOPS))) : 0;
  /** stops on the route that stand on a floor the view has cut away */
  const routeStopsOffView = route ? route.stops.length - labels.filter((l) => l.full).length : 0;
  return (
    <Panel
      title="3D model"
      icon={<Boxes size={15} aria-hidden />}
      actions={
        <>
          {levelNumbers.length > 1 ? (
            <label className="flex items-center gap-1.5 text-xs text-ink-mute">
              Floor
              <select
                aria-label="3D floor"
                value={shown}
                onChange={(e) => onSelectFloor(Number(e.target.value))}
                className="rounded border border-line bg-bg px-1.5 py-1 text-xs text-ink"
              >
                {[...payload!.levels].reverse().map((l) => (
                  <option key={l.i} value={l.i}>
                    {l.label}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <button
            onClick={() => three.current?.fit(visibleRef.current, true)}
            title="Frame the model again, from the front"
            className="rounded border border-line px-2 py-1 text-xs text-ink-dim hover:bg-raised hover:text-ink"
          >
            Reset view
          </button>
          <div className="flex overflow-hidden rounded border border-line" role="group" aria-label="3D view">
            {VIEW_MODES.map((m) => (
              <button
                key={m.key}
                title={m.title}
                aria-pressed={mode === m.key}
                onClick={() => setMode(m.key)}
                className={`border-l border-line px-2 py-1 text-xs first:border-l-0 ${
                  mode === m.key ? "bg-ink font-semibold text-bg" : "text-ink-dim hover:bg-raised hover:text-ink"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </>
      }
      className="h-full"
      bodyClassName="flex flex-col"
    >
      <div className="relative min-h-[320px] w-full flex-1 overflow-hidden bg-bg">
        <div
          ref={mountRef}
          data-model3d={payload?.id ?? ""}
          data-spaces={payload?.counts.spaces ?? 0}
          data-walls={payload?.counts.walls ?? 0}
          className="absolute inset-0 cursor-grab active:cursor-grabbing"
          onMouseMove={(e) => {
            const rect = mountRef.current!.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            // a stop's number or label first (it is drawn over the rooms), then the room under the pointer
            const stop = stopMarks.current.find((m) => m.box && x >= m.box[0] - 2 && x <= m.box[2] + 2 && y >= m.box[1] - 2 && y <= m.box[3] + 2);
            const hit = stop ? null : pick(e);
            setHover(stop ? { x, y, text: stop.text } : hit ? { x, y, text: hit.label } : null);
          }}
          onMouseLeave={() => setHover(null)}
          onPointerDown={(e) => {
            press.current = { x: e.clientX, y: e.clientY, moved: false };
          }}
          onPointerMove={(e) => {
            const p = press.current;
            if (p && !p.moved && Math.hypot(e.clientX - p.x, e.clientY - p.y) > DRAG_PX) p.moved = true;
          }}
          onPointerUp={() => {
            if (press.current?.moved) dragEndedAt.current = performance.now();
            press.current = null;
          }}
          onDoubleClick={(e) => {
            // the click that ends an orbit or a pan drag can pair with the next click into a double-click
            if (performance.now() - dragEndedAt.current < DRAG_CLICK_MS) return;
            const hit = pick(e);
            if (hit) onSelectRoom(hit.roomId);
          }}
        />
        {labels.length > 0 ? (
          <div className="pointer-events-none absolute inset-0 isolate" aria-hidden>
            {/* leaders: a thin line from each name label down to its room, under the labels */}
            <svg className="absolute inset-0 h-full w-full" style={{ filter: "drop-shadow(0 0 1px rgb(0 0 0 / 0.95))" }}>
              {labels.map((c) =>
                c.full ? (
                  <g
                    key={c.key}
                    ref={(el) => {
                      if (el) leaderEls.current.set(c.key, el);
                      else leaderEls.current.delete(c.key);
                    }}
                    data-leader={c.roomId}
                    className={c.full.other ? "text-info" : "text-accent"}
                    style={{ visibility: "hidden" }}
                  >
                    <line stroke="currentColor" strokeWidth={1.25} strokeLinecap="round" />
                    <circle r={2.5} fill="currentColor" />
                  </g>
                ) : null,
              )}
            </svg>
            {labels.map((c) => (
              <Fragment key={c.key}>
                <div
                  ref={(el) => {
                    if (el) labelEls.current.set(c.key, el);
                    else labelEls.current.delete(c.key);
                  }}
                  data-label={c.kind}
                  data-room={c.roomId}
                  className="absolute left-0 top-0"
                  // an earlier stop's number lies over a later one's where two rooms meet on screen
                  style={{ visibility: "hidden", zIndex: c.rank !== undefined ? 1000 - Math.min(c.rank, 900) : undefined }}
                  title={c.title}
                >
                  {c.node}
                </div>
                {c.full ? (
                  <div
                    ref={(el) => {
                      if (el) labelEls.current.set(`${c.key}:full`, el);
                      else labelEls.current.delete(`${c.key}:full`);
                    }}
                    data-label="stop-name"
                    data-room={c.roomId}
                    className="absolute left-0 top-0"
                    style={{ visibility: "hidden", zIndex: 1001 }}
                    title={c.title}
                  >
                    {c.full.node}
                  </div>
                ) : null}
              </Fragment>
            ))}
          </div>
        ) : null}
        {!payload ? (
          <p className="absolute inset-0 flex items-center justify-center text-xs text-ink-mute">
            {error ? `3D model unavailable: ${error}` : "Building the 3D model…"}
          </p>
        ) : null}
        {hover ? (
          <div
            className="pointer-events-none absolute z-10 max-w-[320px] rounded border border-line-strong bg-raised px-2 py-1 text-[11px] text-ink shadow-lg"
            style={{ left: Math.max(0, Math.min(hover.x + 12, (mountRef.current?.clientWidth ?? 860) - 330)), top: hover.y + 12 }}
          >
            {hover.text}
          </div>
        ) : null}
      </div>
      {policy.show_hazard_class && hazards.size > 0 ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line px-3 py-1.5 text-[11px] text-ink-dim">
          <span className="eyebrow !text-[10px]">Hazard class</span>
          {(["fire_room", "open_path", "closed_door"] as const).map((h) =>
            hazardCounts[h] ? (
              <span key={h} className="flex items-center gap-1">
                <HazardChip hazard={h} compact />
                <span className="num">{hazardCounts[h]}</span>
              </span>
            ) : null,
          )}
          <span className="flex basis-full items-center gap-1.5 text-ink">
            <Info size={12} className="shrink-0 text-ink-mute" aria-hidden />
            {HAZARD_LEGEND}
          </span>
        </div>
      ) : null}
      {!policy.route_enabled ? (
        <div className="border-t border-line px-3 py-1.5 text-[11px] text-ink-dim">
          Route planning is off: no stops are marked. Colours show where reports point, as information.
        </div>
      ) : markedStops > 0 ? (
        <div className="flex flex-wrap items-center gap-x-2 border-t border-line px-3 py-1.5 text-[11px] text-ink-dim">
          <span className="inline-block h-[3px] w-5 shrink-0 rounded-full bg-accent" aria-hidden />
          <span>
            Route: the way in {markedStops > 1 ? `and between the next ${markedStops} stops` : "to the next stop"}
            {plan.teams.length > 1 ? " of each team (A, B)" : ""}, on foot and by stairs, as the plan&apos;s travel times count it. Each of
            those stops is named over its room; where two names would overlap, the later stop shows its number only.
            {routeStopsOffView > 0
              ? ` ${routeStopsOffView} of them ${routeStopsOffView === 1 ? "is" : "are"} on a floor this view does not show (Whole building shows every floor).`
              : ""}          </span>
        </div>
      ) : null}
      {payload ? (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-line px-3 py-2 text-[11px] text-ink-mute">
          <span className="text-ink-dim">{payload.source ?? payload.archetype}</span>
          <span>
            {Object.entries(fidelity)
              .map(([k, n]) => `${n} ${FIDELITY_LABEL[k] ?? k}`)
              .join(" · ")}
          </span>
          <span className="ml-auto">Drag to orbit · right-drag or shift-drag to pan · scroll to zoom · double-click a room to select it</span>
        </div>
      ) : null}
    </Panel>
  );
}

/** "3" -> "F3", "Floor 3" -> "F3": a floor label short enough for a stop number */
function floorTag(label: string): string {
  return /^\d/.test(label) ? `F${label}` : label.replace(/^floor\s*/i, "F");
}

/**
 * A point inside a room outline of any shape, for the room's marks and the route. The area
 * centroid when it lies in the room. For an outline that bends around it (an L, a U, a wedge of
 * a ring) the centroid can fall outside: then the middle of the widest stretch of room along
 * lines across the outline, which is always inside.
 */
export function interiorPoint(points: [number, number][]): [number, number] {
  const n = points.length;
  let area = 0;
  let cx = 0;
  let cy = 0;
  let mx = 0;
  let my = 0;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % n]!;
    const cross = a[0] * b[1] - b[0] * a[1];
    area += cross;
    cx += (a[0] + b[0]) * cross;
    cy += (a[1] + b[1]) * cross;
    mx += a[0] / n;
    my += a[1] / n;
    y0 = Math.min(y0, a[1]);
    y1 = Math.max(y1, a[1]);
  }
  if (Math.abs(area) < 1e-9) return [mx, my];
  cx /= 3 * area;
  cy /= 3 * area;
  if (pointInPolygon(cx, cy, points)) return [cx, cy];
  let best: [number, number] | null = null;
  let bestScore = -Infinity;
  for (let k = 1; k < 16; k++) {
    const y = y0 + ((y1 - y0) * k) / 16;
    const xs: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = points[i]!;
      const b = points[(i + 1) % n]!;
      if (a[1] > y !== b[1] > y) xs.push(a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1]));
    }
    xs.sort((p, q) => p - q);
    for (let j = 0; j + 1 < xs.length; j += 2) {
      // the widest stretch wins; between equals, the one nearest the centroid
      const score = xs[j + 1]! - xs[j]! - 0.05 * Math.hypot((xs[j]! + xs[j + 1]!) / 2 - cx, y - cy);
      if (score > bestScore) {
        bestScore = score;
        best = [(xs[j]! + xs[j + 1]!) / 2, y];
      }
    }
  }
  return best ?? [mx, my];
}

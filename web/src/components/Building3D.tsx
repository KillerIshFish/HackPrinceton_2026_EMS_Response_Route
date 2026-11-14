import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import type { ReactNode } from "react";
import { Ambulance, Boxes, DoorClosed, DoorOpen, Flame, Info, LogIn } from "lucide-react";
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
  stamped: "from a scan of the unit type",
  cluster: "scan positions only",
  template: "generated layout",
  roster: "floor known, position unknown",
};

/** stops per team that get a number and a drawn route, as on the floor plan */
const MARKED_STOPS = 5;
/** thickness of the floor slab drawn under each level's rooms */
const SLAB_M = 0.3;
/** height of the route line above the floor it runs on */
const ROUTE_LIFT_M = 1.0;
/** route and stop colour per team (the stop numbers also carry the team letter) */
const TEAM_COLOR = [0xf2a93b, 0x79a8d8];

/** longest dimension of the model, for sizing things that must stay readable on any building */
function modelSpan(payload: Model3DPayload): number {
  const b = payload.bounds;
  return Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1, 8);
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
        ? new THREE.MeshLambertMaterial({ color: style.color, transparent: true, opacity: style.base })
        : new THREE.MeshBasicMaterial({ color: 0x7fc4ff, transparent: true, opacity: GLASS_OPACITY, depthWrite: false });
      const mesh = new THREE.Mesh(geo, material);
      const angle = (b.dir * Math.PI) / 4;
      group.add(mesh);
      walls.push({
        mesh,
        material,
        kind: b.kind,
        base: style?.base ?? GLASS_OPACITY,
        outward: b.dir < 0 ? null : new THREE.Vector3(Math.cos(angle), 0, -Math.sin(angle)),
        color: style ? new THREE.Color(style.color) : null,
        opacity: style?.base ?? GLASS_OPACITY,
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

interface RouteData {
  /** straight pieces of the way in and between the marked stops */
  segments: { team: number; a: RoutePoint; b: RoutePoint }[];
  stops: { team: number; index: number; roomId: string }[];
  entrance: { name: string; at: RoutePoint } | null;
  staging: { name: string; at: RoutePoint } | null;
}

type Hop = { to: string; t: number; edge: Edge };

/**
 * Room-to-room travel times as the search plan counts them (core/building_graph.py): walk each
 * edge at the profile's speed, add the stair rate per floor climbed, no elevator rides, and
 * nothing through a room the officer closed.
 */
function travelGraph(building: Building, travel: TravelProfile, closed: Set<string>): Map<string, Hop[]> {
  const rooms = new Map(building.rooms.map((r) => [r.id, r]));
  const adj = new Map<string, Hop[]>();
  for (const e of building.edges) {
    if (e.kind === "elevator" || closed.has(e.u) || closed.has(e.v)) continue;
    const u = rooms.get(e.u);
    const v = rooms.get(e.v);
    if (!u || !v) continue;
    const length =
      e.length_m ?? Math.hypot(u.centroid[0] - v.centroid[0], u.centroid[1] - v.centroid[1], u.centroid[2] - v.centroid[2]);
    let t = length / Math.max(travel.walk_mps, 0.1);
    if (e.kind === "stairs") {
      const lower = Math.min(u.floor, v.floor);
      const floors = Math.abs(Math.trunc(e.floors || 0)) || Math.abs(u.floor - v.floor);
      for (let f = lower + 1; f <= lower + floors; f++) {
        t += travel.high_floor_from != null && f > travel.high_floor_from ? travel.stair_s_per_floor_high : travel.stair_s_per_floor;
      }
    }
    for (const [from, to] of [[e.u, e.v], [e.v, e.u]] as const) {
      let list = adj.get(from);
      if (!list) adj.set(from, (list = []));
      list.push({ to, t, edge: e });
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
  const marked = new Set<string>();
  plan.teams.forEach((stops, team) => {
    let from = ent?.room_id ?? null;
    stops.slice(0, MARKED_STOPS).forEach((stop, index) => {
      if (!marked.has(stop.room_id)) {
        marked.add(stop.room_id);
        out.stops.push({ team, index, roomId: stop.room_id });
      }
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
}

/** What a browser check can ask the view (set on the mount element as `view3d`). */
export interface View3DProbe {
  /** model coordinates (x east, y north, z up) -> CSS pixels inside the mount */
  project: (x: number, y: number, z: number) => [number, number];
  /** the framed model's outline on screen, in CSS pixels inside the mount: [left, top, right, bottom] */
  frame: () => [number, number, number, number] | null;
  /** what the scene is drawing besides the building */
  drawn: () => { routeLines: number; marker: boolean };
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
  const [payload, setPayload] = useState<Model3DPayload | null>(null);
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
  /** runs after every frame: moves the HTML labels (hazard chips, stop numbers, entrance) over their points */
  const afterRender = useRef<() => void>(() => undefined);
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
        else if (mode === "all") part.opacity = shell ? 0.3 : 0.14;
        else part.opacity = shell ? 1 : part.base * 0.55;
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
      const c = centroid(s.p);
      m.set(s.r, { x: c[0], y: c[1], level: s.l });
    }
    for (const r of building.rooms) if (!m.has(r.id)) m.set(r.id, { x: r.centroid[0], y: r.centroid[1], level: r.floor });
    return m;
  }, [payload, building]);
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
   * is the fire room or behind a closed door). Stop numbers, the entrance and the staging spot:
   * on the levels in view, and only while route planning is on.
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
    for (const st of route.stops) {
      const at = anchors.get(st.roomId);
      const level = at ? levelZ.get(at.level) : undefined;
      if (!at || !level || !isVisible(at.level)) continue;
      const room = rooms.get(st.roomId);
      const next = st.index === 0;
      const team = multiTeam ? String.fromCharCode(65 + st.team) : "";
      const other = st.team % TEAM_COLOR.length === 1;
      out.push({
        key: `stop:${st.team}:${st.index}`,
        kind: "stop",
        roomId: st.roomId,
        // the next stop's number sits on top of the cone that points at its room
        pos: new THREE.Vector3(...v3(at.x, at.y, level.z0 + (next && st.team === 0 ? level.h * 0.5 + coneTop : 1.4))),
        title: `${multiTeam ? `Team ${team} ` : ""}stop ${st.index + 1}: ${room?.name ?? st.roomId}, floor ${room?.floor_label ?? at.level}`,
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
            {team}
            {st.index + 1}
            {at.level !== shown ? <span className="text-[9px] font-semibold opacity-80">{floorTag(room?.floor_label ?? String(at.level))}</span> : null}
          </span>
        ),
      });
    }
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
      if (spot && pos) out.push({ key: kind, kind, pos, title: spot.name, anchor: "above", node: tag(icon, text) });
    }
    return out;
  }, [payload, hazards, rooms, anchors, levelZ, shown, route, plan.teams.length, isVisible, place]);
  const labelEls = useRef(new Map<string, HTMLDivElement>());
  useEffect(() => {
    afterRender.current = () => {
      const t = three.current;
      const mount = mountRef.current;
      if (!t || !mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      const p = new THREE.Vector3();
      for (const label of labels) {
        const el = labelEls.current.get(label.key);
        if (!el) continue;
        p.copy(label.pos).project(t.camera);
        const inView = p.z < 1 && Math.abs(p.x) <= 1.05 && Math.abs(p.y) <= 1.05;
        el.style.display = inView ? "" : "none";
        el.style.transform = `translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px) translate(-50%, ${
          label.anchor === "above" ? "calc(-100% - 2px)" : "-50%"
        })`;
      }
    };
    afterRender.current();
    three.current?.render();
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
  /** how many stops per team carry a number and a route leg */
  const markedStops = route ? Math.max(0, ...plan.teams.map((team) => Math.min(team.length, MARKED_STOPS))) : 0;
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
            const hit = pick(e);
            const rect = mountRef.current!.getBoundingClientRect();
            setHover(hit ? { x: e.clientX - rect.left, y: e.clientY - rect.top, text: hit.label } : null);
          }}
          onMouseLeave={() => setHover(null)}
          onDoubleClick={(e) => {
            const hit = pick(e);
            if (hit) onSelectRoom(hit.roomId);
          }}
        />
        {labels.length > 0 ? (
          <div className="pointer-events-none absolute inset-0" aria-hidden>
            {labels.map((c) => (
              <div
                key={c.key}
                ref={(el) => {
                  if (el) labelEls.current.set(c.key, el);
                  else labelEls.current.delete(c.key);
                }}
                data-label={c.kind}
                data-room={c.roomId}
                className="absolute left-0 top-0"
                title={c.title}
              >
                {c.node}
              </div>
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
            {plan.teams.length > 1 ? " of each team" : ""}, on foot and by stairs, as the plan&apos;s travel times count it.
          </span>
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
          <span className="ml-auto">Drag to orbit · scroll to zoom · double-click a room to select it</span>
        </div>
      ) : null}
    </Panel>
  );
}

/** "3" -> "F3", "Floor 3" -> "F3": a floor label short enough for a stop number */
function floorTag(label: string): string {
  return /^\d/.test(label) ? `F${label}` : label.replace(/^floor\s*/i, "F");
}

function centroid(points: [number, number][]): [number, number] {
  let x = 0;
  let y = 0;
  for (const p of points) {
    x += p[0];
    y += p[1];
  }
  return [x / points.length, y / points.length];
}

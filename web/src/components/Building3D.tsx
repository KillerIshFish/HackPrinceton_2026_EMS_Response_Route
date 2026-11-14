import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Boxes } from "lucide-react";
import { api } from "../api";
import type { Building, Model3DPayload, Posterior, SearchPlan } from "../types";
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
}

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

/** One level's meshes plus what is needed to recolour and pick its floor plates. */
interface LevelMeshes {
  index: number;
  group: THREE.Group;
  plates: THREE.Mesh;
  /** per plate triangle: index into payload.spaces */
  triSpace: Int32Array;
  /** per space drawn on this level: [space index, first vertex, vertex count] */
  ranges: [number, number, number][];
  wallMaterials: THREE.MeshLambertMaterial[];
}

const WALL_COLOR: Record<string, number> = { surveyed: 0xd7dee8, stamped: 0xd7dee8, template: 0x8793a3 };
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

/** Direction from the model toward the camera: a three-quarter view from above the front-left corner. */
const VIEW_DIR = new THREE.Vector3(-0.5, 0.62, 0.86).normalize();

/**
 * Put the camera where the whole box is in frame with a small margin, whatever the canvas
 * aspect ratio and however long, flat or tall the box is. The distance is found by bisection
 * on the projected corners, so nothing is cropped and nothing is left tiny in a corner.
 */
function fitCamera(camera: THREE.PerspectiveCamera, controls: OrbitControls, box: THREE.Box3, dir: THREE.Vector3 = VIEW_DIR) {
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const reach = Math.max(size.length(), 1);
  const corners: THREE.Vector3[] = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
  const margin = 0.9;
  const place = (d: number) => {
    camera.position.copy(centre).addScaledVector(dir, d);
    camera.near = Math.max(d / 500, 0.05);
    camera.far = d * 40 + reach * 4;
    camera.lookAt(centre);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  };
  let lo = reach * 0.05;
  let hi = reach * 40;
  for (let i = 0; i < 28; i++) {
    const mid = (lo + hi) / 2;
    place(mid);
    const fits = corners.every((c) => {
      const p = c.clone().project(camera);
      return Math.abs(p.x) <= margin && Math.abs(p.y) <= margin && p.z < 1;
    });
    if (fits) hi = mid;
    else lo = mid;
  }
  place(hi);
  controls.target.copy(centre);
  controls.update();
}

function buildLevels(payload: Model3DPayload): LevelMeshes[] {
  const parents = new Set(payload.spaces.map((s) => s.par).filter((p): p is string => !!p));
  const out: LevelMeshes[] = [];
  for (const lv of payload.levels) {
    const group = new THREE.Group();
    // ---- floor plates: leaf spaces only (a unit with child rooms is drawn as its rooms)
    const pos: number[] = [];
    const tri: number[] = [];
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
        }
        tri.push(si);
      }
      ranges.push([si, start, pos.length / 3 - start]);
    });
    const plateGeo = new THREE.BufferGeometry();
    plateGeo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    plateGeo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
    plateGeo.computeVertexNormals();
    const plates = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    group.add(plates);

    // ---- walls (pre-cut panels) grouped by fidelity, and window glass
    const byFidelity = new Map<string, number[]>();
    const glass: number[] = [];
    for (const w of payload.walls) {
      if (w.l !== lv.i) continue;
      const key = w.f === "template" || w.f === "roster" ? "template" : "surveyed";
      let arr = byFidelity.get(key);
      if (!arr) byFidelity.set(key, (arr = []));
      const th = Math.max(0.06, w.th || 0.12);
      for (const [t0, t1, z0, z1] of w.p) pushBox(arr, w.a, w.b, t0, t1, z0, z1, th);
      for (const [t0, t1, z0, z1] of w.g) pushBox(glass, w.a, w.b, t0, t1, z0, z1, 0.04);
    }
    const wallMaterials: THREE.MeshLambertMaterial[] = [];
    for (const [key, arr] of byFidelity) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
      geo.computeVertexNormals();
      const mat = new THREE.MeshLambertMaterial({ color: WALL_COLOR[key] ?? 0x8793a3, transparent: true, opacity: 1 });
      mat.userData.base = key === "template" ? 0.62 : 0.95;
      wallMaterials.push(mat);
      group.add(new THREE.Mesh(geo, mat));
    }
    if (glass.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(glass, 3));
      geo.computeVertexNormals();
      group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x7fc4ff, transparent: true, opacity: 0.35, depthWrite: false })));
    }
    out.push({ index: lv.i, group, plates, triSpace: Int32Array.from(tri), ranges, wallMaterials });
  }
  return out;
}

/**
 * The building model in 3D, coloured by the posterior. The geometry comes from the backend's
 * unified model: surveyed walls with their door and window openings where a scan measured
 * them, a generated layout elsewhere. Click a room to select it.
 */
export function Building3D({ building, posterior, plan, floor, onSelectFloor, onSelectRoom }: Props) {
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
  /** which levels the view currently shows, kept for the resize handler and the reset button */
  const visibleRef = useRef<(index: number) => boolean>(() => true);
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

  // Scene: built once per payload.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !payload) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(0.6, 1, 0.4);
    scene.add(sun);
    const b = payload.bounds;
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1, 8);
    const camera = new THREE.PerspectiveCamera(38, 1, span / 200, span * 20);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    let userMoved = false;
    controls.addEventListener("start", () => {
      userMoved = true;
    });
    // extent of each level in scene coordinates, so the camera can frame just the levels on show
    const levelBox = new Map<number, THREE.Box3>();
    for (const lv of payload.levels) {
      const box = new THREE.Box3();
      for (const s of payload.spaces) {
        if (s.l !== lv.i) continue;
        for (const p of s.p) {
          box.expandByPoint(new THREE.Vector3(...v3(p[0], p[1], lv.z0)));
          box.expandByPoint(new THREE.Vector3(...v3(p[0], p[1], lv.z0 + lv.h)));
        }
      }
      if (!box.isEmpty()) levelBox.set(lv.i, box);
    }
    const fit = (visible: (index: number) => boolean, force = false) => {
      if (userMoved && !force) return;
      if (force) userMoved = false;
      const box = new THREE.Box3();
      for (const [index, lb] of levelBox) if (visible(index)) box.union(lb);
      if (box.isEmpty()) for (const lb of levelBox.values()) box.union(lb);
      if (box.isEmpty()) return;
      fitCamera(camera, controls, box);
      render();
    };
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(span * 4, span * 4),
      new THREE.MeshBasicMaterial({ color: 0x0b1118 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.05, -cy);
    scene.add(ground);
    const levels = buildLevels(payload);
    for (const lv of levels) scene.add(lv.group);
    const marker = new THREE.Mesh(
      new THREE.ConeGeometry(Math.max(0.5, span / 70), Math.max(1.4, span / 28), 16),
      new THREE.MeshBasicMaterial({ color: 0xffa62b }),
    );
    marker.rotation.x = Math.PI; // point down at the room
    marker.visible = false;
    scene.add(marker);

    let queued = false;
    const render = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        renderer.render(scene, camera);
      });
    };
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      fit(visibleRef.current); // a panel that changes shape re-frames the model (until the camera is moved by hand)
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    controls.addEventListener("change", render);
    resize();
    three.current = { renderer, scene, camera, controls, levels, marker, render, fit };
    mount.dataset.ready = "true";
    return () => {
      three.current = null;
      observer.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
        else mat?.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
      delete mount.dataset.ready;
    };
  }, [payload]);

  // Which floors are visible, and how solid their walls are.
  const levelNumbers = useMemo(() => (payload ? payload.levels.map((l) => l.i) : []), [payload]);
  const shown = levelNumbers.includes(floor) ? floor : (levelNumbers[0] ?? 1);
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    const visible = (index: number) => (mode === "all" ? true : mode === "floor" ? index === shown : index <= shown);
    visibleRef.current = visible;
    for (const lv of t.levels) {
      lv.group.visible = visible(lv.index);
      const focus = lv.index === shown;
      for (const m of lv.wallMaterials) {
        const base = m.userData.base as number;
        m.opacity = focus ? base : mode === "all" ? 0.16 : base * 0.55;
        m.depthWrite = focus;
      }
    }
    t.fit(visible);
    t.render();
  }, [payload, mode, shown]);

  // Colour the plates by probability and put the marker over the next stop.
  const nextStopId = plan.teams[0]?.[0]?.room_id ?? null;
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
    const stop = nextStopId ? rooms.get(nextStopId) : undefined;
    const level = stop ? payload.levels.find((l) => l.i === stop.floor) : undefined;
    t.marker.visible = !!stop && !!level;
    if (stop && level) {
      const h = (t.marker.geometry as THREE.ConeGeometry).parameters.height;
      // stand the marker on the room: over a scan's own room when the ids were mapped, else the app room
      const target = payload.spaces.find((s) => s.r === stop.id && !s.par);
      const c = target ? centroid(target.p) : [stop.centroid[0], stop.centroid[1]];
      t.marker.position.set(c[0]!, level.z0 + level.h * 0.5 + h / 2, -c[1]!);
    }
    t.render();
  }, [payload, posterior, nextStopId, rooms]);

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
    return {
      roomId: s.r,
      label: `${room?.name ?? s.n}${part} - floor ${room?.floor_label ?? s.l}${p !== undefined ? ` - P ${pct(p)}` : ""} (${FIDELITY_LABEL[s.f] ?? s.f})`,
    };
  }

  const fidelity = payload?.counts.fidelity ?? {};
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
            title="Frame the whole model again"
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
      bodyClassName="flex flex-col"
    >
      <div className="relative h-[440px] w-full bg-bg">
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
        {!payload ? (
          <p className="absolute inset-0 flex items-center justify-center text-xs text-ink-mute">
            {error ? `3D model unavailable: ${error}` : "Building the 3D model…"}
          </p>
        ) : null}
        {hover ? (
          <div
            className="pointer-events-none absolute z-10 max-w-[320px] rounded border border-line-strong bg-raised px-2 py-1 text-[11px] text-ink shadow-lg"
            style={{ left: Math.min(hover.x + 12, 520), top: hover.y + 12 }}
          >
            {hover.text}
          </div>
        ) : null}
      </div>
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

function centroid(points: [number, number][]): [number, number] {
  let x = 0;
  let y = 0;
  for (const p of points) {
    x += p[0];
    y += p[1];
  }
  return [x / points.length, y / points.length];
}

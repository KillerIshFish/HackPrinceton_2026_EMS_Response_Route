"""A ``Building`` from ONE Matterport scan that covers the whole building.

Used by ``ingest.real_building`` for config entries with ``kind: full_scan``.
Unlike a building assembled from suite scans, nothing here is schematic:

* **Rooms** are the scan's rooms with their real outlines (wall vertices), on
  the scan's own floors. Building-local coordinates ARE the scan's model
  coordinates in metres, so sweeps and Matterport's plan images line up with
  the rooms without any transform.
* **Doors** are the scan's doorway openings: an opening that two rooms share is
  a door between them; an opening only one ground-floor room has is an exterior
  door, i.e. an entrance.
* **Stairs** are found from sweeps: a sweep partway up between two floors is on
  a stair, and links its room to the room of the nearest sweep on the floor above.
* **Windows** are the scan's window openings. A window on the building's outer
  wall gets a view from the side it is on (``site.facing_by_side`` in the
  config, read off Vexcel imagery).
* **Position on the ground** comes from building corners matched between the
  scan and Vexcel imagery (``site.control_points``).

What is still judgement: the mapping from Matterport's room classes and labels
to this project's room types (``ROOM_TYPE_RULES``), and the per-side view labels.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import networkx as nx

from core.schema import Building, Room
from ingest import costar, matterport_api
from ingest.georegister import latlng_to_local, local_to_latlng, register_matterport_model
from ingest.matterport import polygon_area, polygon_centroid

MIN_ROOM_M2 = 1.5  # tuned heuristic, same cut as ingest.mp3d.model3d: smaller rings are slivers or shafts
# Tuned heuristic: an "office" (or unlabelled room) of 60 m2 or more is too big for one person, so call it open office.
OPEN_OFFICE_MIN_M2 = 60.0
# Tuned heuristics for finding stairs from sweep positions. 0.6 m is well above the scatter of camera heights on a
# level floor, yet below one stair flight; 3 m chains successive scans up one flight; 5 m is the furthest a flight's
# top or foot may sit from the sweep that links it to a room.
STAIR_SWEEP_RISE_M = 0.6      # a sweep this far above its floor is on a stair
STAIR_FLIGHT_GAP_M = 3.0      # stair sweeps closer than this belong to the same flight
STAIR_LINK_MAX_M = 5.0
# Assumed: walking length of a stair edge. Shorter than the 8 m used by ingest.matterport/ingest.real_building.
STAIR_RUN_M = 6.0
NEAR_RADIUS_M = 8.0  # tuned heuristic, same as configs/demo_buildings.yaml near_radius_m
# Tuned heuristic: an opening within 1.2 m of the ground floor's bounding box is on an outer wall. Covers wall
# thickness plus small facade jogs.
EXTERIOR_WALL_TOLERANCE_M = 1.2
# Assumed: a curb-side staging spot is placed 7 m straight out from an exterior door (past a sidewalk).
STAGING_OFFSET_M = 7.0

# (words to look for in the room's label / class / tags, room_type). First match wins;
# the label typed by whoever made the scan is checked before Matterport's own class.
ROOM_TYPE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("conference", "meeting", "board"), "conference"),
    (("warehouse", "stock", "loading"), "warehouse"),
    (("roof",), "rooftop"),
    (("control room", "server", "it room"), "server_room"),
    (("common", "lounge", "living"), "common_area"),
    (("kitchen", "break", "dining"), "break_room"),
    (("bath", "restroom", "toilet"), "restroom"),
    (("stair",), "stairwell"),
    (("elevator",), "elevator"),
    (("hall", "corridor"), "corridor"),
    (("entrance", "lobby", "reception", "foyer"), "lobby"),
    (("closet", "storage", "laundry"), "storage"),
    (("utility", "mechanical", "garage"), "mechanical"),
    (("office", "bedroom"), "office"),
]
TYPE_NAME = {
    "office": "Office", "open_office": "Open office", "conference": "Conference room", "warehouse": "Warehouse",
    "rooftop": "Rooftop", "server_room": "Control room", "common_area": "Common space", "break_room": "Kitchen",
    "restroom": "Restroom", "stairwell": "Stairs", "elevator": "Elevator", "corridor": "Hallway",
    "lobby": "Entrance", "storage": "Storage", "mechanical": "Utility room", "room": "Room",
}


# ---------------------------------------------------------------------------
# Scan geometry
# ---------------------------------------------------------------------------


def room_ring(room: dict) -> list[tuple[float, float]]:
    """The room's outline as an ordered ring of (x, y), from its boundary edges.

    Dangling wall stubs (vertices with one edge) are pruned first; ``[]`` when no
    closed outline is left.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    pos: dict[str, tuple[float, float]] = {}
    for e in (room.get("boundary") or {}).get("edges") or []:
        vs = e.get("vertices") or []
        if len(vs) != 2 or vs[0]["id"] == vs[1]["id"]:
            continue
        a, b = vs[0]["id"], vs[1]["id"]
        pos[a] = (vs[0]["position"]["x"], vs[0]["position"]["y"])
        pos[b] = (vs[1]["position"]["x"], vs[1]["position"]["y"])
        adj[a].add(b)
        adj[b].add(a)
    changed = True
    while changed:
        changed = False
        for v in [v for v, ns in adj.items() if len(ns) <= 1]:
            for n in adj.pop(v):
                adj[n].discard(v)
            changed = True
    if len(adj) < 3:
        return []
    best: list[str] = []
    seen: set[str] = set()
    for start in adj:
        if start in seen:
            continue
        ring, prev, cur = [start], None, start
        while True:
            seen.add(cur)
            nxt = [n for n in adj[cur] if n != prev and (n == start or n not in ring)]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            ring.append(cur)
        if len(ring) > len(best):
            best = ring
    return [(round(pos[v][0], 3), round(pos[v][1], 3)) for v in best] if len(best) >= 3 else []


def scan_room_type(room: dict, area_m2: float) -> str:
    label = (room.get("label") or "").lower()
    classes = [c for c in (room.get("classifications") or []) if c.get("label")]
    cls = (classes[0]["label"] if classes else "").lower()
    tags = " ".join(str(t).lower() for t in (room.get("tags") or []) if t)
    for text in (label, cls, tags):
        if not text or text in ("other", "void"):
            continue
        for words, rtype in ROOM_TYPE_RULES:
            if any(w in text for w in words):
                if rtype == "office" and area_m2 >= OPEN_OFFICE_MIN_M2:
                    return "open_office"
                return rtype
    return "open_office" if area_m2 >= OPEN_OFFICE_MIN_M2 else "room"


def _floor_level(zs: list[float]) -> float:
    """Finished-floor height from sweep heights: sweeps on stairs sit above it, so use the low cluster."""
    low = min(zs)
    # Sweeps within 0.5 m of the lowest are on the level floor (tripod heights vary a little); median of those.
    near = sorted(z for z in zs if z - low <= 0.5)
    return near[len(near) // 2]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_scan_building(cfg: dict, refresh: bool = False) -> tuple[Building, dict]:
    """``(Building, meta)`` for a ``kind: full_scan`` entry of ``configs/real_buildings.yaml``."""
    model_id = str(cfg["matterport_model"])
    model = matterport_api.fetch_model(model_id, refresh=refresh)
    plans = matterport_api.fetch_floorplans(model_id, refresh=refresh)
    prop = costar.load_property(int(cfg["costar_property_id"]), refresh=refresh) if cfg.get("costar_property_id") else {}
    site = cfg.get("site") or {}

    scan_floors = sorted(model.get("floors") or [], key=lambda f: f["sequence"])
    floor_no = {f["id"]: i + 1 for i, f in enumerate(scan_floors)}
    floor_label = {}
    for f in scan_floors:
        digits = "".join(ch for ch in (f.get("label") or "") if ch.isdigit())
        floor_label[f["id"]] = digits or str(floor_no[f["id"]])

    sweeps = [s for s in (model.get("locations") or []) if s.get("position") and (s.get("floor") or {}).get("id") in floor_no]
    z_by_floor: dict[str, list[float]] = defaultdict(list)
    for s in sweeps:
        z_by_floor[s["floor"]["id"]].append(s["position"]["z"])
    levels = {fid: _floor_level(zs) for fid, zs in z_by_floor.items()}
    base = min(levels.values()) if levels else 0.0
    floors = [floor_no[f["id"]] for f in scan_floors]
    floor_heights = {
        # A floor with no sweeps falls back to an assumed 3 m per storey above the lowest floor.
        floor_no[f["id"]]: round(levels.get(f["id"], base + 3.0 * i) - base, 2) for i, f in enumerate(scan_floors)
    }

    # ---- rooms ------------------------------------------------------------
    sweeps_by_room: dict[str, list[dict]] = defaultdict(list)
    for s in sweeps:
        rid = (s.get("room") or {}).get("id")
        if rid:
            sweeps_by_room[rid].append(s)

    kept: dict[str, dict] = {}  # scan room id -> working record
    for r in model.get("rooms") or []:
        fid = (r.get("floor") or {}).get("id")
        ring = room_ring(r)
        classes = [c.get("label") for c in (r.get("classifications") or [])]
        if fid not in floor_no or len(ring) < 3 or (classes[:1] == ["Void"] and not r.get("label")):
            continue
        area = float((r.get("dimensions") or {}).get("areaFloor") or 0.0) or polygon_area(ring)
        if area < MIN_ROOM_M2:
            continue
        kept[r["id"]] = {"scan": r, "ring": ring, "area": area, "floor": floor_no[fid],
                         "type": scan_room_type(r, area), "centroid": polygon_centroid(ring)}

    # Names: the scan's own label when it has one, else "<Type> <n>" counted west to east per floor.
    counters: dict[tuple[int, str], int] = defaultdict(int)
    totals: dict[tuple[int, str], int] = defaultdict(int)
    for rec in kept.values():
        if not (rec["scan"].get("label") or "").strip():
            totals[(rec["floor"], rec["type"])] += 1
    for rec in sorted(kept.values(), key=lambda k: (k["floor"], k["centroid"][0], k["centroid"][1])):
        label = (rec["scan"].get("label") or "").strip()
        if label:
            rec["name"] = label
            continue
        key = (rec["floor"], rec["type"])
        counters[key] += 1
        base_name = TYPE_NAME.get(rec["type"], rec["type"].replace("_", " ").capitalize())
        rec["name"] = f"{base_name} {counters[key]}" if totals[key] > 1 else base_name
    for i, (sid, rec) in enumerate(sorted(kept.items(), key=lambda kv: (kv[1]["floor"], kv[1]["centroid"][0], kv[1]["centroid"][1]))):
        rec["id"] = f"f{rec['floor']}-r{i + 1:02d}"

    # ---- openings: doors, exterior doors, windows -------------------------
    ground = min(floors)
    xs = [p[0] for rec in kept.values() if rec["floor"] == ground for p in rec["ring"]]
    ys = [p[1] for rec in kept.values() if rec["floor"] == ground for p in rec["ring"]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    def outer_side(x: float, y: float) -> Optional[str]:
        """Which outer wall of the building a point is on: "+x" | "-x" | "+y" | "-y" | None."""
        dists = {"-x": abs(x - xmin), "+x": abs(x - xmax), "-y": abs(y - ymin), "+y": abs(y - ymax)}
        side = min(dists, key=dists.get)
        return side if dists[side] <= EXTERIOR_WALL_TOLERANCE_M else None

    openings: dict[str, dict] = {}
    shared_edges: dict[str, set[str]] = defaultdict(set)
    for sid, rec in kept.items():
        for e in (rec["scan"].get("boundary") or {}).get("edges") or []:
            vs = e.get("vertices") or []
            if len(vs) != 2:
                continue
            a = (vs[0]["position"]["x"], vs[0]["position"]["y"])
            b = (vs[1]["position"]["x"], vs[1]["position"]["y"])
            if e.get("type") == "invisible":  # no wall here: the two rooms are open to each other
                shared_edges[e["id"]].add(sid)
                openings.setdefault("edge:" + e["id"], {"type": "open", "xy": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                                                         "width": math.hypot(b[0] - a[0], b[1] - a[1]), "rooms": set()})["rooms"].add(sid)
            for o in e.get("openings") or []:
                t = float(o.get("relativeCenter") or 0.5)
                entry = openings.setdefault(o["id"], {
                    "type": o.get("type"), "xy": (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t),
                    "width": float(o.get("width") or 0.0), "rooms": set(),
                })
                entry["rooms"].add(sid)

    facing_by_side: dict[str, str] = dict(site.get("facing_by_side") or {})
    window_width: dict[str, dict[Optional[str], float]] = defaultdict(lambda: defaultdict(float))
    for o in openings.values():
        if o["type"] != "window":
            continue
        side = outer_side(*o["xy"]) if len(o["rooms"]) == 1 else None
        for sid in o["rooms"]:
            # Floor at 10 cm so a window with no recorded width still counts toward its side.
            window_width[sid][facing_by_side.get(side) if side else None] += max(o["width"], 0.1)

    rooms: list[Room] = []
    by_id: dict[str, Room] = {}
    room_meta: dict[str, dict] = {}
    for sid, rec in sorted(kept.items(), key=lambda kv: kv[1]["id"]):
        views = window_width.get(sid) or {}
        outward = {k: v for k, v in views.items() if k}
        mine = sweeps_by_room.get(sid) or []
        sweep_id = None
        if mine:
            cx, cy = rec["centroid"]
            level = levels.get(rec["scan"]["floor"]["id"], 0.0)
            on_floor = [s for s in mine if s["position"]["z"] - level < STAIR_SWEEP_RISE_M] or mine
            sweep_id = min(on_floor, key=lambda s: (s["position"]["x"] - cx) ** 2 + (s["position"]["y"] - cy) ** 2)["id"]
        room = Room(
            id=rec["id"], name=rec["name"], floor=rec["floor"], floor_label=floor_label[rec["scan"]["floor"]["id"]],
            polygon=rec["ring"],
            centroid=(round(rec["centroid"][0], 3), round(rec["centroid"][1], 3), floor_heights[rec["floor"]]),
            area_m2=round(rec["area"], 2), room_type=rec["type"],
            has_windows=bool(views), window_facing=max(outward, key=outward.get) if outward else None,
            near_elevator=False, near_stairs=False,
            occupancy_factor=1.0,
        )
        rooms.append(room)
        by_id[room.id] = room
        classes = [c.get("label") for c in (rec["scan"].get("classifications") or [])]
        room_meta[room.id] = {
            "source": "matterport", "model_id": model_id, "sweep_id": sweep_id, "scan_room_id": sid,
            "scan_class": classes[0] if classes else None, "scan_label": rec["scan"].get("label"),
            "windows_source": "scan",
        }

    # ---- edges ------------------------------------------------------------
    edges: list[dict] = []
    edge_keys: set[tuple[str, str]] = set()

    def add_edge(u: str, v: str, kind: str, length_m: float, floors_delta: int = 0, extra: Optional[dict] = None) -> bool:
        key = (u, v) if u < v else (v, u)
        if u == v or key in edge_keys:
            return False
        edge_keys.add(key)
        edges.append({"u": u, "v": v, "kind": kind, "length_m": round(float(length_m), 2), "floors": int(floors_delta),
                      **(extra or {})})
        return True

    def via(u: Room, v: Room, xy: tuple[float, float]) -> float:
        return math.hypot(u.centroid[0] - xy[0], u.centroid[1] - xy[1]) + math.hypot(v.centroid[0] - xy[0], v.centroid[1] - xy[1])

    exterior_doors: list[dict] = []
    for o in openings.values():
        if o["type"] == "window":
            continue
        members = sorted(kept[sid]["id"] for sid in o["rooms"])
        if len(members) == 2:
            u, v = by_id[members[0]], by_id[members[1]]
            kind = "corridor" if u.room_type == "corridor" and v.room_type == "corridor" else "door"
            add_edge(u.id, v.id, kind, via(u, v, o["xy"]), extra={"xy": [round(o["xy"][0], 2), round(o["xy"][1], 2)]})
        elif len(members) == 1 and o["type"] == "doorway" and by_id[members[0]].floor == ground:
            side = outer_side(*o["xy"])
            if side:
                exterior_doors.append({"room": by_id[members[0]], "xy": o["xy"], "side": side})

    # Stairs: scan positions partway up a flight (well above their floor's level) are grouped
    # into flights; each flight links the room at its foot to the room it arrives in.
    sweeps_on = {fid: [s for s in sweeps if s["floor"]["id"] == fid] for fid in floor_no}

    def xyz(sweep: dict) -> tuple[float, float, float]:
        return sweep["position"]["x"], sweep["position"]["y"], sweep["position"]["z"]

    def unit(a: tuple, b: tuple) -> Optional[list[float]]:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        return [round(dx / n, 3), round(dy / n, 3)] if n >= 0.3 else None  # under 30 cm the direction is noise

    stairs_found = 0
    for lower, upper in zip(scan_floors, scan_floors[1:]):
        lf, uf = lower["id"], upper["id"]
        if lf not in levels or uf not in levels:
            continue
        rise = levels[uf] - levels[lf]
        # On a stair: higher than STAIR_SWEEP_RISE_M, or 40% of the storey rise when storeys are short. At the foot:
        # within 0.3 m of the floor level.
        on_stairs = [s for s in sweeps_on[lf] if (s.get("room") or {}).get("id") in kept
                     and s["position"]["z"] - levels[lf] >= min(STAIR_SWEEP_RISE_M, 0.4 * rise)]
        landings = [t for t in sweeps_on[uf] if (t.get("room") or {}).get("id") in kept]
        foot = [b for b in sweeps_on[lf] if (b.get("room") or {}).get("id") in kept and b["position"]["z"] - levels[lf] < 0.3]
        # flights: stair positions chained by proximity
        flights: list[list[dict]] = []
        for s in sorted(on_stairs, key=lambda s: s["position"]["z"]):
            for flight in flights:
                if any(math.dist(xyz(s), xyz(o)) <= STAIR_FLIGHT_GAP_M for o in flight):
                    flight.append(s)
                    break
            else:
                flights.append([s])
        for flight in flights:
            s_hi, s_lo = flight[-1], flight[0]  # sorted by height
            top = min(landings, key=lambda t: math.dist(xyz(s_hi), xyz(t)), default=None)
            if top is None or math.dist(xyz(s_hi), xyz(top)) > STAIR_LINK_MAX_M:
                continue
            # Where the flight arrives and which way the walker is heading there (used by route
            # clues such as "I turned right at the top of the stairs"); the same at its foot.
            geometry: dict = {"top_xy": [round(v, 2) for v in xyz(top)[:2]]}
            if (h := unit(xyz(s_hi), xyz(top))):
                geometry["heading"] = h
            bottom = min(foot, key=lambda b: math.dist(xyz(s_lo)[:2], xyz(b)[:2]), default=None)
            if bottom is not None and math.dist(xyz(s_lo)[:2], xyz(bottom)[:2]) <= STAIR_LINK_MAX_M:
                # going down, the walker's heading is the flight's own direction at its lowest step
                if (h := (unit(xyz(flight[1]), xyz(s_lo)) if len(flight) > 1 else None) or unit(xyz(s_lo), xyz(bottom))):
                    geometry["bottom_xy"] = [round(v, 2) for v in xyz(bottom)[:2]]
                    geometry["bottom_heading"] = h
            u = kept[(bottom or s_lo)["room"]["id"]]["id"] if bottom is not None else kept[s_lo["room"]["id"]]["id"]
            v = kept[top["room"]["id"]]["id"]
            stairs_found += add_edge(u, v, "stairs", STAIR_RUN_M, 1, extra=geometry)

    # Anything still cut off gets one inferred link to the closest room, so every room is reachable.
    inferred_links: list[str] = []

    def graph() -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(by_id)
        g.add_edges_from((e["u"], e["v"]) for e in edges)
        return g

    g = graph()
    main = max(nx.connected_components(g), key=lambda c: sum(by_id[r].area_m2 for r in c))
    while True:
        others = [c for c in nx.connected_components(g) if not (c & main)]
        if not others:
            break
        comp = min(others, key=len)
        best = None
        for a in comp:
            for b in main:
                ra, rb = by_id[a], by_id[b]
                # Each floor of separation costs 6 m (one STAIR_RUN_M): a same-floor link wins unless much longer.
                d = math.hypot(ra.centroid[0] - rb.centroid[0], ra.centroid[1] - rb.centroid[1]) + 6.0 * abs(ra.floor - rb.floor)
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        ra, rb = by_id[a], by_id[b]
        if ra.floor != rb.floor:
            add_edge(a, b, "stairs", STAIR_RUN_M, abs(ra.floor - rb.floor))
        else:
            add_edge(a, b, "door", math.hypot(ra.centroid[0] - rb.centroid[0], ra.centroid[1] - rb.centroid[1]))
        inferred_links.append(f"{ra.name} (F{ra.floor_label}) - {rb.name} (F{rb.floor_label})")
        g = graph()
        main = max(nx.connected_components(g), key=lambda c: sum(by_id[r].area_m2 for r in c))

    # near_stairs: stairwell rooms, rooms a stair edge touches, and rooms next to either.
    stair_rooms = {r.id for r in rooms if r.room_type == "stairwell"} | {
        e[k] for e in edges if e["kind"] == "stairs" for k in ("u", "v")
    }
    for room in rooms:
        room.near_stairs = room.id in stair_rooms or any(
            by_id[s].floor == room.floor
            and math.hypot(room.centroid[0] - by_id[s].centroid[0], room.centroid[1] - by_id[s].centroid[1]) <= NEAR_RADIUS_M
            for s in stair_rooms
        )
        room.near_elevator = room.room_type == "elevator"

    # ---- position on the ground, entrances, staging -----------------------
    control = site.get("control_points") or []
    if len(control) >= 2:
        georef = register_matterport_model(control)
    else:  # fall back to the scan's own (phone-grade) geolocation
        geo = model.get("geocoordinates") or {}
        q = geo.get("rotation") or {}
        georef = {"origin_latlng": [geo.get("latitude"), geo.get("longitude")],
                  "rotation_deg": round(math.degrees(2 * math.atan2(q.get("z", 0.0), q.get("w", 1.0))), 2), "rms_m": None}
    origin, rotation = georef["origin_latlng"], georef["rotation_deg"]

    def to_latlng(x: float, y: float) -> list[float]:
        lat, lng = local_to_latlng(x, y, origin, rotation)
        return [round(lat, 7), round(lng, 7)]

    outward = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}
    entrances: list[dict] = []
    staging: list[dict] = []
    exterior_doors.sort(key=lambda d: (facing_by_side.get(d["side"]) != "street", -d["room"].area_m2 if d["room"].room_type != "lobby" else -1e9))
    for i, d in enumerate(exterior_doors, start=1):
        facing = facing_by_side.get(d["side"], "street")
        x, y = d["xy"]
        ent = {"id": f"ent-{i}", "name": f"{d['room'].name} door ({facing.replace('_', ' ')} side)",
               "room_id": d["room"].id, "xy": [round(x, 2), round(y, 2)], "latlng": to_latlng(x, y), "facing": facing}
        entrances.append(ent)
        dx, dy = outward[d["side"]]
        sx, sy = x + dx * STAGING_OFFSET_M, y + dy * STAGING_OFFSET_M
        # one staging spot per stretch of frontage: skip a door within 10 m of an existing spot (tuned heuristic)
        if all(math.hypot(sx - s["xy"][0], sy - s["xy"][1]) > 10.0 for s in staging):
            staging.append({"id": f"stage-{len(staging) + 1}", "name": f"Curb outside the {d['room'].name.lower()} door",
                            "xy": [round(sx, 2), round(sy, 2)], "latlng": to_latlng(sx, sy),
                            "kind": "street" if facing == "street" else "parking_lot", "near_entrance_id": ent["id"]})
    for s in site.get("staging_spots") or []:  # hand-placed spots from the config
        lx, ly = latlng_to_local(s["latlng"][0], s["latlng"][1], origin, rotation)
        near = min(entrances, key=lambda e: math.hypot(e["xy"][0] - lx, e["xy"][1] - ly))["id"] if entrances else ""
        staging.append({"id": s["id"], "name": s["name"], "xy": [round(lx, 2), round(ly, 2)],
                        "latlng": [float(s["latlng"][0]), float(s["latlng"][1])], "kind": s.get("kind", "street"),
                        "near_entrance_id": s.get("near_entrance_id") or near})

    building = Building(
        id=cfg["id"], name=cfg.get("name") or prop.get("name") or model.get("name") or cfg["id"],
        use_type=cfg.get("use_type", "office"), floors=floors, floor_heights_m=floor_heights,
        rooms=rooms, edges=edges, entrances=entrances, staging_spots=staging,
    )

    footprint = [to_latlng(x, y) for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))]
    center = to_latlng((xmin + xmax) / 2, (ymin + ymax) / 2)
    seq_of = {f["id"]: floor_no[f["id"]] for f in scan_floors}
    meta = {
        "building_id": building.id,
        "name": building.name,
        "address": prop.get("address") or (model.get("address") or {}).get("address"),
        "real": True,
        "kind": "full_scan",
        "center_latlng": center,
        "georeference": {"origin_latlng": origin, "rotation_deg": rotation, "rms_m": georef.get("rms_m"),
                         "frame": "matterport model coordinates"},
        "site": {
            "source": site.get("source"),
            # Default view when the config has none: Vexcel's urban layer at zoom 21 (its finest; see ingest.vexcel
            # MAX_ZOOM), 45 m around the centre, close to the 42 m set for the full-scan entry in real_buildings.yaml.
            "imagery": site.get("imagery") or {"layer": "urban", "zoom": 21, "radius_m": 45},
            "footprint_latlng": footprint,
        },
        "costar": {k: prop.get(k) for k in ("property_id", "stories", "rba_sf", "typical_floor_sf", "parking_spaces")},
        "matterport_models": [{
            "model_id": model_id, "suite": None, "name": (model.get("name") or "").strip(), "floor": None,
            "created": model.get("created"), "sweeps": len(sweeps), "has_room_boundaries": True,
            "dollhouse_enabled": (model.get("options") or {}).get("dollhouseEnabled"),
            "floorplan_enabled": (model.get("options") or {}).get("floorplanEnabled"),
        }],
        # Matterport's plan image per floor, in the same coordinates as the rooms.
        "floorplans": {
            str(seq_of[fid]): {"file": p["file"], "width": p["width"], "height": p["height"],
                               "resolution": p["resolution"], "origin": p["origin"]}
            for fid, p in plans.items() if fid in seq_of
        },
        "room_sources": {"matterport": len(rooms)},
        "rooms": room_meta,
        "stairs_from_sweeps": stairs_found,
        "inferred_links": inferred_links,
        "caveats": [
            "Rooms, doors, windows and stairs come from one Matterport scan of the whole building.",
            "Room types are mapped from the scan's labels and Matterport's room classes; unlabelled rooms are 'room'.",
            "The building is placed on the map from three corners matched to Vexcel imagery"
            + (f" (fit residual {georef['rms_m']} m)." if georef.get("rms_m") is not None else "."),
        ] + ([f"{len(inferred_links)} room link(s) had no door or stair in the scan and were inferred: "
              + "; ".join(inferred_links) + "."] if inferred_links else []),
    }
    return building, meta

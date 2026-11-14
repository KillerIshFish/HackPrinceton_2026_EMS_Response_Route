"""``BuildingModel`` -> what the app consumes.

* ``to_building``: the frozen ``core.schema.Building`` used by the posterior, the
  search plan and the 2D floor plan. Only top-level spaces become rooms: a unit is
  one room however many child rooms its interior has, because the question is
  which unit the caller is in.
* ``viewer_payload``: compact JSON for the 3D viewer, with every wall already cut
  into solid panels around its openings (Track A's ``walls.solid_panels``).

Position on the map and what each side faces come from the optional ``SiteContext``
sidecar. Without it the building still projects; it just has no views and its
entrances carry the property's own coordinates.
"""

from __future__ import annotations

import math
from typing import Optional

from core.schema import Building, Room
from ingest.georegister import local_to_latlng
from ingest.mp3d.walls import solid_panels
from ingest.unified.schema import BuildingModel

ROOM_TYPE_BY_KIND = {"unit": "unit", "tenant": "suite", "floor": "suite", "corridor": "corridor", "stair": "stairwell",
                     "elevator": "elevator", "lobby": "lobby", "walkway": "walkway", "parking": "parking",
                     "amenity": "common_area", "room": "room"}
EDGE_KIND = {"door": "door", "opening": "door", "corridor": "corridor", "walkway": "corridor", "stairs": "stairs",
             "elevator": "elevator"}
NEAR_M = 10.0
STAGING_OFFSET_M = 7.0
FACING_TOLERANCE_DEG = 60.0


def _bounds(bm: BuildingModel) -> tuple[float, float, float, float]:
    xs = [p[0] for s in bm.spaces for p in s.polygon] or [0.0]
    ys = [p[1] for s in bm.spaces for p in s.polygon] or [0.0]
    return min(xs), min(ys), max(xs), max(ys)


def placement(bm: BuildingModel, context: Optional[dict] = None) -> dict:
    """How local metres map to the ground: ``{"origin_latlng", "yaw_deg", "centre", "scale"}``.

    ``scale`` (x, y) stretches a generated layout about its centre so that its extent matches
    the real building's outline box; it is (1, 1) unless the sidecar supplies a fit. It only
    affects where things are drawn on the map: the model's own metres are not changed.
    """
    x0, y0, x1, y1 = _bounds(bm)
    scan = bm.archetype in ("whole_scan", "scan_clusters")
    # the sidecar's anchor (the real building's centre, when OpenStreetMap has its outline)
    # beats the bare address point
    geo = (None if scan else (context or {}).get("anchor_latlng")) or (context or {}).get("geolocation") or bm.geolocation
    yaw = (context or {}).get("yaw_deg")
    return {"origin_latlng": list(geo) if geo else None, "yaw_deg": float(bm.yaw_deg if yaw is None else yaw),
            # a scan's own origin is what its geolocation refers to; a generated building is centred on the address
            "centre": (0.0, 0.0) if scan else ((x0 + x1) / 2, (y0 + y1) / 2),
            "scale": (1.0, 1.0) if scan else tuple((context or {}).get("fit_scale") or (1.0, 1.0))}


def to_latlng(place: dict, x: float, y: float, offset_m: tuple = (0.0, 0.0)) -> list[float]:
    """Model point -> (lat, lng). ``offset_m`` is added afterwards in true metres along the model's
    axes (used for points defined by a real distance, such as a staging spot 7 m outside a door)."""
    if not place["origin_latlng"]:
        return [0.0, 0.0]
    sx, sy = place.get("scale") or (1.0, 1.0)
    lat, lng = local_to_latlng((x - place["centre"][0]) * sx + offset_m[0], (y - place["centre"][1]) * sy + offset_m[1],
                               place["origin_latlng"], place["yaw_deg"])
    return [round(lat, 7), round(lng, 7)]


def _compass(vx: float, vy: float, yaw_deg: float) -> float:
    theta = math.radians(yaw_deg)
    east = vx * math.cos(theta) - vy * math.sin(theta)
    north = vx * math.sin(theta) + vy * math.cos(theta)
    return math.degrees(math.atan2(east, north)) % 360.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def room_ids(bm: BuildingModel) -> dict[str, str]:
    """space id -> id of the room that stands for it in the app (itself, or its parent unit)."""
    return {s.id: (s.parent or s.id) for s in bm.spaces}


def to_building(bm: BuildingModel, context: Optional[dict] = None) -> Building:
    root = room_ids(bm)
    top = [s for s in bm.spaces if s.parent is None]
    by_id = {s.id: s for s in top}
    place = placement(bm, context)
    street = ((context or {}).get("street") or {}).get("bearing_deg")
    multi_block = len({s.block for s in top if s.block}) > 1

    # windows: which rooms have one, and which way the glass faces
    window_dirs: dict[str, list[float]] = {}
    for w in bm.walls:
        if not any(o.kind == "window" for o in w.openings):
            continue
        for sid in w.space_ids:
            rid = root.get(sid)
            if rid not in by_id:
                continue
            c = by_id[rid].centroid
            mx, my = (w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2
            ex, ey = w.b[0] - w.a[0], w.b[1] - w.a[1]
            n = math.hypot(ex, ey) or 1.0
            nx, ny = -ey / n, ex / n
            if (mx + nx - c[0]) ** 2 + (my + ny - c[1]) ** 2 < (mx - nx - c[0]) ** 2 + (my - ny - c[1]) ** 2:
                nx, ny = -nx, -ny
            window_dirs.setdefault(rid, []).append(_compass(nx, ny, place["yaw_deg"]))

    vertical = {"stair": [s for s in top if s.kind == "stair"], "elevator": [s for s in top if s.kind == "elevator"]}
    linked: dict[str, set[str]] = {}
    for ln in bm.links:
        a, b = root.get(ln.a), root.get(ln.b)
        if a in by_id and b in by_id and a != b:
            linked.setdefault(a, set()).add(b)
            linked.setdefault(b, set()).add(a)

    def near(space, kind: str) -> bool:
        if space.kind == kind:
            return True
        return any(v.level == space.level and (math.dist(v.centroid, space.centroid) <= NEAR_M
                                               or v.id in linked.get(space.id, ()))
                   for v in vertical[kind])

    z_of = {lv.index: lv.z0 for lv in bm.levels}
    label_of = {lv.index: lv.label for lv in bm.levels}
    rooms = []
    for s in top:
        dirs = window_dirs.get(s.id) or []
        facing = None
        if dirs and street is not None and any(_gap(d, street) < FACING_TOLERANCE_DEG for d in dirs):
            facing = "street"
        name = f"Unit {s.label}" if s.kind == "unit" else s.label
        if multi_block and s.block and s.kind != "unit":
            name = f"{name} (bldg {s.block})"
        rooms.append(Room(
            id=s.id, name=name, floor=s.level, floor_label=label_of.get(s.level, str(s.level)),
            polygon=[tuple(p) for p in s.polygon], centroid=(s.centroid[0], s.centroid[1], z_of.get(s.level, 0.0)),
            area_m2=max(s.area_m2, 0.01), room_type=s.room_type or ROOM_TYPE_BY_KIND.get(s.kind, "room"),
            has_windows=bool(dirs), window_facing=facing, near_elevator=near(s, "elevator"),
            near_stairs=near(s, "stair"),
            occupancy_factor=0.05 if s.kind == "walkway" else 0.5 if s.kind == "floor" else 1.0,
        ))

    edges, seen = [], set()
    for ln in bm.links:
        a, b = root.get(ln.a), root.get(ln.b)
        if a not in by_id or b not in by_id or a == b:
            continue
        kind = EDGE_KIND.get(ln.kind, "door")
        key = (min(a, b), max(a, b), kind)
        if key in seen:
            continue
        seen.add(key)
        edge = {"u": key[0], "v": key[1], "kind": kind, "length_m": ln.length_m, "floors": ln.floors}
        if ln.xy:
            edge["xy"] = [ln.xy[0], ln.xy[1]]
        edges.append(edge)

    entrances, staging = [], []
    found = list(bm.entrances)
    # where the layout's positions on the map are invented (a multi-building property), every
    # entrance and staging spot carries the address point: the one location that is real
    address = (context or {}).get("geolocation") if (context or {}).get("layout_placed") is False else None

    def ground(x: float, y: float, offset_m: tuple = (0.0, 0.0)) -> list[float]:
        return [round(float(address[0]), 7), round(float(address[1]), 7)] if address else to_latlng(place, x, y, offset_m)

    for i, e in enumerate(found, start=1):
        rid = root.get(e.space_id)
        if rid not in by_id:
            continue
        entrances.append({"id": e.id, "name": e.name, "room_id": rid, "xy": [e.xy[0], e.xy[1]],
                          "latlng": ground(*e.xy), "facing": "street"})
        sx, sy = e.xy[0] + e.outward[0] * STAGING_OFFSET_M, e.xy[1] + e.outward[1] * STAGING_OFFSET_M
        if all(math.dist((sx, sy), s["xy"]) > 10.0 for s in staging):
            staging.append({"id": f"stage-{len(staging) + 1}", "name": f"Outside {e.name[:1].lower()}{e.name[1:]}",
                            "xy": [round(sx, 2), round(sy, 2)],
                            "latlng": ground(e.xy[0], e.xy[1], (e.outward[0] * STAGING_OFFSET_M,
                                                                e.outward[1] * STAGING_OFFSET_M)),
                            "kind": "street", "near_entrance_id": e.id})
    if not entrances and top:
        # no door to the outside in the data: assume the largest room on the lowest level
        lowest = min(s.level for s in top)
        s = max((s for s in top if s.level == lowest), key=lambda s: (s.area_m2, s.id))
        entrances.append({"id": "ent-1", "name": f"{s.label} (assumed entrance)", "room_id": s.id,
                          "xy": [s.centroid[0], s.centroid[1]], "latlng": to_latlng(place, *s.centroid), "facing": "street"})
        staging.append({"id": "stage-1", "name": "Outside the building", "xy": [s.centroid[0], s.centroid[1]],
                        "latlng": to_latlng(place, *s.centroid), "kind": "street", "near_entrance_id": "ent-1"})

    floors = sorted({rm.floor for rm in rooms})
    return Building(id=bm.id, name=bm.name, use_type=bm.use_type if bm.use_type != "other" else "office",
                    floors=floors, floor_heights_m={f: z_of.get(f, 0.0) for f in floors}, rooms=rooms, edges=edges,
                    entrances=entrances, staging_spots=staging)


def viewer_payload(bm: BuildingModel, room_of: Optional[dict[str, str]] = None) -> dict:
    """Everything the browser needs to draw the model: plates, pre-cut wall panels, glass."""
    root = room_ids(bm)
    room_of = room_of or {}
    slab = {lv.index: lv for lv in bm.levels}

    def app_room(space_id: str) -> str:
        rid = root.get(space_id, space_id)
        return room_of.get(rid, rid)

    walls = []
    for w in bm.walls:
        lv = slab.get(w.level)
        if lv is None or w.open_boundary:
            continue
        spans = [(o.t0, o.t1, o.z0, o.z1) for o in w.openings]
        panels = solid_panels(spans, lv.z0, lv.height_m)
        if not panels:
            continue
        walls.append({
            "l": w.level, "a": list(w.a), "b": list(w.b), "th": w.thickness_m, "x": w.exterior, "f": w.fidelity,
            "p": [[round(p.t0, 4), round(p.t1, 4), round(p.z0, 3), round(p.z1, 3)] for p in panels],
            "g": [[o.t0, o.t1, o.z0, o.z1] for o in w.openings if o.kind == "window"],
        })
    x0, y0, x1, y1 = _bounds(bm)
    return {
        "schema": bm.schema, "id": bm.id, "name": bm.name, "tier": bm.tier, "archetype": bm.archetype,
        "notes": list(bm.notes),
        "levels": [{"i": lv.index, "label": lv.label, "z0": lv.z0, "h": lv.height_m, "src": lv.source} for lv in bm.levels],
        "spaces": [{"id": s.id, "l": s.level, "k": s.kind, "n": s.label, "f": s.fidelity, "p": [list(p) for p in s.polygon],
                    "par": s.parent, "r": app_room(s.id)} for s in bm.spaces],
        "walls": walls,
        # o: unit vector pointing out of the building at the door (the viewer faces that side by default);
        # r: the app room the door opens into
        "entrances": [{"id": e.id, "n": e.name, "xy": list(e.xy), "l": slab and min(slab), "o": list(e.outward),
                       "r": app_room(e.space_id)} for e in bm.entrances],
        "bounds": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                   "z1": max((lv.z0 + lv.height_m for lv in bm.levels), default=3.0)},
        "counts": {"spaces": len(bm.spaces), "walls": len(walls),
                   "fidelity": {f: sum(1 for s in bm.spaces if s.fidelity == f)
                                for f in sorted({s.fidelity for s in bm.spaces})}},
    }

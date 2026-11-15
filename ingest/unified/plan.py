"""A building traced by hand from a published floor plan -> ``BuildingModel``.

For a building that has neither a property record nor a Matterport scan the Model API can
read, but does have published floor plans: the plans are traced into
``configs/plan_buildings.yaml`` (one rectangle or polygon per room, in the pixels of each
level's plan image) and this module turns that file into a model.

What comes from the plan: which rooms exist, their outlines, levels and names, which sides
have windows, where the stairs and elevators are. What is derived here, by rule:

* **Doors.** A room gets a door to each corridor or lobby it shares a wall with (the two
  longest shared walls; ``doors:`` in the file overrides this). A small room marked
  ``in: <space>`` opens into that space. A room that touches no corridor opens into the
  neighbour it shares the longest wall with.
* **Stairs and elevators.** Spaces with the same ``shaft`` on different levels are joined.
* **Overlaps.** Where a room is drawn inside a larger space (a meeting room inside a
  department's area), the room is cut out of the larger space.
* **Position on the map.** Control points (plan pixel <-> lat/lng of a building corner) are
  fitted with ``ingest.georegister``; the model keeps its own origin and rotation, like a scan.

Fidelity is ``traced``: real rooms in their real places, to about a metre, not measured.

Local frame: +x is the plan's right, +y the plan's top, metres, origin at ``origin_px`` of
the reference level.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import yaml
from shapely.geometry import LineString, Point as ShPoint, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points

from ingest.georegister import register_matterport_model
from ingest.unified.layout import _Walls, _opening
from ingest.unified.lift import _connect
from ingest.unified.schema import SCHEMA_VERSION, BuildingModel, Entrance, Level, Link, make_space, r, rp

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "plan_buildings.yaml"

ARCHETYPE = "floor_plan"
FIDELITY = "traced"
TOUCH_M = 0.6            # two outlines this close share a wall
MIN_SHARED_M = 0.9       # shortest shared wall that can hold a door
MIN_OVERLAP_M2 = 0.3     # smaller overlaps are tracing noise, not a room inside a room
DOOR_W, DOOR_H = 0.95, 2.05
ENTRANCE_W = 1.8
WINDOW_W, WINDOW_SILL, WINDOW_HEAD = 1.6, 0.9, 2.4
WINDOW_SPACING_M = 3.5
MIN_WINDOW_WALL_M = 2.0
WALL_T = 0.2
STAIR_RUN_M = 6.0        # walking length of one storey of stairs (same as the generated layouts)
MAX_AUTO_DOORS = 2
SIDES = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}
KIND_OF_TYPE = {"corridor": "corridor", "stairwell": "stair", "elevator": "elevator", "lobby": "lobby", "suite": "tenant"}
VERTICAL = {"stair": "stairs", "elevator": "elevator"}


def plan_configs(path: Optional[Path] = None) -> list[dict]:
    target = Path(path or CONFIG_PATH)
    if not target.is_file():
        return []
    return list((yaml.safe_load(target.read_text(encoding="utf-8")) or {}).get("buildings") or [])


def plan_config(building_id: str, path: Optional[Path] = None) -> dict:
    """The entry for ``building_id`` (its id, or its id without the ``plan:`` prefix)."""
    wanted = str(building_id).strip().lower()
    for cfg in plan_configs(path):
        if str(cfg.get("id", "")).lower() == wanted:
            return cfg
    known = ", ".join(str(c.get("id")) for c in plan_configs(path)) or "none"
    raise KeyError(f"no traced floor plan called {building_id!r} in {CONFIG_PATH.name} (known: {known})")


def _ring(poly: Polygon) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in list(orient(poly, 1.0).exterior.coords)[:-1]]


def _largest(geom) -> Optional[Polygon]:
    """The biggest polygon of a geometry (a cut can leave slivers behind)."""
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
    return max(parts, key=lambda g: g.area) if parts else None


def _without_holes(poly: Polygon) -> Polygon:
    """A room cut out of the middle of a space leaves a hole, which a single outline cannot
    carry: join each hole to the outside with a hair-thin slit. If that fails, drop the hole."""
    for _ in range(len(poly.interiors)):
        if not poly.interiors:
            break
        hole = poly.interiors[0]
        a, b = nearest_points(hole, poly.exterior)
        dx, dy = b.x - a.x, b.y - a.y
        n = math.hypot(dx, dy) or 1.0
        slit = LineString([(a.x - dx / n * 0.05, a.y - dy / n * 0.05), (b.x + dx / n * 0.05, b.y + dy / n * 0.05)])
        cut = _largest(poly.difference(slit.buffer(0.02, cap_style=2)))
        if cut is None or len(cut.interiors) >= len(poly.interiors):
            return Polygon(poly.exterior)
        poly = cut
    return poly


def _contact(a: Polygon, b: Polygon) -> Optional[tuple[float, tuple[float, float]]]:
    """``(shared wall length, a point on it)`` for two outlines, or None when they do not touch."""
    if a.distance(b) > TOUCH_M:
        return None
    band = a.buffer(TOUCH_M / 2, join_style=2).intersection(b.buffer(TOUCH_M / 2, join_style=2))
    piece = _largest(band)
    if piece is None:
        return None
    box = piece.minimum_rotated_rectangle
    if box.geom_type != "Polygon":
        return None
    pts = list(box.exterior.coords)
    length = max(math.dist(pts[0], pts[1]), math.dist(pts[1], pts[2]))
    if length < MIN_SHARED_M:
        return None
    c = piece.centroid
    return length, (c.x, c.y)


def build_plan_model(cfg: dict) -> BuildingModel:
    """Pure function of one ``plan_buildings.yaml`` entry."""
    scale = float(cfg["metres_per_px"])
    ox, oy = (float(v) for v in cfg.get("origin_px") or (0.0, 0.0))
    default_h = float(cfg.get("storey_height_m") or 4.0)

    def to_model(frame: dict, px) -> tuple[float, float]:
        s = float(frame.get("scale", 1.0))
        rx, ry = s * float(px[0]) + float(frame.get("dx", 0.0)), s * float(px[1]) + float(frame.get("dy", 0.0))
        return ((rx - ox) * scale, -(ry - oy) * scale)

    # ---- levels and outlines -------------------------------------------------
    levels: list[Level] = []
    recs: list[dict] = []
    index_of: dict[str, int] = {}
    frames: dict[int, dict] = {}
    z = 0.0
    for i, lv in enumerate(cfg.get("levels") or [], start=1):
        h = float(lv.get("height_m") or default_h)
        label = str(lv["label"])
        levels.append(Level(i, label, r(z), r(h), "assumed" if not lv.get("height_m") else "plan"))
        index_of[label] = i
        frames[i] = lv.get("frame") or {}
        for sp in lv.get("spaces") or []:
            if sp.get("box"):
                x0, y0, x1, y1 = sp["box"]
                px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                px = sp["poly"]
            poly = Polygon([to_model(frames[i], p) for p in px]).buffer(0)
            poly = _largest(poly)
            if poly is None or poly.area < 0.5:
                raise ValueError(f"{cfg['id']}: space {sp.get('id')!r} has no area")
            stype = str(sp["type"])
            kind = KIND_OF_TYPE.get(stype, "room")
            recs.append({"id": str(sp["id"]), "level": i, "z0": z, "cfg": sp, "type": stype, "kind": kind,
                         "poly": poly, "drawn": poly,
                         "circ": kind in ("corridor", "lobby") or bool(sp.get("circulation"))})
        z += h
    ids = [rec["id"] for rec in recs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{cfg['id']}: duplicate space ids {dupes}")
    by_id = {rec["id"]: rec for rec in recs}
    on_level: dict[int, list[dict]] = {}
    for rec in recs:
        on_level.setdefault(rec["level"], []).append(rec)

    # ---- a room drawn inside a larger space is cut out of it -----------------
    carved = 0
    for group in on_level.values():
        for big in sorted(group, key=lambda g: -g["drawn"].area):
            for small in group:
                if small is big or small["drawn"].area >= big["drawn"].area:
                    continue
                if big["poly"].intersection(small["drawn"]).area < MIN_OVERLAP_M2:
                    continue
                cut = _largest(big["poly"].difference(small["drawn"]))
                if cut is not None and cut.area > 0.5:
                    big["poly"] = cut
                    big.setdefault("inside", []).append(small["id"])
                    carved += 1
            big["poly"] = _without_holes(big["poly"])

    # ---- who touches whom ----------------------------------------------------
    touch: dict[str, dict[str, tuple[float, tuple[float, float]]]] = {i: {} for i in ids}
    for group in on_level.values():
        for ai, a in enumerate(group):
            for b in group[ai + 1:]:
                hit = _contact(a["poly"], b["poly"])
                if hit:
                    touch[a["id"]][b["id"]] = hit
                    touch[b["id"]][a["id"]] = hit

    # ---- links ---------------------------------------------------------------
    links: dict[tuple[str, str], Link] = {}
    doors: dict[str, list[tuple[tuple[float, float], float, str]]] = {i: [] for i in ids}
    counts = {"corridor": 0, "door": 0, "room_to_room": 0, "vertical": 0}

    def centre(i: str) -> tuple[float, float]:
        c = by_id[i]["poly"].representative_point() if not by_id[i]["poly"].contains(by_id[i]["poly"].centroid) \
            else by_id[i]["poly"].centroid
        return (c.x, c.y)

    def join(a: str, b: str, kind: str, xy=None, width: float = DOOR_W, opening: str = "doorway") -> bool:
        key = (min(a, b), max(a, b))
        if a == b or key in links:
            return False
        if xy is None:  # not drawn touching: the door goes where the two outlines are closest
            pa, pb = nearest_points(by_id[a]["poly"], by_id[b]["poly"])
            xy = ((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
        length = math.dist(centre(a), xy) + math.dist(xy, centre(b))
        links[key] = Link(a=key[0], b=key[1], kind=kind, evidence="plan", length_m=r(max(length, 1.0), 2), xy=rp(xy))
        doors[a].append((xy, width, opening))
        doors[b].append((xy, width, opening))
        return True

    for rec in recs:  # corridors and lobbies run into one another through a wide opening
        if not rec["circ"]:
            continue
        for other, (length, xy) in sorted(touch[rec["id"]].items()):
            if by_id[other]["circ"] and join(rec["id"], other, "corridor", xy, max(1.2, min(length - 0.6, 6.0)), "opening"):
                counts["corridor"] += 1
    for rec in recs:
        if rec["circ"]:
            continue
        sp = rec["cfg"]
        wanted = [str(t) for t in sp.get("doors") or []]
        if sp.get("in"):
            wanted.append(str(sp["in"]))
        for target in wanted:
            if target not in by_id or by_id[target]["level"] != rec["level"]:
                raise ValueError(f"{cfg['id']}: {rec['id']!r} names a door to {target!r}, which is not on its level")
            hit = touch[rec["id"]].get(target)
            if join(rec["id"], target, "door", hit[1] if hit else None):
                counts["door"] += 1
        if wanted:
            continue
        near = sorted(((length, other, xy) for other, (length, xy) in touch[rec["id"]].items() if by_id[other]["circ"]),
                      key=lambda t: (-t[0], t[1]))
        limit = MAX_AUTO_DOORS + (1 if rec["kind"] == "tenant" else 0)
        for length, other, xy in near[:limit]:
            if join(rec["id"], other, "door", xy):
                counts["door"] += 1
    linked = {i for key in links for i in key}
    for rec in recs:  # a room off no corridor opens into the neighbour it shares most wall with
        if rec["id"] in linked or not touch[rec["id"]]:
            continue
        other, (length, xy) = max(touch[rec["id"]].items(), key=lambda kv: (kv[1][0], kv[0]))
        if join(rec["id"], other, "door", xy):
            counts["room_to_room"] += 1
            linked.update((rec["id"], other))

    shafts: dict[str, list[dict]] = {}
    for rec in recs:
        if rec["cfg"].get("shaft") and rec["kind"] in VERTICAL:
            shafts.setdefault(str(rec["cfg"]["shaft"]), []).append(rec)
    for name in sorted(shafts):
        stops = sorted(shafts[name], key=lambda s: s["level"])
        for lo, hi in zip(stops, stops[1:]):
            floors = hi["level"] - lo["level"]
            key = (min(lo["id"], hi["id"]), max(lo["id"], hi["id"]))
            links[key] = Link(a=key[0], b=key[1], kind=VERTICAL[lo["kind"]], evidence="plan",
                              length_m=r(STAIR_RUN_M * floors, 2), floors=floors)
            counts["vertical"] += 1

    # ---- spaces --------------------------------------------------------------
    spaces = []
    for rec in recs:
        sp = rec["cfg"]
        attrs = {k: sp.get(k) for k in ("tenant", "capacity", "owner", "shaft")}
        spaces.append(make_space(
            rec["id"], rec["level"], rec["kind"], str(sp.get("name") or rec["id"]), _ring(rec["poly"]), FIDELITY,
            room_type="" if rec["kind"] == "tenant" else rec["type"], raw_class=rec["type"],
            source_ref=f"plan:{cfg['id']}", attrs=attrs))
    link_list = sorted(links.values(), key=lambda ln: (ln.a, ln.b))
    inferred = _connect(spaces, link_list)

    # ---- entrances -----------------------------------------------------------
    entrances: list[Entrance] = []
    for e in cfg.get("entrances") or []:
        sid = str(e["space"])
        if sid not in by_id:
            raise ValueError(f"{cfg['id']}: entrance {e.get('id')!r} opens into unknown space {sid!r}")
        level = index_of.get(str(e.get("level")), by_id[sid]["level"])
        xy = to_model(frames[level], e["px"])
        edge = nearest_points(by_id[sid]["poly"].exterior, ShPoint(xy))[0]  # the door sits on the room's wall
        out = SIDES[str(e.get("outward", "S")).upper()]
        doors[sid].append(((edge.x, edge.y), ENTRANCE_W, "doorway"))
        entrances.append(Entrance(str(e["id"]), str(e.get("name") or "Entrance"), sid, rp((edge.x, edge.y)),
                                  (r(out[0]), r(out[1])), "plan"))

    # ---- walls ---------------------------------------------------------------
    walls = _Walls()
    for rec in recs:
        ring = _ring(rec["poly"])
        others = [o["poly"] for o in on_level[rec["level"]] if o is not rec]
        # the lines this room was drawn with: its own outline and the rooms cut out of it
        drawn = [rec["drawn"].exterior] + [by_id[i]["drawn"].exterior for i in rec.get("inside", [])]
        z0 = rec["z0"]
        sides = [SIDES[c] for c in str(rec["cfg"].get("windows") or "").upper() if c in SIDES]
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            length = math.dist(a, b)
            if length < 0.05:
                continue
            mid = ShPoint((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if min(line.distance(mid) for line in drawn) > 0.1:
                continue  # an edge of the slit that joins a cut-out room to the outside: not a wall
            nx, ny = (b[1] - a[1]) / length, -(b[0] - a[0]) / length  # outward (the ring is counter-clockwise)
            probes = [ShPoint(a[0] + (b[0] - a[0]) * t + nx * 0.3, a[1] + (b[1] - a[1]) * t + ny * 0.3)
                      for t in (0.25, 0.5, 0.75)]
            inner = sum(1 for p in probes if any(o.distance(p) < 0.35 for o in others)) >= 2
            ops = []
            for xy, width, kind in doors[rec["id"]]:
                t = ((xy[0] - a[0]) * (b[0] - a[0]) + (xy[1] - a[1]) * (b[1] - a[1])) / (length * length)
                foot = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                if -0.02 <= t <= 1.02 and math.dist(foot, xy) <= TOUCH_M:
                    o = _opening(kind, a, b, xy, width, z0, z0 + DOOR_H + (0.4 if kind == "opening" else 0.0))
                    if o:
                        ops.append(o)
            if not inner and not ops and length >= MIN_WINDOW_WALL_M and any(nx * sx + ny * sy > 0.7 for sx, sy in sides):
                n = max(1, int(length // WINDOW_SPACING_M))
                for k in range(n):
                    t = (k + 0.5) / n
                    o = _opening("window", a, b, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t),
                                 min(WINDOW_W, length / n - 0.6), z0 + WINDOW_SILL, z0 + WINDOW_HEAD)
                    if o:
                        ops.append(o)
            walls.add(rec["level"], a, b, rec["id"], "partition" if inner else "side", ops, thickness=WALL_T,
                      fidelity=FIDELITY)

    # ---- where it sits on the map --------------------------------------------
    geo, yaw, fit = None, 0.0, None
    points = (cfg.get("georeference") or {}).get("control_points") or []
    if len(points) >= 2:
        fit = register_matterport_model([{"model_xy": list(to_model({}, p["px"])), "latlng": p["latlng"]} for p in points])
        geo, yaw = (float(fit["origin_latlng"][0]), float(fit["origin_latlng"][1])), float(fit["rotation_deg"])

    notes = [str(n) for n in cfg.get("notes") or []]
    if fit:
        notes.append(f"Placed on the map from {len(points)} building corners matched to the map outline "
                     f"(fit residual {fit['rms_m']:.1f} m).")
    if inferred:
        notes.append(f"{inferred} room link(s) could not be read off the plan and are inferred.")
    tour = cfg.get("matterport") or {}
    return BuildingModel(
        schema=SCHEMA_VERSION, id=str(cfg["id"]), name=str(cfg["name"]), address=str(cfg.get("address") or ""),
        use_type=str(cfg.get("use_type") or "office"), archetype=ARCHETYPE, tier="traced-plan", yaw_deg=r(yaw, 2),
        geolocation=geo, levels=tuple(levels), spaces=tuple(sorted(spaces, key=lambda s: s.id)),
        walls=tuple(walls.build()), links=tuple(sorted(link_list, key=lambda ln: (ln.a, ln.b, ln.kind))),
        entrances=tuple(entrances),
        stats={"rooms": len(spaces), "levels": len(levels), "links_corridor": counts["corridor"],
               "links_door": counts["door"], "links_room_to_room": counts["room_to_room"],
               "links_vertical": counts["vertical"], "links_inferred": inferred, "rooms_cut_out": carved,
               "georeference_rms_m": fit["rms_m"] if fit else None, "metres_per_px": scale,
               "source": str(cfg.get("source") or "").strip(),
               "tour": {"model_id": str(tour["model_id"]), "name": str(tour.get("name") or ""),
                        "covers": [str(c) for c in tour.get("covers") or []]} if tour.get("model_id") else None},
        notes=tuple(notes),
    )

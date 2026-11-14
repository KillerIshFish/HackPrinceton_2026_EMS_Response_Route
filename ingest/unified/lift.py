"""Track A models (and legacy app buildings) lifted into ``BuildingModel``.

* ``from_model3d``: a surveyed scan, one to one. Rooms, walls and openings keep
  their measured geometry. Room-to-room links come from the scan's own evidence:
  a doorway or open boundary two rooms share, and (when a ``ScanModel`` of the same
  scan is supplied) the scan-position graph. Stair geometry is not in the scan and
  is not invented: floors are joined by the scan graph where it crosses floors,
  otherwise by one *inferred* link between the closest rooms, labelled as such.
* ``from_scan_model``: a scan with no structural layer. Rooms are clusters with a
  synthetic square footprint; links are the measured scan graph. No walls.
* ``from_building``: a ``core.schema.Building`` built by the older generators, so
  the 3D viewer has something to draw for every building the app knows.
"""

from __future__ import annotations

import math
from typing import Optional

from core.schema import Building
from ingest.mp3d.clusters import ScanModel
from ingest.mp3d.model3d import Model3D
from ingest.mp3d.walls import TRAVERSABLE
from ingest.unified.schema import (SCHEMA_VERSION, BuildingModel, Entrance, Level, Link, Opening, Wall,
                                   make_space, polygon_centroid, r, rp)

# Matterport's room classes -> the clue vocabulary's room types (first match wins).
CLASS_RULES: list[tuple[tuple[str, ...], str]] = [
    (("conference", "meeting"), "conference"), (("bath", "restroom", "toilet", "powder"), "restroom"),
    (("kitchen", "dining", "break"), "break_room"), (("stair",), "stairwell"), (("elevator",), "elevator"),
    (("hall", "corridor"), "corridor"), (("entrance", "lobby", "foyer", "reception"), "lobby"),
    (("closet", "storage", "laundry", "pantry"), "storage"), (("utility", "mechanical"), "mechanical"),
    (("garage", "parking"), "parking"), (("living", "family", "lounge", "common"), "common_area"),
    (("office", "bedroom", "den", "study"), "office"), (("balcony", "patio", "deck", "roof"), "rooftop"),
]


def room_type_for(classification: str) -> str:
    text = (classification or "").lower()
    for words, room_type in CLASS_RULES:
        if any(w in text for w in words):
            return room_type
    return "room"


def _connect(spaces, links: list[Link], evidence: str = "inferred") -> int:
    """Join disconnected groups with one link each between their closest rooms. Returns links added."""
    ids = [s.id for s in spaces]
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ln in links:
        if ln.a in parent and ln.b in parent:
            parent[find(ln.a)] = find(ln.b)
    by_id = {s.id: s for s in spaces}
    added = 0
    while True:
        groups: dict[str, list[str]] = {}
        for i in ids:
            groups.setdefault(find(i), []).append(i)
        if len(groups) <= 1:
            return added
        main = max(groups.values(), key=lambda g: (sum(by_id[i].area_m2 for i in g), g[0]))
        best = None
        for root, members in sorted(groups.items()):
            if members is main:
                continue
            for a in members:
                for b in main:
                    sa, sb = by_id[a], by_id[b]
                    d = math.dist(sa.centroid, sb.centroid) + 6.0 * abs(sa.level - sb.level)
                    if best is None or (d, a, b) < best:
                        best = (d, a, b)
        d, a, b = best
        sa, sb = by_id[a], by_id[b]
        floors = abs(sa.level - sb.level)
        links.append(Link(a=min(a, b), b=max(a, b), kind="stairs" if floors else "door", evidence=evidence,
                          length_m=r(d, 2), floors=floors))
        parent[find(a)] = find(b)
        added += 1


def from_model3d(model: Model3D, scan: Optional[ScanModel] = None, model_id: Optional[str] = None,
                 address: str = "", use_type: str = "other") -> BuildingModel:
    levels = tuple(Level(s.floor + 1, s.label or str(s.floor + 1), r(s.z0), r(s.height_m), s.elevation_source)
                   for s in model.slabs)
    spaces = [make_space(rm.id, rm.floor + 1, "room", rm.classification or "Room", rm.polygon, "surveyed",
                         raw_class=rm.classification or "", room_type=room_type_for(rm.classification),
                         source_ref=f"{model.model_id}:{rm.id}")
              for rm in model.rooms]
    by_id = {s.id: s for s in spaces}
    walls = [Wall(id=w.id, level=w.floor + 1, a=rp(w.a), b=rp(w.b), thickness_m=r(w.thickness_m),
                  exterior=w.exterior, open_boundary=w.open_boundary,
                  space_ids=tuple(sorted(i for i in w.room_ids if i in by_id)),
                  openings=tuple(Opening(o.kind, r(o.t0, 4), r(o.t1, 4), r(o.z0), r(o.z1)) for o in w.openings),
                  fidelity="surveyed") for w in model.walls]

    links: dict[tuple[str, str], Link] = {}

    def add_link(a: str, b: str, kind: str, evidence: str, xy=None, floors: int = 0, length=None) -> None:
        if a == b or a not in by_id or b not in by_id:
            return
        key = (min(a, b), max(a, b))
        if key in links:
            return
        ca, cb = by_id[key[0]].centroid, by_id[key[1]].centroid
        if length is None:
            length = (math.dist(ca, xy) + math.dist(xy, cb)) if xy else math.dist(ca, cb)
        links[key] = Link(a=key[0], b=key[1], kind=kind, evidence=evidence, length_m=r(length, 2), floors=floors,
                          xy=rp(xy) if xy else None)

    def at(w: Wall, t: float):
        return (w.a[0] + (w.b[0] - w.a[0]) * t, w.a[1] + (w.b[1] - w.a[1]) * t)

    entrances: list[Entrance] = []
    lowest = min((lv.index for lv in levels), default=1)
    for w in sorted(walls, key=lambda w: w.id):
        if len(w.space_ids) == 2:
            if w.open_boundary:
                add_link(w.space_ids[0], w.space_ids[1], "opening", "opening", xy=at(w, 0.5))
            for o in w.openings:
                if o.kind in TRAVERSABLE:
                    add_link(w.space_ids[0], w.space_ids[1], "door", "opening", xy=at(w, (o.t0 + o.t1) / 2))
        elif len(w.space_ids) == 1 and w.exterior and w.level == lowest:
            for o in w.openings:
                if o.kind != "doorway":
                    continue
                room = by_id[w.space_ids[0]]
                xy = at(w, (o.t0 + o.t1) / 2)
                ex, ey = w.b[0] - w.a[0], w.b[1] - w.a[1]
                n = math.hypot(ex, ey) or 1.0
                nx, ny = -ey / n, ex / n
                if (xy[0] + nx - room.centroid[0]) ** 2 + (xy[1] + ny - room.centroid[1]) ** 2 < \
                        (xy[0] - nx - room.centroid[0]) ** 2 + (xy[1] - ny - room.centroid[1]) ** 2:
                    nx, ny = -nx, -ny
                entrances.append(Entrance(f"ent-{len(entrances) + 1}", f"{room.label} door", room.id, rp(xy),
                                          (r(nx), r(ny)), "opening"))
    # an "Entrance" room's doors first, then the rest, capped
    entrances.sort(key=lambda e: (by_id[e.space_id].room_type != "lobby", e.id))
    entrances = [Entrance(f"ent-{i + 1}", e.name, e.space_id, e.xy, e.outward, e.evidence)
                 for i, e in enumerate(entrances[:6])]

    from_graph = 0
    if scan is not None:
        for ln in scan.links:
            before = len(links)
            add_link(ln.a, ln.b, "stairs" if ln.floors else "opening", "scan-graph", floors=int(ln.floors),
                     length=max(float(ln.gap_m or 0.0), 1.0) + (6.0 if ln.floors else 0.0))
            from_graph += len(links) - before
    link_list = sorted(links.values(), key=lambda ln: (ln.a, ln.b))
    inferred = _connect(spaces, link_list)
    notes = list(model.notes)
    notes.append("Stairs and elevators are not in the scan; floors are joined by "
                 + ("the scan-position graph" if from_graph else "inferred links") + ".")
    if inferred:
        notes.append(f"{inferred} room link(s) had no doorway in the scan and are inferred.")
    mid = model_id or model.model_id
    return BuildingModel(
        schema=SCHEMA_VERSION, id=f"mp-{mid}", name=model.name or mid, address=address, use_type=use_type,
        archetype="whole_scan", tier=f"surveyed-{model.tier}", yaw_deg=r(model.yaw_deg, 2),
        geolocation=tuple(model.geolocation) if model.geolocation else None, levels=levels,
        spaces=tuple(sorted(spaces, key=lambda s: s.id)), walls=tuple(sorted(walls, key=lambda w: w.id)),
        links=tuple(sorted(link_list, key=lambda ln: (ln.a, ln.b, ln.kind))), entrances=tuple(entrances),
        stats={"rooms": len(spaces), "walls": len(walls), "links_from_openings": len(link_list) - from_graph - inferred,
               "links_from_scan_graph": from_graph, "links_inferred": inferred, "source": dict(model.stats)},
        notes=tuple(notes),
    )


def from_scan_model(scan: ScanModel, address: str = "", use_type: str = "other") -> BuildingModel:
    floors = sorted({rm.floor for rm in scan.rooms} | set(scan.floor_heights))
    z = 0.0
    levels = []
    for f in floors:
        h = float(scan.floor_heights.get(f) or 3.0)
        levels.append(Level(f + 1, str(f + 1), r(z), r(h), "scan"))
        z += h
    spaces = []
    for rm in scan.rooms:
        poly = rm.footprint or rm.hull
        if len(poly) < 3:
            continue
        spaces.append(make_space(rm.id, rm.floor + 1, "room", rm.classification or "Room", poly, "cluster",
                                 raw_class=rm.classification or "", room_type=room_type_for(rm.classification),
                                 source_ref=f"{scan.model_id}:{rm.id}"))
    ids = {s.id for s in spaces}
    links = [Link(a=min(ln.a, ln.b), b=max(ln.a, ln.b), kind="stairs" if ln.floors else "opening", evidence="scan-graph",
                  length_m=r(max(float(ln.gap_m or 0.0), 1.0) + (6.0 if ln.floors else 0.0), 2), floors=int(ln.floors))
             for ln in scan.links if ln.a in ids and ln.b in ids and ln.a != ln.b]
    inferred = _connect(spaces, links)
    return BuildingModel(
        schema=SCHEMA_VERSION, id=f"mp-{scan.model_id}", name=scan.name or scan.model_id, address=address,
        use_type=use_type, archetype="scan_clusters", tier="scan-cluster", yaw_deg=r(scan.yaw_deg, 2),
        geolocation=None, levels=tuple(levels), spaces=tuple(sorted(spaces, key=lambda s: s.id)), walls=(),
        links=tuple(sorted(links, key=lambda ln: (ln.a, ln.b, ln.kind))), entrances=(),
        stats={"rooms": len(spaces), "links_from_scan_graph": len(links) - inferred, "links_inferred": inferred},
        notes=tuple(scan.notes) + ("This scan has no wall geometry: rooms are positions with a nominal square "
                                   "footprint of the reported area.",),
    )


def from_building(building: Building, fidelity: str = "template") -> BuildingModel:
    """A legacy app building as a BuildingModel: rooms extruded from their outlines, doors where edges say."""
    from ingest.unified.layout import _Walls, _opening
    from core.route_clues import opening_point

    heights = {int(k): float(v) for k, v in building.floor_heights_m.items()}
    order = sorted(building.floors)
    gaps = [heights[b] - heights[a] for a, b in zip(order, order[1:]) if heights[b] - heights[a] >= 2.0]
    typical = sum(gaps) / len(gaps) if gaps else 3.5
    levels = []
    for i, f in enumerate(order):
        gap = heights[order[i + 1]] - heights[f] if i + 1 < len(order) else typical
        levels.append(Level(f, str(f), r(heights[f]), r(gap if gap >= 2.0 else typical), "building"))
    z_of = {lv.index: lv.z0 for lv in levels}
    kind_of = {"corridor": "corridor", "stairwell": "stair", "elevator": "elevator", "lobby": "lobby",
               "parking": "parking"}
    spaces = [make_space(rm.id, rm.floor, kind_of.get(rm.room_type, "room"), rm.name, rm.polygon, fidelity,
                         room_type=rm.room_type, raw_class=rm.room_type) for rm in building.rooms if len(rm.polygon) >= 3]
    rooms = {rm.id: rm for rm in building.rooms}
    walls = _Walls()
    doors: dict[str, list] = {}
    links = []
    for e in building.edges:
        u, v = rooms.get(e.get("u")), rooms.get(e.get("v"))
        if u is None or v is None:
            continue
        floors = abs(u.floor - v.floor)
        xy = None
        if not floors:
            xy = opening_point(u, v, e)
            doors.setdefault(u.id, []).append(xy)
            doors.setdefault(v.id, []).append(xy)
        links.append(Link(a=min(u.id, v.id), b=max(u.id, v.id), kind=str(e.get("kind") or "door"),
                          evidence="building", length_m=r(float(e.get("length_m") or 1.0), 2), floors=floors,
                          xy=rp(xy) if xy else None))
    for rm in building.rooms:
        n = len(rm.polygon)
        z0 = z_of.get(rm.floor, 0.0)
        for i in range(n):
            a, b = rm.polygon[i], rm.polygon[(i + 1) % n]
            length = math.dist(a, b)
            ops = []
            for xy in doors.get(rm.id, []):
                # a door belongs to the wall segment it lies on
                t = ((xy[0] - a[0]) * (b[0] - a[0]) + (xy[1] - a[1]) * (b[1] - a[1])) / (length * length or 1.0)
                px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                if 0.0 <= t <= 1.0 and math.dist((px, py), xy) < 0.35:
                    o = _opening("doorway", a, b, xy, 0.95, z0, z0 + 2.05)
                    if o:
                        ops.append(o)
            walls.add(rm.floor, a, b, rm.id, "side", ops, fidelity=fidelity)
    entrances = []
    for e in building.entrances:
        room = rooms.get(e.get("room_id"))
        if room is None or not e.get("xy"):
            continue
        cx, cy = polygon_centroid(room.polygon)
        n = math.hypot(e["xy"][0] - cx, e["xy"][1] - cy) or 1.0
        entrances.append(Entrance(str(e.get("id")), str(e.get("name") or "Entrance"), room.id, rp(e["xy"]),
                                  (r((e["xy"][0] - cx) / n), r((e["xy"][1] - cy) / n)), "building"))
    return BuildingModel(
        schema=SCHEMA_VERSION, id=building.id, name=building.name, address="", use_type=building.use_type,
        archetype="app_building", tier=fidelity, yaw_deg=0.0, geolocation=None, levels=tuple(levels),
        spaces=tuple(sorted(spaces, key=lambda s: s.id)), walls=tuple(walls.build()),
        links=tuple(sorted(links, key=lambda ln: (ln.a, ln.b, ln.kind))), entrances=tuple(entrances),
        stats={"rooms": len(spaces)}, notes=(),
    )

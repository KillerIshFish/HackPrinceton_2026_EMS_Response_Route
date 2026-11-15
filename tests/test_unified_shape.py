"""A generated building takes the real building's outline (ingest/unified/shape).

No network and no dataset: outlines are written inline in metres and turned into the lat/lng
rings the site context would hand over. Every case checks the same things: the plates cover
the outline and nothing else, every room can be reached, every unit of the record is there,
and the entrances sit on the outline's walls.

Run: uv run pytest tests/test_unified_shape.py
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from core.building_graph import build_graph
from core.schema import Building
from ingest.georegister import latlng_to_local, local_to_latlng
from ingest.unified import context as ctx_mod
from ingest.unified.pipeline import build_property_model, building_count
from ingest.unified.project import to_building, viewer_payload
from ingest.unified.shape import site_frame
from ingest.unified.stamp import make_interior

ADDRESS = (34.0, -118.0)
SHAPES = {
    "rectangle": ([(0, 0), (60, 0), (60, 20), (0, 20)], []),
    "L": ([(0, 0), (60, 0), (60, 18), (18, 18), (18, 50), (0, 50)], []),
    "U": ([(0, 0), (70, 0), (70, 50), (52, 50), (52, 18), (18, 18), (18, 50), (0, 50)], []),
    "courtyard": ([(0, 0), (70, 0), (70, 60), (0, 60)], [[(20, 20), (50, 20), (50, 40), (20, 40)]]),
    "irregular": ([(0, 0), (80, 0), (90, 25), (70, 55), (0, 45)], []),
    "sliver": ([(0, 0), (40, 0), (40, 6), (0, 6)], []),
}


def _ring(points, yaw: float = 0.0) -> list[list[float]]:
    return [list(local_to_latlng(x, y, ADDRESS, yaw)) for x, y in points]


def _site(name: str, street: float | None = 180.0, yaw: float = 0.0, source: str = "Vexcel") -> dict:
    outer, holes = SHAPES[name]
    return {"source": source, "detail": "test 2026-01-01", "polygons_latlng": [_ring(outer, yaw)],
            "holes_latlng": [[_ring(h, yaw) for h in holes]], "street_bearing_deg": street,
            "address_latlng": list(ADDRESS)}


def _facts(per_floor: int = 12, floors: int = 3, **over) -> dict:
    units = [{"id": str(100 + i), "number": f"{floor}{n:02d}", "building": "", "type": "A1", "beds": 1 + n % 2, "baths": 1,
              "area_m2": 50.0 + 12.0 * (n % 3), "scans": []}
             for i, (floor, n) in enumerate((f, n) for f in range(1, floors + 1) for n in range(1, per_floor + 1))]
    base = {"property_id": 42, "name": "Test Court", "address": "1 Test St, Testville, TS", "latlng": list(ADDRESS),
            "use_type": "apartment", "style": "mid_rise", "stories": floors, "gross_m2": 2400.0, "typical_floor_m2": 800.0,
            "floor_area_m2": {}, "storey_height_m": None, "num_buildings": 1, "total_units": len(units), "has_elevator": True,
            "units": units, "tenants": [], "property_scans": []}
    base.update(over)
    return base


def _poly(space) -> Polygon:
    return Polygon(space.polygon).buffer(0)


def _check(bm, site: dict, slots: int) -> Building:
    """The checks every outline-shaped building must pass; returns the app building."""
    assert bm.stats["shape"]["kind"] == "outline", bm.notes
    outline = unary_union(site_frame(site)["polygons"])
    indoors = [s for s in bm.spaces if s.kind != "walkway"]
    # nothing is drawn outside the real outline, and each floor covers all of it
    drawn = unary_union([_poly(s) for s in indoors])
    assert drawn.difference(outline.buffer(0.02)).area <= 0.02 * drawn.area
    for level in {s.level for s in indoors}:
        plate = unary_union([_poly(s) for s in indoors if s.level == level and s.parent is None])
        assert abs(plate.area - outline.area) <= 0.02 * outline.area, level
    # the units of the record are all there, once each
    assert sum(s.kind in ("unit", "tenant") for s in bm.spaces) == slots
    # a unit's interior stays inside the unit
    by_id = {s.id: s for s in bm.spaces}
    for child in (s for s in bm.spaces if s.parent):
        assert _poly(by_id[child.parent]).buffer(0.05).contains(_poly(child)), child.id
    # every room can be reached from the entrances, which sit on the outline's walls
    b = to_building(bm)
    Building.model_validate(b.model_dump())
    assert nx.is_connected(build_graph(b))
    assert bm.entrances and b.staging_spots
    for e in bm.entrances:
        assert outline.exterior.distance(Point(e.xy)) < 0.05, e.name
        assert abs(math.hypot(*e.outward) - 1.0) < 1e-6
        assert not outline.contains(Point(e.xy[0] + e.outward[0], e.xy[1] + e.outward[1])), "outward points out"
    # the notes say where the outline came from and that the interior is generated
    notes = " ".join(bm.notes)
    assert site["source"] in notes and "interior is generated" in notes
    return b


@pytest.mark.parametrize("name", ["rectangle", "L", "U", "courtyard", "irregular"])
def test_plates_follow_the_outline(name: str) -> None:
    site = _site(name)
    facts = _facts()
    bm = build_property_model(facts, site=site)
    assert bm.to_json() == build_property_model(_facts(), site=_site(name)).to_json(), "same record and outline, same bytes"
    b = _check(bm, site, slots=36)
    # every unit has a door and, where it has an outer wall, a window in it
    for u in (s for s in bm.spaces if s.kind == "unit"):
        mine = [w for w in bm.walls if u.id in w.space_ids]
        assert any(o.kind == "doorway" for w in mine for o in w.openings), u.label
        if any(w.exterior for w in mine):
            assert any(o.kind == "window" and w.exterior for w in mine for o in w.openings), u.label
    assert sum(r.has_windows for r in b.rooms if r.room_type == "unit") >= 18
    # the main entrance is first, and on the side that faces the street (south: -y)
    assert bm.entrances[0].name.startswith("Lobby") and bm.entrances[0].outward[1] < -0.7
    # stairs and the elevator stack: every floor is joined to the next
    assert sum(1 for ln in bm.links if ln.floors) >= 2 * 2
    payload = viewer_payload(bm)
    assert payload["counts"]["spaces"] == len(bm.spaces) and payload["walls"] and payload["entrances"]
    assert any("outline" in n for n in payload["notes"])


def test_corridors_follow_the_wings() -> None:
    """An L gets a corridor down each arm, a courtyard block a ring: not one bar with corners cut off."""
    def corridors(name: str) -> list:
        bm = build_property_model(_facts(), site=_site(name))
        return [s for s in bm.spaces if s.kind == "corridor" and s.level == 1]

    assert len(corridors("rectangle")) == 1
    arms = corridors("L")
    assert len(arms) == 2
    spans = sorted((max(p[0] for p in c.polygon) - min(p[0] for p in c.polygon),
                    max(p[1] for p in c.polygon) - min(p[1] for p in c.polygon)) for c in arms)
    assert spans[0][1] > 20 > spans[0][0] and spans[1][0] > 20 > spans[1][1], "one runs along x, the other along y"
    assert len(corridors("U")) == 3
    ring = corridors("courtyard")
    assert len(ring) == 4
    hole = Polygon(SHAPES["courtyard"][1][0])
    frame = site_frame(_site("courtyard"))
    centre = frame["polygons"][0].centroid
    assert all(not _poly(c).contains(centre) for c in ring) and hole.area > 0


def test_units_keep_their_order_and_proportions() -> None:
    site = _site("rectangle")
    facts = _facts(per_floor=20)
    bm = build_property_model(facts, site=site)
    _check(bm, site, slots=60)
    recorded = {f"u{u['id']}": u["area_m2"] for u in facts["units"]}
    scale = [s.area_m2 / recorded[s.id] for s in bm.spaces if s.kind == "unit"]
    assert max(scale) / min(scale) < 2.0, "units are scaled together to fill the plate"
    # numbers run along the corridor: on one side of it, a higher number is never behind a lower one
    front = sorted((s for s in bm.spaces if s.kind == "unit" and s.level == 2 and s.centroid[1] < 0),
                   key=lambda s: s.centroid[0])
    assert [s.label for s in front] == sorted(s.label for s in front)


def test_main_entrance_turns_to_the_street() -> None:
    # the street lies off the east end of the bar: the lobby is at that end, not on a long side
    site = _site("rectangle", street=90.0)
    bm = build_property_model(_facts(), site=site)
    _check(bm, site, slots=36)
    main = bm.entrances[0]
    frame = site_frame(site)
    assert main.outward[0] * frame["front"][0] + main.outward[1] * frame["front"][1] > 0.7


def test_thin_sliver_gets_a_single_row_of_rooms() -> None:
    site = _site("sliver")
    bm = build_property_model(_facts(per_floor=3), site=site)
    _check(bm, site, slots=9)
    corridor = next(s for s in bm.spaces if s.kind == "corridor")
    assert max(p[1] for p in corridor.polygon) - min(p[1] for p in corridor.polygon) < 3.5


@pytest.mark.parametrize("points, why", [
    ([(0, 0), (6, 0), (6, 5), (0, 5)], "too small"),                 # a shed
    ([(0, 0), (40, 0), (40, 3), (0, 3)], "too narrow"),              # a wall, not a building
    ([(0, 0), (22, 0), (22, 9), (0, 9)], "far smaller than the units"),  # a neighbour's garage
])
def test_an_outline_that_cannot_be_the_building_falls_back_to_the_bar(points, why: str) -> None:
    site = {"source": "OpenStreetMap", "detail": "", "polygons_latlng": [_ring(points)], "street_bearing_deg": 180.0,
            "address_latlng": list(ADDRESS)}
    facts = _facts(per_floor=3 if why == "too narrow" else 12)
    bm = build_property_model(facts, site=site)
    assert bm.stats["shape"]["kind"] == "assumed" and why in bm.stats["shape"]["reason"]
    assert "shape of the building is assumed" in bm.notes[0] and "OpenStreetMap" in bm.notes[0]
    assert bm.yaw_deg == 0.0 and nx.is_connected(build_graph(to_building(bm)))
    # the bar itself is what it was before outlines were used
    plain = build_property_model(facts)
    assert [s.polygon for s in bm.spaces] == [s.polygon for s in plain.spaces]
    assert "no building outline was found" in plain.notes[0]


def test_stamped_interior_stays_inside_its_unit() -> None:
    from tests.test_unified import _test_model

    model = _test_model()
    facts = _facts()
    for u in facts["units"][:8]:
        u["scans"] = ["TESTSCAN"]
    site = _site("L")
    bm = build_property_model(facts, {"TESTSCAN": model}, site=site)
    assert bm.tier == "template+stamped" and bm.stats["units_with_stamped_interior"] == 8
    assert sum(s.fidelity == "stamped" for s in bm.spaces) == 16
    _check(bm, site, slots=36)
    assert make_interior({})({"scans": ["missing"], "id": "u"}, (0, 0, 5, 9), "S", 1, 0.0, 3.0, None) is None


def test_several_buildings_use_outlines_only_when_the_counts_match() -> None:
    def ring_at(dx: float, dy: float, w: float, h: float) -> list:
        return _ring([(dx, dy), (dx + w, dy), (dx + w, dy + h), (dx, dy + h)])

    facts = _facts(per_floor=12, floors=2, num_buildings=2, style="garden", has_elevator=False)
    for i, u in enumerate(facts["units"]):
        u["building"] = "A" if i % 3 else "B"   # A has twice the units of B
    assert building_count(facts) == 2
    two = {"vexcel_all": {"source": "Vexcel", "detail": "", "polygons_latlng": [ring_at(0, 0, 30, 14), ring_at(0, 40, 60, 18)]}}
    site = ctx_mod.layout_site(two, ADDRESS, 2)
    bm = build_property_model(facts, site=site)
    assert bm.stats["shape"]["kind"] == "outline" and bm.stats["blocks"] == 2
    # largest building to largest outline, and the notes say that this is assumed
    area = {k: unary_union([_poly(s) for s in bm.spaces if s.block == k and s.level == 1 and s.parent is None
                            and s.kind != "walkway"]).area for k in ("A", "B")}
    assert area["A"] > 1000 > area["B"] > 400
    assert any("largest" in n and "assumed" in n for n in bm.notes)
    b = to_building(bm)
    assert nx.is_connected(build_graph(b)), "the buildings are joined on the ground by walkways"
    assert sum(r.room_type == "unit" for r in b.rooms) == 24

    # three outlines for two buildings: nothing to match, so today's grid, and the notes say why
    three = {"vexcel_all": {"source": "Vexcel", "detail": "", "polygons_latlng": two["vexcel_all"]["polygons_latlng"]
                            + [ring_at(80, 0, 20, 12)]}}
    site = ctx_mod.layout_site(three, ADDRESS, 2)
    assert site["polygons_latlng"] == [] and "2 buildings and 3 outlines" in site["reason"]
    grid = build_property_model(facts, site=site)
    assert grid.stats["shape"]["kind"] == "assumed" and "2 buildings and 3 outlines" in grid.notes[0]
    assert any("placed on a grid" in n for n in grid.notes)


def test_model_sits_on_the_real_outline_on_the_map(tmp_path, monkeypatch) -> None:
    """Through the site context: the model's outline and the real one coincide, entrances are on it."""
    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", tmp_path)
    yaw = 35.0
    site = _site("L", street=215.0, yaw=yaw)
    outline = {"source": "Vexcel", "match": "contains the address point", "detail": "test 2026-01-01", "area_m2": 1656.0,
               "polygons_latlng": site["polygons_latlng"], "centre_latlng": site["polygons_latlng"][0][0],
               "length_m": 60.0, "width_m": 50.0, "axis_deg": yaw}
    state = {"vexcel": outline, "street": {"bearing_deg": 215.0, "name": "Test St"}, "vexcel_done": True, "osm_done": True}
    laid = ctx_mod.layout_site(state, ADDRESS, 1)
    assert laid["source"] == "Vexcel" and laid["street_bearing_deg"] == 215.0
    bm = build_property_model(_facts(), site=laid)
    ctx = ctx_mod.build_context(bm, ADDRESS, sources=state)
    assert ctx["layout_placed"] is True and ctx["fit_scale"] is None and ctx["shape"]["kind"] == "outline"
    assert ctx["placement_source"].startswith("the Vexcel building outline")
    kinds = [o["kind"] for o in ctx["outlines"]]
    assert kinds == ["building", "model"]
    real = Polygon([latlng_to_local(la, lo, ADDRESS) for la, lo in ctx["outlines"][0]["latlng"]])
    model = Polygon([latlng_to_local(la, lo, ADDRESS) for la, lo in ctx["outlines"][1]["latlng"]])
    assert real.exterior.hausdorff_distance(model.exterior) < 1.0, "the two outlines coincide to about a metre"
    b = to_building(bm, ctx)
    for e in b.entrances:
        assert real.exterior.distance(Point(latlng_to_local(e["latlng"][0], e["latlng"][1], ADDRESS))) < 0.5, e["name"]
    for s in b.staging_spots:
        d = real.exterior.distance(Point(latlng_to_local(s["latlng"][0], s["latlng"][1], ADDRESS)))
        assert 6.0 < d < 7.5 and not real.contains(Point(latlng_to_local(s["latlng"][0], s["latlng"][1], ADDRESS)))
    # the main entrance faces the street (bearing 215: south-west)
    main = b.entrances[0]
    stage = next(s for s in b.staging_spots if s["near_entrance_id"] == main["id"])
    east, north = latlng_to_local(stage["latlng"][0], stage["latlng"][1], main["latlng"])
    assert abs((math.degrees(math.atan2(east, north)) - 215.0 + 180.0) % 360.0 - 180.0) < 50.0

    # the sidecar is reused on the next run: nothing is fetched again
    def no_network(*a, **k):
        raise AssertionError("a finished sidecar must not be asked for again")

    monkeypatch.setattr(ctx_mod, "fetch_vexcel_outline", no_network)
    monkeypatch.setattr(ctx_mod, "fetch_site", no_network)
    again = ctx_mod.site_sources(bm.id, ADDRESS)
    assert ctx_mod.layout_site(again, ADDRESS, 1)["polygons_latlng"] == laid["polygons_latlng"]


def test_window_views_come_from_the_new_window_positions(tmp_path, monkeypatch) -> None:
    """A car park mapped east of the building is seen from the east wall's windows only."""
    from ingest.unified.surroundings import surroundings_from_payload

    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", tmp_path)
    site = _site("rectangle")
    lot = _ring([(70, -10), (110, -10), (110, 30), (70, 30)])
    payload = {"elements": [{"type": "way", "tags": {"amenity": "parking"},
                             "geometry": [{"lat": la, "lon": lo} for la, lo in lot + lot[:1]]}]}
    state = {"vexcel": {"source": "Vexcel", "detail": "", "polygons_latlng": site["polygons_latlng"],
                        "centre_latlng": list(ADDRESS), "length_m": 60.0, "width_m": 20.0, "axis_deg": 0.0},
             "street": {"bearing_deg": 180.0, "name": "Test St"},
             "surroundings": surroundings_from_payload(payload, *ADDRESS), "vexcel_done": True, "osm_done": True}
    bm = build_property_model(_facts(), site=ctx_mod.layout_site(state, ADDRESS, 1))
    b = to_building(bm, ctx_mod.build_context(bm, ADDRESS, sources=state))
    facing = {r.id: r.window_facing for r in b.rooms if r.room_type == "unit" and r.floor == 1}
    assert "parking_lot" in facing.values() and "street" in facing.values()
    x_max = max(r.centroid[0] for r in b.rooms)
    for r in b.rooms:
        if r.id in facing and facing[r.id] == "parking_lot":
            assert r.centroid[0] > x_max - 25.0, "only the east end looks at the car park"


def test_vexcel_answers_are_kept_on_disk(tmp_path, monkeypatch) -> None:
    import ingest.mp3d.aerial as aerial
    import ingest.vexcel as vexcel

    monkeypatch.setattr(vexcel, "vexcel_data_dir", lambda: tmp_path)
    calls = []

    class Client:
        def __init__(self, **_):
            pass

        def property_extract(self, lat, lon):
            calls.append((lat, lon))
            return {"features": []}

    monkeypatch.setattr(aerial, "VexcelClient", Client)
    assert ctx_mod.vexcel_property(34.0, -118.0) == {"features": []}
    assert ctx_mod.vexcel_property(34.0, -118.0) == {"features": []}
    assert calls == [(34.0, -118.0)], "the second answer comes from disk"
    assert ctx_mod.wkt_polygons("POLYGON ((0 0, 4 0, 4 4, 0 0), (1 1, 2 1, 2 2, 1 1))")[0][1][0] == (1.0, 1.0)


def test_late_outline_rebuilds_the_layout(tmp_path, monkeypatch) -> None:
    """OpenStreetMap answers after the deadline with the only outline: the building is laid out again."""
    import threading
    import time

    from ingest.unified import index, pipeline

    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", tmp_path / "context")
    monkeypatch.setattr(pipeline, "PROCESSED", tmp_path)
    monkeypatch.setattr(pipeline, "MODELS_DIR", tmp_path / "models")
    facts = _facts()
    monkeypatch.setattr(index, "lookup", lambda q, limit=5: [{"property_id": 42, "address": "1 Test St", "city": "Testville"}])
    monkeypatch.setattr(pipeline, "property_facts", lambda pid: facts)
    monkeypatch.setattr(ctx_mod, "STREET_DEADLINE_S", 0.3)
    release = threading.Event()
    ring = SHAPES["L"][0]
    way = {"type": "way", "tags": {"building": "yes"},
           "geometry": [{"lat": la, "lon": lo} for la, lo in _ring([(x - 5, y - 5) for x, y in ring + ring[:1]])]}

    def slow_site(lat, lon):
        release.wait(5.0)
        return {"elements": [way]}

    monkeypatch.setattr(ctx_mod, "fetch_vexcel_outline", lambda lat, lon, single=True: None)
    monkeypatch.setattr(ctx_mod, "fetch_site", slow_site)
    monkeypatch.setattr(pipeline, "site_sources",
                        lambda *a, **k: ctx_mod.site_sources(*a, **{**k, "deadline_s": 0.3}))
    report = pipeline.generate("42", stamp=False)
    first = pipeline.load_model("prop-42")
    assert first.stats["shape"]["kind"] == "assumed" and report["building_id"] == "prop-42"
    release.set()
    for _ in range(100):
        time.sleep(0.05)
        late = pipeline.load_model("prop-42")
        if late.stats["shape"]["kind"] == "outline":
            break
    assert late.stats["shape"] == {**late.stats["shape"], "kind": "outline", "source": "OpenStreetMap"}
    assert ctx_mod.load_context("prop-42")["pending"] is False

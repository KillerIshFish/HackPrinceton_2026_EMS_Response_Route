"""What each window looks at, from OpenStreetMap features (ingest/unified/surroundings).

No network: Overpass payloads are written inline, in metres around a made-up address
point, and the site-context sources are faked.

Run: uv run pytest tests/test_unified_surroundings.py
"""

from __future__ import annotations

from ingest.georegister import local_to_latlng
from ingest.unified.surroundings import (facing_for_windows, local_features, mark_own, ray_view,
                                         surroundings_from_payload)

ADDRESS = (34.0, -118.0)


def _geom(points) -> list[dict]:
    out = []
    for x, y in points:
        lat, lon = local_to_latlng(x, y, ADDRESS)
        out.append({"lat": lat, "lon": lon})
    return out


def _way(tags: dict, points, closed: bool = True) -> dict:
    pts = list(points) + ([points[0]] if closed else [])
    return {"type": "way", "tags": tags, "geometry": _geom(pts)}


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _payload() -> dict:
    """The modelled building (50 x 18 m) with a street to the south, a car park to the north,
    houses to the east and an office block close by on the west."""
    return {"elements": [
        _way({"highway": "residential", "name": "Test Street"}, [(-100, -30), (100, -30)], closed=False),
        _way({"highway": "service"}, [(-100, 14), (100, 14)], closed=False),          # a parking aisle: not a view
        _way({"building": "yes", "name": "Own"}, _rect(-25, -9, 25, 9)),               # OSM's tracing of the building itself
        _way({"amenity": "parking"}, _rect(-30, 12, 30, 55)),
        _way({"amenity": "parking", "parking": "underground"}, _rect(-30, -28, 30, -12)),  # not visible
        _way({"building": "house"}, _rect(40, -6, 52, 6)),
        _way({"building": "yes"}, _rect(60, -6, 70, 4)),                               # small, on residential land
        _way({"landuse": "residential"}, _rect(35, -60, 110, 60)),
        _way({"building": "commercial", "name": "Block B"}, _rect(-60, -15, -33, 15)),
    ]}


def test_overpass_elements_become_classed_features() -> None:
    s = surroundings_from_payload(_payload(), *ADDRESS)
    assert s["source"] == "OpenStreetMap" and s["places"] == []
    # the service road, the underground car park and the land-use polygon are not features
    assert s["counts"] == {"street": 1, "parking_lot": 1, "residential": 2, "adjacent_building": 2}
    assert [f["shape"] for f in s["features"] if f["class"] == "street"] == ["line"]
    assert s == surroundings_from_payload(_payload(), *ADDRESS), "same payload, same sidecar"
    assert surroundings_from_payload({"elements": []}, *ADDRESS)["features"] == []


def test_own_building_is_flagged_and_ignored() -> None:
    s = surroundings_from_payload(_payload(), *ADDRESS)
    # the chosen outline is another tracing of the same roof, a metre or two off
    outline = [list(local_to_latlng(x, y, ADDRESS)) for x, y in _rect(-24, -10, 27, 8)]
    marked = mark_own(s, [outline], list(ADDRESS))
    assert marked["own_buildings"] == 1
    assert [f["name"] for f in marked["features"] if f["own"]] == ["Own"]
    assert all(not f["own"] for f in s["features"]), "the input is not modified"
    assert len(local_features(marked, ADDRESS)) == len(s["features"]) - 1
    # with no outline at all, the building the model's centre stands in is still its own
    assert mark_own(s, [], list(ADDRESS))["own_buildings"] == 1
    assert mark_own(None, [], list(ADDRESS)) is None


def test_a_window_faces_the_nearest_thing_in_front_of_it() -> None:
    feats = local_features(mark_own(surroundings_from_payload(_payload(), *ADDRESS), [], list(ADDRESS)), ADDRESS)
    assert facing_for_windows([(0.0, -9.0, 180.0)], feats) == "street"
    assert facing_for_windows([(0.0, 9.0, 0.0)], feats) == "parking_lot"
    assert facing_for_windows([(25.0, 0.0, 90.0)], feats) == "residential"
    assert facing_for_windows([(-25.0, 0.0, 270.0)], feats) == "adjacent_building"
    cls, dist = ray_view(0.0, -9.5, 180.0, feats)
    assert cls == "street" and abs(dist - 20.5) < 0.05
    # nothing mapped within range: no opinion, so the caller can fall back to the street bearing
    assert facing_for_windows([(0.0, 300.0, 0.0)], feats) is None
    assert facing_for_windows([], feats) is None

    # a car park between the glass and the street: the caller sees the car park
    lot = [("street", False, [(-100.0, -60.0), (100.0, -60.0)]), ("parking_lot", True, _rect(-40, -50, 40, -12))]
    assert facing_for_windows([(0.0, -9.0, 180.0)], lot) == "parking_lot"
    # a room with glass on two sides takes the class most of its rays agree on
    assert facing_for_windows([(0.0, 9.0, 0.0), (10.0, 9.0, 0.0), (0.0, -9.0, 180.0)], feats) == "parking_lot"


def test_generated_building_gets_views_from_the_site_context(tmp_path, monkeypatch) -> None:
    from ingest.georegister import latlng_to_local
    from ingest.unified import context as ctx_mod
    from ingest.unified.pipeline import build_property_model
    from ingest.unified.project import to_building
    from tests.test_unified import _facts

    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", tmp_path)
    ring = [list(local_to_latlng(x, y, ADDRESS)) for x, y in _rect(-25, -9, 25, 9)]
    outline = ctx_mod._outline("Vexcel", [ctx_mod._measure([latlng_to_local(la, lo, ADDRESS) for la, lo in ring])],
                               "contains the address point", *ADDRESS)
    monkeypatch.setattr(ctx_mod, "fetch_vexcel_outline", lambda lat, lon, single=True: outline)
    monkeypatch.setattr(ctx_mod, "fetch_site", lambda lat, lon: _payload())
    bm = build_property_model(_facts())
    ctx = ctx_mod.build_context(bm, ADDRESS, deadline_s=5.0)
    assert ctx["pending"] is False and ctx["street"]["name"] == "Test Street"
    assert ctx["surroundings"]["own_buildings"] == 1

    b = to_building(bm, ctx)
    windowed = [r for r in b.rooms if r.has_windows]
    assert windowed
    mid_y = (min(p[1] for r in b.rooms for p in r.polygon) + max(p[1] for r in b.rooms for p in r.polygon)) / 2
    front = {r.window_facing for r in windowed if r.centroid[1] < mid_y}
    rear = {r.window_facing for r in windowed if r.centroid[1] > mid_y}
    assert front == {"street"} and rear == {"parking_lot"}, (front, rear)

    # without the surroundings the old rule still holds: glass turned toward the street faces it
    plain = dict(ctx, surroundings=None)
    b2 = to_building(bm, plain)
    assert {r.window_facing for r in b2.rooms if r.has_windows and r.centroid[1] < mid_y} == {"street"}
    assert {r.window_facing for r in b2.rooms if r.has_windows and r.centroid[1] > mid_y} == {None}


def test_local_extract_is_indexed_and_answers_like_overpass(tmp_path) -> None:
    import pytest

    pytest.importorskip("osmium")
    from ingest.unified import osm_local

    # a tiny extract: a street, a service drive, a building, a car park and a named shop
    nodes = {1: (34.0000, -118.0010), 2: (34.0000, -117.9990),                       # the street
             3: (34.0002, -118.0002), 4: (34.0002, -117.9998), 5: (34.0004, -117.9998), 6: (34.0004, -118.0002),  # building
             7: (34.0006, -118.0003), 8: (34.0006, -117.9997), 9: (34.0009, -117.9997), 10: (34.0009, -118.0003),  # car park
             11: (34.0001, -118.0001)}
    ways = [(100, [1, 2], {"highway": "residential", "name": "Test Street"}),
            (101, [3, 4, 5, 6, 3], {"building": "retail", "name": "Shop Block"}),
            (102, [7, 8, 9, 10, 7], {"amenity": "parking"}),
            (103, [1, 3], {"highway": "footway"})]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
    for nid, (lat, lon) in nodes.items():
        tags = '<tag k="shop" v="convenience"/><tag k="name" v="Corner Shop"/>' if nid == 11 else ""
        xml.append(f'<node id="{nid}" version="1" lat="{lat}" lon="{lon}">{tags}</node>')
    for wid, refs, tags in ways:
        body = "".join(f'<nd ref="{r}"/>' for r in refs) + "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        xml.append(f'<way id="{wid}" version="1">{body}</way>')
    xml.append("</osm>")
    (tmp_path / "tiny.osm").write_text("\n".join(xml), encoding="utf-8")

    index = tmp_path / "osm_index.sqlite"
    regions = {"test": ("tiny.osm", "", (33.99, -118.01, 34.01, -117.99))}
    assert osm_local.build_index(index, tmp_path, regions) == {"test": 4}  # the footway is not kept

    site = osm_local.fetch_site(34.0003, -118.0, 150.0, index)
    assert sorted(e["type"] for e in site["elements"]) == ["node", "way", "way", "way"]
    s = surroundings_from_payload(site, 34.0003, -118.0)
    assert s["source"] == "local OpenStreetMap extract"
    assert s["counts"] == {"street": 1, "parking_lot": 1, "residential": 0, "adjacent_building": 1}
    assert [p["name"] for p in s["places"]] == ["Corner Shop"]
    # a point whose neighbourhood leaves the indexed box is not answered locally (Overpass is asked instead)
    assert osm_local.fetch_site(34.0099, -118.0, 150.0, index) is None
    assert osm_local.fetch_site(40.0, -100.0, 150.0, index) is None
    assert osm_local.fetch_site(34.0003, -118.0, 150.0, tmp_path / "missing.sqlite") is None

    # the site pack: cut from the full index around given points, gzipped, unpacked elsewhere
    pack = tmp_path / "pack" / "site_pack.sqlite.gz"
    stats = osm_local.export_pack([(34.0003, -118.0), (40.0, -100.0)], pack, index)
    assert stats["sites"] == 1 and stats["elements"] == 4, "a point outside the full index is left out"
    packed_index = tmp_path / "unpacked.sqlite"
    assert osm_local.unpack(pack, packed_index) is True
    assert osm_local.unpack(pack, packed_index) is False, "an index already there is left alone"
    assert osm_local.fetch_site(34.0003, -118.0, 150.0, packed_index)["elements"] == site["elements"]
    # the pack answers only near the properties it was cut for
    assert osm_local.fetch_site(34.0030, -118.0, 150.0, packed_index) is None
    osm_local._close_all()

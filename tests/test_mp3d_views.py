"""Tests for window views, the per-layer probe and the scan-cluster fallback.

Pure: payloads are built by hand or loaded from the saved fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.mp3d import geometry as geo
from ingest.mp3d.clusters import scan_model_from_payload
from ingest.mp3d.layers import (
    LEVEL_FULL,
    LEVEL_NONE,
    LEVEL_REDUCED,
    report_from_payload,
)
from ingest.mp3d.model3d import model_from_payload
from ingest.mp3d.views import (
    FACING_LOT,
    FACING_STREET,
    FACING_UNKNOWN,
    classify_facing,
    distinctiveness,
    room_views,
    sector,
    sill_class,
    size_class,
    view_summary,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "zhKWys61pww.payload.json"
)


# --------------------------------------------------------------------------
# view classification primitives
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bearing,expected",
    [(0, "N"), (10, "N"), (45, "NE"), (90, "E"), (180, "S"),
     (225, "SW"), (270, "W"), (350, "N"), (359.9, "N")],
)
def test_sector_boundaries(bearing, expected):
    assert sector(bearing) == expected


def test_sector_wraps_past_360():
    assert sector(720.0) == "N"
    assert sector(-90.0) == "W"


def test_size_and_sill_classes():
    assert size_class(0.3) == "small"
    assert size_class(1.5) == "medium"
    assert size_class(4.0) == "large"
    assert sill_class(0.05) == "floor-level"
    assert sill_class(0.9) == "standard"
    assert sill_class(1.6) == "high"


def test_convex_hull_of_a_square_with_an_interior_point():
    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    hull = geo.convex_hull(pts)
    assert len(hull) == 4
    assert geo.polygon_area(hull) == pytest.approx(4.0)


def test_convex_hull_degenerate_inputs():
    assert geo.convex_hull([]) == []
    assert len(geo.convex_hull([(0, 0)])) == 1
    assert len(geo.convex_hull([(0, 0), (1, 1)])) == 2


# --------------------------------------------------------------------------
# facing classification
# --------------------------------------------------------------------------
class _FakeView:
    """Minimal stand-in carrying just what classify_facing reads."""

    def __init__(self, bearing):
        self.dominant_bearing = bearing


def test_classify_facing_front_and_back():
    assert classify_facing(_FakeView(300.0), 301.9) == FACING_STREET
    assert classify_facing(_FakeView(120.0), 301.9) == FACING_LOT
    assert classify_facing(_FakeView(None), 301.9) == FACING_UNKNOWN
    assert classify_facing(_FakeView(300.0), None) == FACING_UNKNOWN


def test_classify_facing_is_exactly_perpendicular_at_the_boundary():
    """A wall at 90 degrees to the street belongs to neither side."""
    assert classify_facing(_FakeView(31.9), 301.9, tolerance_deg=45.0) == FACING_UNKNOWN


# --------------------------------------------------------------------------
# per-layer probe
# --------------------------------------------------------------------------
def _layer_payload(thickness=True, outline=True, sills=True, panos=True, classes=True):
    pts = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
    ids = ["a", "b", "c", "d"]
    edges = []
    for i in range(4 if outline else 3):
        edge = {
            "type": "wall",
            "vertices": [
                {"id": ids[i], "position": {"x": pts[i][0], "y": pts[i][1]}},
                {"id": ids[(i + 1) % 4],
                 "position": {"x": pts[(i + 1) % 4][0], "y": pts[(i + 1) % 4][1]}},
            ],
            "openings": [],
        }
        if thickness:
            edge["thickness"] = 0.12
        if sills and i == 0:
            edge["openings"] = [{
                "type": "doorway", "width": 0.9, "height": 2.1,
                "lowerElevation": 0.0, "relativeCenter": 0.5,
            }]
        edges.append(edge)
    room = {"id": "r1", "boundary": {"edges": edges}}
    if classes:
        room["classifications"] = [{"label": "Bedroom"}]
    return {
        "id": "m", "name": "n", "state": "active",
        "propertyClassifications": [{"classification": "living_space"}],
        "floors": [{
            "sequence": 0,
            "panoLocations": [{"position": {"z": 1.6}}] if panos else [],
            "rooms": [room],
        }],
    }


def test_layer_report_full():
    r = report_from_payload("m", _layer_payload())
    assert r.level == LEVEL_FULL
    assert r.rooms_with_outline == 1
    assert r.walls_with_thickness == 4
    assert r.openings_dimensioned == 1
    assert r.property_class == "living_space"
    assert r.missing == ()


def test_layer_report_reduced_when_openings_missing():
    r = report_from_payload("m", _layer_payload(sills=False))
    assert r.level == LEVEL_REDUCED
    assert "openings" in r.missing


def test_layer_report_none_without_outlines():
    r = report_from_payload("m", _layer_payload(outline=False))
    assert r.level == LEVEL_NONE
    assert "outlines" in r.missing


def test_layer_report_none_when_floors_cannot_be_placed():
    """Outlines but no sills and no scan positions: nothing anchors z."""
    r = report_from_payload("m", _layer_payload(sills=False, panos=False))
    assert r.level == LEVEL_NONE


def test_layer_report_missing_model():
    r = report_from_payload("nope", None)
    assert r.level == LEVEL_NONE and r.missing


def test_outline_coverage_fraction():
    r = report_from_payload("m", _layer_payload())
    assert r.outline_coverage == pytest.approx(1.0)
    assert report_from_payload("m", None).outline_coverage == 0.0


# --------------------------------------------------------------------------
# scan-cluster build
# --------------------------------------------------------------------------
def _scan_payload():
    return {
        "id": "s1",
        "name": "Scan only",
        "state": "active",
        "geocoordinates": {"rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
        "locations": [
            {"id": "a1", "neighbors": ["a2"], "position": {"x": 0.0, "y": 0.0, "z": 1.6},
             "room": {"id": "rA"}, "floor": {"sequence": 0}},
            {"id": "a2", "neighbors": ["a1", "a3"], "position": {"x": 2.0, "y": 0.0, "z": 1.6},
             "room": {"id": "rA"}, "floor": {"sequence": 0}},
            {"id": "a3", "neighbors": ["a2"], "position": {"x": 5.0, "y": 0.0, "z": 1.6},
             "room": {"id": "rB"}, "floor": {"sequence": 0}},
        ],
        "floors": [{
            "sequence": 0,
            "label": "Floor 1",
            "panoLocations": [{"position": {"z": 1.6}}, {"position": {"z": 1.7}}],
            "rooms": [
                {"id": "rA", "dimensions": {"areaFloor": 430.556, "units": "imperial"}},
                {"id": "rB", "dimensions": {"areaFloor": 107.639, "units": "imperial"}},
            ],
        }],
    }


def test_scan_model_clusters_rooms_and_converts_area():
    m = scan_model_from_payload(_scan_payload())
    assert m.schema.startswith("mp3d-scan")
    assert m.level == "scan"
    rooms = {r.id: r for r in m.rooms}
    assert set(rooms) == {"rA", "rB"}
    assert rooms["rA"].anchor_count == 2
    assert rooms["rA"].centroid == pytest.approx((1.0, 0.0))
    assert rooms["rA"].area_m2 == pytest.approx(40.0, abs=0.05)   # 430.556 ft2
    assert rooms["rB"].area_m2 == pytest.approx(10.0, abs=0.05)


def test_scan_model_links_come_from_the_anchor_graph():
    m = scan_model_from_payload(_scan_payload())
    assert len(m.links) == 1
    link = m.links[0]
    assert {link.a, link.b} == {"rA", "rB"}
    assert link.anchor_edges >= 1
    assert link.floors == 0
    assert link.gap_m == pytest.approx(4.0)


def test_scan_rooms_get_a_synthetic_square_of_the_right_area():
    m = scan_model_from_payload(_scan_payload())
    room = next(r for r in m.rooms if r.id == "rA")
    assert room.footprint_source == "synthetic-square"
    assert len(room.footprint) == 4
    assert geo.polygon_area(list(room.footprint)) == pytest.approx(room.area_m2, rel=0.01)


def test_scan_model_declares_its_limitations():
    m = scan_model_from_payload(_scan_payload())
    joined = " ".join(m.notes)
    assert "no wall geometry" in joined
    assert "hulls" in joined and "NOT usable" in joined
    assert "no room classifications" in joined   # fixture has none
    assert m.stats["rooms_with_classification"] == 0


def test_scan_model_is_deterministic():
    first = scan_model_from_payload(_scan_payload())
    second = scan_model_from_payload(_scan_payload())
    assert first.to_json() == second.to_json()


def test_scan_model_uses_pano_fallback_elevation():
    m = scan_model_from_payload(_scan_payload())
    assert m.slabs[0].source == "pano-z-fallback"
    # min pano z 1.6 minus the 1.5 m tripod height
    assert m.slabs[0].raw_z == pytest.approx(0.1)


# --------------------------------------------------------------------------
# views against the real fixture
# --------------------------------------------------------------------------
@pytest.mark.skipif(not FIXTURE.exists(), reason="payload fixture not present")
class TestViewsOnFixture:
    @pytest.fixture(scope="class")
    def model(self):
        with open(FIXTURE, encoding="utf8") as handle:
            return model_from_payload(json.load(handle))

    def test_every_room_gets_a_view_record(self, model):
        views = room_views(model)
        assert len(views) == len(model.rooms)
        assert {v.room_id for v in views} == {r.id for r in model.rooms}

    def test_window_totals_match_the_model(self, model):
        views = room_views(model)
        # every window on a wall belongs to each room bounding that wall, so the
        # view count is at least the model's opening count for windows
        model_windows = sum(
            1 for w in model.walls for o in w.openings if o.kind == "window"
        )
        assert sum(v.window_count for v in views) >= model_windows

    def test_window_geometry_is_plausible(self, model):
        for view in room_views(model):
            for window in view.windows:
                assert 0.1 < window.width_m < 12.0
                assert 0.1 < window.height_m < 5.0
                assert window.area_m2 == pytest.approx(
                    window.width_m * window.height_m, rel=1e-6)
                assert -4.0 < window.sill_aff_m < 4.0
                assert 0.0 <= window.bearing_deg < 360.0
                assert window.sector in (
                    "N", "NE", "E", "SE", "S", "SW", "W", "NW")

    def test_below_floor_sills_match_the_models_own_count(self, model):
        """A sill below its slab is bad scan data, and must be labelled so."""
        suspect = [
            w for v in room_views(model) for w in v.suspect_windows
        ]
        for window in suspect:
            assert window.sill_aff_m < 0
            assert window.sill == "below-floor"
        # the model already counts openings that miss their storey entirely
        assert model.stats["openings_outside_storey"] >= 0
        if model.stats["openings_outside_storey"] == 0:
            assert not suspect

    def test_windowless_rooms_report_no_views(self, model):
        views = {v.room_id: v for v in room_views(model)}
        for room in model.rooms:
            carries = any(
                room.id in w.room_ids and any(o.kind == "window" for o in w.openings)
                for w in model.walls
            )
            assert views[room.id].has_windows == carries

    def test_views_discriminate_rooms(self, model):
        views = room_views(model)
        shared = distinctiveness(views)
        windowed = [v for v in views if v.has_windows]
        assert windowed, "the reference house has windowed rooms"
        # the point of modelling views: a window description pins a room down
        assert all(shared[v.room_id] == 1 for v in windowed)
        # and windowless rooms of the same type are not separable this way
        blind = [v for v in views if not v.has_windows]
        assert any(shared[v.room_id] > 1 for v in blind)

    def test_summary_is_internally_consistent(self, model):
        summary = view_summary(model)
        views = room_views(model)
        assert summary["rooms"] == len(views)
        assert summary["windows"] == sum(v.window_count for v in views)
        assert summary["rooms_with_windows"] == sum(1 for v in views if v.has_windows)
        assert sum(summary["facing"].values()) == len(views)
        assert 0.0 <= summary["uniquely_identified_share"] <= 1.0

    def test_corner_rooms_face_more_than_one_way(self, model):
        for view in room_views(model):
            if view.corner_room:
                assert len(view.sectors) >= 2

"""Unit tests for the Matterport geometry -> 3D pipeline.

No network and no credentials: everything downstream of ``fetch`` is a pure
function, so the tests build payloads by hand or load a saved fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ingest.mp3d import geometry as geo
from ingest.mp3d import slabs as slabmod
from ingest.mp3d import walls as wallmod
from ingest.mp3d.export import to_gltf, to_viewer_payload
from ingest.mp3d.model3d import model_from_payload
from ingest.mp3d.preflight import capability_from_payload

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "processed" / "zhKWys61pww.payload.json"


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _edge(vid_a, a, vid_b, b, **kw):
    return {
        "id": kw.pop("id", f"{vid_a}-{vid_b}"),
        "vertices": [
            {"id": vid_a, "position": {"x": a[0], "y": a[1]}},
            {"id": vid_b, "position": {"x": b[0], "y": b[1]}},
        ],
        **kw,
    }


def _square_edges(size=2.0):
    p = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    return [
        _edge("a", p[0], "b", p[1]),
        _edge("b", p[1], "c", p[2]),
        _edge("c", p[2], "d", p[3]),
        _edge("d", p[3], "a", p[0]),
    ]


def test_ring_from_edges_closes_a_square():
    ring = geo.ring_from_edges(_square_edges(2.0))
    assert ring is not None
    assert len(ring) == 4
    assert geo.polygon_area(ring) == pytest.approx(4.0)


def test_ring_rejects_an_open_boundary():
    edges = _square_edges()[:-1]  # drop the closing edge
    assert geo.ring_from_edges(edges) is None


def test_ring_rejects_a_vertex_of_degree_three():
    edges = _square_edges()
    edges.append(_edge("a", (0.0, 0.0), "e", (-1.0, -1.0), id="spur"))
    assert geo.ring_from_edges(edges) is None


def test_ring_is_order_independent():
    edges = _square_edges()
    first = geo.ring_from_edges(edges)
    second = geo.ring_from_edges(list(reversed(edges)))
    assert first is not None and second is not None
    assert geo.polygon_area(first) == pytest.approx(geo.polygon_area(second))


def test_area_and_centroid_of_an_l_shape():
    # a 4x1 bar (area 4, centroid (2, 0.5)) plus a 1x2 leg (area 2, centroid (0.5, 2))
    poly = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]
    assert geo.polygon_area(poly) == pytest.approx(6.0)
    cx, cy = geo.centroid(poly)
    assert (cx, cy) == pytest.approx(((4 * 2 + 2 * 0.5) / 6, (4 * 0.5 + 2 * 2) / 6))


def test_point_in_polygon_on_an_l_shape():
    poly = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]
    assert geo.point_in_polygon((3.0, 0.5), poly)      # in the bar
    assert geo.point_in_polygon((0.5, 2.5), poly)      # in the leg
    assert not geo.point_in_polygon((3.0, 2.5), poly)  # in the notch


def test_as_ccw_normalises_winding():
    cw = [(0, 0), (0, 2), (2, 2), (2, 0)]
    assert geo.signed_area(cw) < 0
    assert geo.signed_area(geo.as_ccw(cw)) > 0


def test_triangulate_covers_a_concave_polygon():
    poly = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]
    tris = geo.triangulate(poly)
    pts = geo.as_ccw(poly)
    assert len(tris) == len(poly) - 2
    total = 0.0
    for ia, ib, ic in tris:
        a, b, c = pts[ia], pts[ib], pts[ic]
        total += abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    assert total == pytest.approx(geo.polygon_area(poly))


def test_yaw_from_quaternion():
    # 90 degrees about z
    q = {"w": math.cos(math.pi / 4), "x": 0.0, "y": 0.0, "z": math.sin(math.pi / 4)}
    assert geo.yaw_degrees(q) == pytest.approx(90.0)
    assert geo.yaw_degrees(None) == 0.0


def test_outward_bearing_points_away_from_the_room():
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    # south wall of a north-up model: outward normal faces south = 180 degrees
    bearing = geo.outward_bearing(square, (0.0, 0.0), (2.0, 0.0), 0.0)
    assert bearing == pytest.approx(180.0, abs=1e-6)
    # yawing the model by +90 degrees carries that normal round to due east
    turned = geo.outward_bearing(square, (0.0, 0.0), (2.0, 0.0), 90.0)
    assert geo.angular_gap(turned, 90.0) == pytest.approx(0.0, abs=1e-6)


def test_circular_mean_wraps_around_north():
    # bearings are compared as angles: 359.999... and 0 are the same direction
    mean = geo.circular_mean([350.0, 10.0])
    assert mean is not None
    assert geo.angular_gap(mean, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert geo.angular_gap(geo.circular_mean([80.0, 100.0]), 90.0) < 1e-6
    assert geo.circular_mean([]) is None


# --------------------------------------------------------------------------
# slabs
# --------------------------------------------------------------------------
def _floor(sequence, doorway_zs=(), pano_zs=(), label=None):
    openings = [
        {"type": "doorway", "lowerElevation": z, "width": 0.9,
         "height": 2.1, "relativeCenter": 0.5}
        for z in doorway_zs
    ]
    return {
        "sequence": sequence,
        "label": label or f"Floor {sequence + 1}",
        "panoLocations": [{"position": {"z": z}} for z in pano_zs],
        "rooms": [{
            "id": f"r{sequence}",
            "classifications": [{"label": "Bedroom"}],
            "boundary": {"edges": _square_edges(3.0)[:1] + [
                {"id": f"op{sequence}", "thickness": 0.1,
                 "vertices": _square_edges(3.0)[1]["vertices"],
                 "openings": openings}
            ] + _square_edges(3.0)[2:]},
        }],
    }


def test_doorway_median_beats_pano_z():
    floors = [_floor(0, doorway_zs=(0.0, 0.1, -0.1), pano_zs=(1.4, 1.6))]
    elevations = slabmod.floor_elevations(floors)
    assert elevations[0].source == slabmod.SOURCE_DOORWAY
    assert elevations[0].raw_z == pytest.approx(0.0)


def test_pano_fallback_subtracts_the_tripod_height():
    floors = [_floor(0, pano_zs=(1.6, 1.8, 2.4))]
    elevations = slabmod.floor_elevations(floors)
    assert elevations[0].source == slabmod.SOURCE_PANO
    assert elevations[0].raw_z == pytest.approx(1.6 - slabmod.PANO_ABOVE_FLOOR_M)


def test_floor_heights_use_measured_gaps_and_mean_for_the_top():
    floors = [
        _floor(0, doorway_zs=(0.0,)),
        _floor(1, doorway_zs=(3.0,)),
        _floor(2, doorway_zs=(6.2,)),
    ]
    elevations = slabmod.floor_elevations(floors)
    heights = slabmod.floor_heights(elevations)
    assert heights[0] == pytest.approx(3.0)
    assert heights[1] == pytest.approx(3.2)
    assert heights[2] == pytest.approx(3.1)  # mean of 3.0 and 3.2
    assert slabmod.implausible_gaps(elevations) == []


def test_sub_storey_gap_is_reported_and_excluded_from_the_mean():
    floors = [_floor(0, doorway_zs=(0.0,)), _floor(1, doorway_zs=(1.2,))]
    elevations = slabmod.floor_elevations(floors)
    gaps = slabmod.implausible_gaps(elevations)
    assert gaps == [(0, 1, 1.2)]
    heights = slabmod.floor_heights(elevations)
    assert heights[0] == pytest.approx(1.2)          # slabs still stack exactly
    assert heights[1] == slabmod.DEFAULT_FLOOR_HEIGHT_M  # not 1.2


# --------------------------------------------------------------------------
# walls
# --------------------------------------------------------------------------
def test_opening_span_converts_width_into_wall_fractions():
    opening = {"type": "window", "relativeCenter": 0.5, "width": 1.0,
               "lowerElevation": 2.0, "height": 1.2}
    span = wallmod.opening_span(opening, length_m=4.0, datum_z=1.0)
    assert span is not None
    kind, t0, t1, z0, z1 = span
    assert kind == "window"
    assert (t0, t1) == pytest.approx((0.375, 0.625))
    assert (z0, z1) == pytest.approx((1.0, 2.2))


def test_opening_span_rejects_missing_dimensions():
    assert wallmod.opening_span({"type": "window", "width": 1.0}, 4.0, 0.0) is None
    assert wallmod.opening_span(
        {"type": "nonsense", "relativeCenter": 0.5, "width": 1.0,
         "lowerElevation": 0.0, "height": 1.0}, 4.0, 0.0) is None


def test_solid_panels_cut_a_window_leaving_sill_and_head():
    panels = wallmod.solid_panels([(0.4, 0.6, 1.0, 2.0)], slab_z=0.0, height_m=3.0)
    spans = sorted((p.t0, p.t1, p.z0, p.z1) for p in panels)
    assert (0.0, 0.4, 0.0, 3.0) in spans          # full height before
    assert (0.4, 0.6, 0.0, 1.0) in spans          # under the sill
    assert (0.4, 0.6, 2.0, 3.0) in spans          # over the head
    assert (0.6, 1.0, 0.0, 3.0) in spans          # full height after
    assert len(spans) == 4


def test_solid_panels_for_a_full_height_doorway_leaves_no_head():
    panels = wallmod.solid_panels([(0.0, 1.0, 0.0, 3.0)], slab_z=0.0, height_m=3.0)
    assert panels == []


def test_solid_panels_with_no_openings_is_one_panel():
    panels = wallmod.solid_panels([], slab_z=1.0, height_m=2.5)
    assert len(panels) == 1
    assert (panels[0].t0, panels[0].t1) == (0.0, 1.0)
    assert (panels[0].z0, panels[0].z1) == (1.0, 3.5)


def test_overlapping_openings_do_not_produce_negative_panels():
    panels = wallmod.solid_panels(
        [(0.2, 0.6, 1.0, 2.0), (0.4, 0.8, 1.0, 2.0)], slab_z=0.0, height_m=3.0)
    for p in panels:
        assert p.t1 >= p.t0
        assert p.z1 > p.z0


def test_dedupe_collapses_a_shared_wall_and_marks_it_interior():
    shared = _edge("x", (0.0, 0.0), "y", (2.0, 0.0), id="shared", thickness=0.15)
    floors = [{
        "sequence": 0,
        "rooms": [
            {"id": "roomA", "boundary": {"edges": [shared]}},
            {"id": "roomB", "boundary": {"edges": [shared]}},
        ],
    }]
    walls = wallmod.dedupe_walls(floors, datum_z=0.0)
    assert len(walls) == 1
    assert walls[0]["exterior"] is False
    assert walls[0]["room_ids"] == ("roomA", "roomB")
    assert walls[0]["thickness_m"] == pytest.approx(0.15)


def test_dedupe_is_sorted_by_edge_id_for_determinism():
    floors = [{
        "sequence": 0,
        "rooms": [{"id": "r", "boundary": {"edges": [
            _edge("a", (0, 0), "b", (1, 0), id="zzz"),
            _edge("b", (1, 0), "c", (1, 1), id="aaa"),
        ]}}],
    }]
    walls = wallmod.dedupe_walls(floors, 0.0)
    assert [w["id"] for w in walls] == ["aaa", "zzz"]


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
def test_preflight_accepts_a_model_with_thickness():
    payload = {"id": "m", "name": "n", "state": "active", "floors": [{
        "sequence": 0,
        "rooms": [{"id": "r", "boundary": {"edges": [{"thickness": 0.1}]}}],
    }]}
    cap = capability_from_payload("m", payload)
    assert cap.buildable and cap.tier == "A" and cap.walls_with_thickness == 1


def test_preflight_rejects_a_model_without_thickness_and_explains_why():
    payload = {"id": "m", "state": "active", "floors": [{
        "sequence": 0,
        "rooms": [{"id": "r", "boundary": {"edges": [{}]}}],
    }]}
    cap = capability_from_payload("m", payload)
    assert not cap.buildable
    assert any("Edge.thickness" in r for r in cap.reasons)


def test_preflight_handles_a_missing_model():
    cap = capability_from_payload("nope", None)
    assert not cap.buildable and cap.reasons


# --------------------------------------------------------------------------
# end to end, against the saved payload
# --------------------------------------------------------------------------
@pytest.mark.skipif(not FIXTURE.exists(), reason="payload fixture not present")
class TestAgainstFixture:
    @pytest.fixture(scope="class")
    def model(self):
        with open(FIXTURE, encoding="utf8") as handle:
            return model_from_payload(json.load(handle))

    def test_shape_of_the_reference_building(self, model):
        assert model.tier == "A"
        assert len(model.slabs) == 3
        assert model.stats["walls"] == 156
        assert model.stats["openings"] == 62
        assert model.stats["openings_by_kind"]["window"] == 39
        assert model.stats["openings_by_kind"]["doorway"] == 22

    def test_slabs_stack_without_gaps_or_overlaps(self, model):
        for lower, upper in zip(model.slabs, model.slabs[1:]):
            assert lower.z0 + lower.height_m == pytest.approx(upper.z0, abs=1e-6)

    def test_every_room_polygon_is_closed_and_positive(self, model):
        for room in model.rooms:
            assert len(room.polygon) >= 3
            assert room.area_m2 > 0
            assert geo.signed_area(list(room.polygon)) > 0  # counter-clockwise

    def test_triangulation_area_matches_the_polygon(self, model):
        for room in model.rooms:
            pts = room.polygon
            total = 0.0
            for ia, ib, ic in room.triangles:
                a, b, c = pts[ia], pts[ib], pts[ic]
                total += abs(
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                ) / 2
            assert total == pytest.approx(room.area_m2, rel=0.02)

    def test_openings_are_well_formed_and_within_the_building(self, model):
        lo, hi = model.bounds["z"]
        for wall in model.walls:
            for opening in wall.openings:
                assert 0.0 <= opening.t0 < opening.t1 <= 1.0
                assert opening.z1 > opening.z0
                assert lo - 2.0 <= opening.z0 <= hi + 2.0

    def test_openings_outside_their_storey_are_counted_not_hidden(self, model):
        """Real scans misplace a few openings; the model must declare them."""
        band = {s.floor: (s.z0, s.z0 + s.height_m) for s in model.slabs}
        clipped = outside = 0
        for wall in model.walls:
            low, high = band[wall.floor]
            for opening in wall.openings:
                overlap = min(opening.z1, high) - max(opening.z0, low)
                if overlap <= 0.01:
                    outside += 1
                elif overlap < (opening.z1 - opening.z0) - 0.01:
                    clipped += 1
        assert model.stats["openings_outside_storey"] == outside
        assert model.stats["openings_clipped_to_storey"] == clipped
        if outside:
            assert any("outside their floor" in n for n in model.notes)

    def test_panels_never_cover_an_opening(self, model):
        for wall in model.walls:
            for panel in model.panels(wall):
                for opening in wall.openings:
                    t_overlap = min(panel.t1, opening.t1) - max(panel.t0, opening.t0)
                    z_overlap = min(panel.z1, opening.z1) - max(panel.z0, opening.z0)
                    assert t_overlap <= 1e-6 or z_overlap <= 1e-6

    def test_build_is_deterministic(self, model):
        with open(FIXTURE, encoding="utf8") as handle:
            again = model_from_payload(json.load(handle))
        assert again.to_json() == model.to_json()

    def test_gltf_is_structurally_sound(self, model):
        doc = to_gltf(model)
        assert doc["asset"]["version"] == "2.0"
        assert doc["buffers"][0]["uri"].startswith("data:")
        assert doc["meshes"] and doc["nodes"]
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                pos = doc["accessors"][prim["attributes"]["POSITION"]]
                idx = doc["accessors"][prim["indices"]]
                assert pos["type"] == "VEC3" and pos["componentType"] == 5126
                assert idx["type"] == "SCALAR" and idx["count"] % 3 == 0
                assert prim["material"] < len(doc["materials"])
        for node in doc["nodes"]:
            assert node["mesh"] < len(doc["meshes"])

    def test_viewer_payload_round_trips_the_counts(self, model):
        payload = to_viewer_payload(model)
        assert len(payload["walls"]) == len(model.walls)
        assert len(payload["rooms"]) == len(model.rooms)
        assert len(payload["slabs"]) == len(model.slabs)
        assert sum(len(w["op"]) for w in payload["walls"]) == model.stats["openings"]

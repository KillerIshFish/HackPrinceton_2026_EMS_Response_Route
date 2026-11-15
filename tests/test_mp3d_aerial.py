"""Tests for the aerial sidecar.

Pure: no network and no credentials. The Overpass response for the reference
house is a saved fixture, and Vexcel metadata is a hand-built payload - the
imagery bytes are deliberately not fixtured, since the sidecar exists precisely
because they are time-varying.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.mp3d.aerial import (
    SOURCE_ENTRANCE,
    SOURCE_OSM,
    SOURCE_UNKNOWN,
    AerialRef,
    RoadCandidate,
    StreetBearing,
    angular_gap,
    bearing_between,
    coverage_report,
    load_sidecar,
    overpass_query,
    point_to_segment,
    point_wkt,
    road_candidates_from_payload,
    reconcile_bearings,
    save_sidecar,
    square_wkt,
    street_bearing_from_roads,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OSM_FIXTURE = FIXTURES / "osm_roads_zhKWys61pww.json"

# the reference house
LAT, LON = 29.810283, -95.4158103
ENTRANCE_BEARING = 301.9


# --------------------------------------------------------------------------
# WKT conventions - the documented number-one bug is transposing lon/lat
# --------------------------------------------------------------------------
def test_point_wkt_puts_longitude_first():
    assert point_wkt(LAT, LON) == f"POINT({LON} {LAT})"
    # a Houston point must have a negative first ordinate
    assert float(point_wkt(LAT, LON).split("(")[1].split()[0]) < 0


def test_square_wkt_is_closed_and_centred():
    wkt = square_wkt(LAT, LON, 100.0)
    coords = wkt[len("POLYGON(("):-2].split(",")
    pairs = [tuple(float(v) for v in c.split()) for c in coords]
    assert len(pairs) == 5
    assert pairs[0] == pairs[-1]                      # closed ring
    lons = [p[0] for p in pairs]
    lats = [p[1] for p in pairs]
    assert min(lons) < LON < max(lons)
    assert min(lats) < LAT < max(lats)
    # ~200 m across
    span_m = (max(lats) - min(lats)) * 110540.0
    assert span_m == pytest.approx(200.0, rel=0.02)


# --------------------------------------------------------------------------
# bearings and distances
# --------------------------------------------------------------------------
def test_bearing_cardinals():
    assert bearing_between(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=0.01)
    assert bearing_between(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=0.01)
    assert bearing_between(0.0, 0.0, -1.0, 0.0) == pytest.approx(180.0, abs=0.01)
    assert bearing_between(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0, abs=0.01)


def test_angular_gap_wraps():
    assert angular_gap(350.0, 10.0) == pytest.approx(20.0)
    assert angular_gap(10.0, 350.0) == pytest.approx(20.0)
    assert angular_gap(0.0, 180.0) == pytest.approx(180.0)


def test_point_to_segment_clamps_to_the_endpoints():
    # segment running east from the origin point
    dist, clat, clon = point_to_segment(LAT, LON, LAT, LON, LAT, LON + 0.001)
    assert dist == pytest.approx(0.0, abs=0.5)
    # a point north of the segment measures the perpendicular
    dist, _, _ = point_to_segment(LAT + 0.001, LON, LAT, LON, LAT, LON + 0.001)
    assert dist == pytest.approx(0.001 * 110540.0, rel=0.02)


def test_overpass_query_is_bounded_and_filtered():
    q = overpass_query(LAT, LON, 150)
    assert "around:150" in q
    assert "highway" in q
    assert str(LAT) in q and str(LON) in q


# --------------------------------------------------------------------------
# road selection, against the saved fixture
# --------------------------------------------------------------------------
@pytest.mark.skipif(not OSM_FIXTURE.exists(), reason="OSM fixture not present")
class TestAgainstOsmFixture:
    @pytest.fixture(scope="class")
    def candidates(self):
        payload = json.loads(OSM_FIXTURE.read_text(encoding="utf8"))
        return road_candidates_from_payload(payload, LAT, LON)

    def test_fixture_yields_candidates(self, candidates):
        assert candidates
        assert all(0.0 <= c.bearing_deg < 360.0 for c in candidates)
        assert all(c.distance_m >= 0 for c in candidates)

    def test_a_service_drive_is_the_physically_nearest(self, candidates):
        nearest = min(candidates, key=lambda c: c.distance_m)
        assert nearest.is_service
        assert nearest.distance_m < 20.0

    def test_selection_prefers_a_named_street_over_a_closer_drive(self, candidates):
        """The gated-complex case: do not call a driveway the street."""
        chosen = street_bearing_from_roads(candidates)
        assert chosen.bearing_deg is not None
        assert chosen.source == SOURCE_OSM
        assert chosen.confidence == "high"
        assert not chosen.ambiguous
        assert "West 26th Street" in chosen.note
        # due south, and decisively not the 359 deg driveway
        assert chosen.bearing_deg == pytest.approx(178.8, abs=1.0)

    def test_candidates_are_rank_then_distance_ordered(self, candidates):
        keys = [(c.class_rank, c.distance_m) for c in candidates]
        assert keys == sorted(keys)

    def test_reconciliation_separates_approach_from_frontage(self, candidates):
        """A 123 deg gap here is two products, not a contradiction."""
        osm = street_bearing_from_roads(candidates)
        verdict = reconcile_bearings(osm, ENTRANCE_BEARING)
        assert verdict["gap_deg"] == pytest.approx(123.1, abs=1.0)
        assert verdict["approach_bearing_deg"] == pytest.approx(178.8, abs=1.0)
        assert verdict["frontage_bearing_deg"] == ENTRANCE_BEARING
        assert verdict["internal_drive_m"] is not None
        assert verdict["internal_drive_m"] < 20.0
        # the caller-facing product stays the Entrance bearing
        assert verdict["recommended_source"] == SOURCE_ENTRANCE
        assert verdict["recommended_bearing_deg"] == ENTRANCE_BEARING
        assert "not a conflict" in verdict["verdict"]


# --------------------------------------------------------------------------
# selection logic, on hand-built candidates
# --------------------------------------------------------------------------
def _road(rank, dist, bearing, name="", highway="residential", wid=1):
    return RoadCandidate(way_id=wid, name=name, highway=highway,
                         class_rank=rank, distance_m=dist, bearing_deg=bearing)


def test_no_roads_gives_no_bearing():
    out = street_bearing_from_roads([])
    assert out.bearing_deg is None
    assert out.confidence == "none"


def test_service_only_is_low_confidence():
    out = street_bearing_from_roads([_road(8, 12.0, 90.0, highway="service")])
    assert out.bearing_deg == 90.0
    assert out.confidence == "low"
    assert "service road" in out.note


def test_two_comparable_streets_pointing_apart_are_ambiguous():
    out = street_bearing_from_roads([
        _road(5, 40.0, 0.0, name="A St", wid=1),
        _road(5, 45.0, 180.0, name="B St", wid=2),
    ])
    assert out.ambiguous
    assert out.confidence == "medium"
    assert "point elsewhere" in out.note


def test_comparable_streets_pointing_the_same_way_are_not_ambiguous():
    out = street_bearing_from_roads([
        _road(5, 40.0, 10.0, name="A St", wid=1),
        _road(5, 45.0, 20.0, name="B St", wid=2),
    ])
    assert not out.ambiguous
    assert out.confidence == "high"


def test_named_road_beats_unnamed_of_the_same_class():
    out = street_bearing_from_roads([
        _road(5, 40.0, 10.0, name="", wid=1),
        _road(5, 44.0, 12.0, name="Main St", wid=2),
    ])
    assert "Main St" in out.note


def test_higher_class_beats_closer_lower_class():
    out = street_bearing_from_roads([
        _road(8, 5.0, 270.0, highway="service", wid=1),
        _road(2, 90.0, 90.0, name="Big Rd", highway="primary", wid=2),
    ])
    assert out.bearing_deg == 90.0
    assert "Big Rd" in out.note


# --------------------------------------------------------------------------
# reconciliation branches
# --------------------------------------------------------------------------
def _osm(bearing, ambiguous=False, candidates=()):
    return StreetBearing(bearing, SOURCE_OSM, "high", ambiguous, "", tuple(candidates))


def test_reconcile_agreement():
    v = reconcile_bearings(_osm(300.0), 310.0)
    assert v["gap_deg"] == pytest.approx(10.0)
    assert v["recommended_source"] == SOURCE_OSM
    assert "agree" in v["verdict"]


def test_reconcile_opposite_refuses_to_choose():
    v = reconcile_bearings(_osm(0.0), 180.0)
    assert v["recommended_source"] == SOURCE_UNKNOWN
    assert v["recommended_bearing_deg"] is None
    assert "opposite" in v["verdict"]


def test_reconcile_without_entrance_uses_osm():
    v = reconcile_bearings(_osm(123.0), None)
    assert v["recommended_source"] == SOURCE_OSM
    assert v["recommended_bearing_deg"] == 123.0


def test_reconcile_without_osm_uses_entrance():
    v = reconcile_bearings(StreetBearing(None, SOURCE_OSM, "none"), 42.0)
    assert v["recommended_source"] == SOURCE_ENTRANCE
    assert v["recommended_bearing_deg"] == 42.0


def test_reconcile_with_neither():
    v = reconcile_bearings(StreetBearing(None, SOURCE_OSM, "none"), None)
    assert v["recommended_source"] == SOURCE_UNKNOWN
    assert "neither" in v["verdict"]


def test_ambiguous_osm_is_not_recommended_even_when_it_agrees():
    v = reconcile_bearings(_osm(300.0, ambiguous=True), 310.0)
    assert v["recommended_source"] == SOURCE_UNKNOWN
    assert "ambiguous" in v["verdict"]


# --------------------------------------------------------------------------
# coverage ceiling
# --------------------------------------------------------------------------
def test_coverage_report_counts_missing_geolocation():
    models = [
        {"id": "a", "geolocation": {"lat": "1.0", "long": "2.0"}},
        {"id": "b", "geolocation": {}},
        {"id": "c"},
        {"id": "d", "geolocation": {"lat": "1.0", "long": None}},
    ]
    report = coverage_report(models)
    assert report["models"] == 4
    assert report["with_geolocation"] == 1
    assert report["without_geolocation"] == 3
    assert report["coverage_share"] == pytest.approx(0.25)
    assert set(report["missing_ids"]) == {"b", "c", "d"}


def test_coverage_report_on_an_empty_corpus():
    report = coverage_report([])
    assert report["models"] == 0
    assert report["coverage_share"] == 0.0


# --------------------------------------------------------------------------
# sidecar round-trip - and it must stay out of the deterministic model
# --------------------------------------------------------------------------
def test_sidecar_round_trip(tmp_path):
    ref = AerialRef(
        model_id="m1", provider="vexcel", lat=LAT, lng=LON, yaw_deg=-95.31,
        zoom=None, bbox=(-1.0, 2.0, 3.0, 4.0), captured_at="2026-01-29T20:29:30",
        content_sha256="deadbeef", local_path="x.jpg",
        street_bearing_deg=178.8, bearing_source=SOURCE_OSM,
        road_candidates=(_road(5, 50.0, 178.8, name="West 26th Street"),),
        oblique_directions=("oblique-south",),
    )
    save_sidecar(ref, tmp_path)
    again = load_sidecar("m1", tmp_path)
    assert again is not None
    assert again.model_id == "m1"
    assert again.street_bearing_deg == pytest.approx(178.8)
    assert again.bbox == (-1.0, 2.0, 3.0, 4.0)
    assert again.oblique_directions == ("oblique-south",)
    assert again.road_candidates[0].name == "West 26th Street"
    assert again.to_json() == ref.to_json()


def test_missing_sidecar_is_none_not_an_error(tmp_path):
    assert load_sidecar("nope", tmp_path) is None


def test_aerial_is_not_part_of_the_model_payload():
    """The determinism guarantee: Model3D must not carry time-varying tiles."""
    import json as _json
    from ingest.mp3d.model3d import Model3D

    fields = set(Model3D.__dataclass_fields__)
    for leaked in ("aerial", "tile", "imagery", "captured_at", "content_sha256"):
        assert leaked not in fields
    # and the viewer payload shape is separate too
    assert "AerialRef" not in _json.dumps(sorted(fields))

"""Tests for the unified pipeline (ingest/unified): roster, archetype layout, lifts, projections.

No network and no dataset: property facts are written inline, and the Track A lift is
exercised on a small Matterport-shaped payload. The corpus tests at the bottom run only
when the CoStar kit is on disk.

Run: uv run pytest tests/test_unified.py
"""

from __future__ import annotations

import networkx as nx
import pytest

from core.building_graph import build_graph
from core.schema import Building
from ingest.costar import kit_dir
from ingest.unified import index
from ingest.unified.layout import choose_archetype
from ingest.unified.lift import from_building, from_model3d
from ingest.unified.pipeline import build_property_model
from ingest.unified.project import to_building, viewer_payload
from ingest.unified.roster import build_roster, infer_scheme, parse_number
from ingest.unified.stamp import make_interior


def _facts(**over) -> dict:
    units = [{"id": str(100 + i), "number": f"{floor}{n:02d}", "building": "", "type": "A1", "beds": 1, "baths": 1,
              "area_m2": 62.0, "scans": []} for i, (floor, n) in enumerate((f, n) for f in (1, 2, 3) for n in range(1, 9))]
    base = {"property_id": 42, "name": "Test Court", "address": "1 Test St, Testville, TS", "latlng": [34.0, -118.0],
            "use_type": "apartment", "style": "mid_rise", "stories": 3, "gross_m2": 2400.0, "typical_floor_m2": 800.0,
            "floor_area_m2": {}, "storey_height_m": None, "num_buildings": 1, "total_units": 24, "has_elevator": True,
            "units": units, "tenants": [], "property_scans": []}
    base.update(over)
    return base


def test_address_normalisation() -> None:
    assert index.normalize("11900 West Olympic Boulevard, Los Angeles") == "11900 w olympic blvd los angeles"
    assert index.normalize("709 S. Mariposa Avenue") == "709 s mariposa ave"


def test_unit_numbers_give_floors_in_several_schemes() -> None:
    assert parse_number("709 S. Mariposa Avenue - #105") == ("", 105, "105")
    assert parse_number("01-0317") == ("1", 317, "01-0317")
    assert infer_scheme([101, 204, 305, 412], stories=4, num_buildings=1) == "floor_hundreds"
    assert infer_scheme([7156, 3092, 2110, 5158], stories=7, num_buildings=1) == "floor_first_digit"
    assert infer_scheme([1807, 1208, 414, 2304], stories=2, num_buildings=23) == "building_hundreds"
    assert infer_scheme([1, 2, 3, 13], stories=16, num_buildings=1) == "sequential"

    roster = build_roster(_facts())
    assert roster["scheme"] == "floor_hundreds" and roster["stories"] == 3
    assert {s["label"]: s["floor"] for s in roster["slots"]}["305"] == 3
    assert all(s["floor_source"] == "number" for s in roster["slots"])

    # numbers that carry no floor are spread over the floors and say so
    plain = _facts(units=[{**u, "number": str(i + 1)} for i, u in enumerate(_facts()["units"])])
    spread = build_roster(plain)
    assert spread["scheme"] == "sequential"
    assert {s["floor"] for s in spread["slots"]} == {1, 2, 3}
    assert all(s["floor_source"] == "spread" for s in spread["slots"])


def test_archetype_follows_use_height_and_elevator() -> None:
    def archetype(**over) -> str:
        facts = _facts(**over)
        return choose_archetype(facts, build_roster(facts))

    assert archetype() == "double_loaded_corridor"
    assert archetype(style="garden", stories=2, has_elevator=False) == "garden_walkup"
    assert archetype(stories=12) == "central_core"
    assert archetype(units=[], total_units=0) == "floor_stack"
    assert archetype(use_type="retail", stories=1, units=[], tenants=[
        {"floor": 1, "suite": "A", "tenant": "Shop", "area_m2": 180.0, "id": "1"}]) == "bays"


def test_generated_building_is_deterministic_connected_and_queryable() -> None:
    bm = build_property_model(_facts())
    assert bm.to_json() == build_property_model(_facts()).to_json(), "same record, same bytes"
    assert bm.tier == "template" and bm.archetype == "double_loaded_corridor"
    assert [lv.index for lv in bm.levels] == [1, 2, 3] and bm.levels[1].z0 == 3.0
    units = [s for s in bm.spaces if s.kind == "unit"]
    assert len(units) == 24 and {s.level for s in units} == {1, 2, 3}
    assert any(s.kind == "elevator" for s in bm.spaces) and sum(s.kind == "stair" for s in bm.spaces) == 6
    # every unit has a door onto the corridor and a window on its facade
    for u in units:
        mine = [w for w in bm.walls if u.id in w.space_ids]
        assert any(o.kind == "doorway" for w in mine for o in w.openings), u.label
        assert any(o.kind == "window" and w.exterior for w in mine for o in w.openings), u.label
    # stairs and the elevator are template links between consecutive floors, never scan evidence
    vertical = [ln for ln in bm.links if ln.floors]
    assert vertical and all(ln.evidence == "template" for ln in vertical)

    b = to_building(bm, {"geolocation": [34.0, -118.0], "yaw_deg": 0.0, "street": {"bearing_deg": 180.0}})
    Building.model_validate(b.model_dump())
    assert nx.is_connected(build_graph(b))
    assert sum(r.room_type == "unit" for r in b.rooms) == 24, "a unit is one room, whatever its interior"
    assert b.entrances and b.staging_spots
    # the front row (low y) faces the street bearing of the sidecar; the back row is not labelled
    front = [r for r in b.rooms if r.room_type == "unit" and r.centroid[1] < 9.0]
    back = [r for r in b.rooms if r.room_type == "unit" and r.centroid[1] > 11.0]
    assert front and all(r.window_facing == "street" for r in front)
    assert back and all(r.window_facing is None and r.has_windows for r in back)

    payload = viewer_payload(bm)
    assert payload["counts"]["spaces"] == len(bm.spaces) and payload["walls"]
    by_space = {s["id"]: s for s in payload["spaces"]}
    child = next(s for s in payload["spaces"] if s["par"])
    assert child["r"] == child["par"] and by_space[child["par"]]["k"] == "unit", "a child room counts toward its unit"
    # a wall with a door is cut into panels around the opening
    assert any(len(w["p"]) >= 2 for w in payload["walls"])


def _scan_payload() -> dict:
    """Two rooms side by side sharing a wall with a doorway; an exterior door and a window. Metres."""

    def edge(eid, a, b, openings=()):
        return {"id": eid, "type": "wall", "thickness": 0.12, "openings": list(openings),
                "vertices": [{"id": f"v{a}", "position": {"x": a[0], "y": a[1]}}, {"id": f"v{b}", "position": {"x": b[0], "y": b[1]}}]}

    door = {"id": "o1", "type": "doorway", "width": 0.9, "height": 2.0, "lowerElevation": 0.0, "relativeCenter": 0.5}
    front = {"id": "o2", "type": "doorway", "width": 0.9, "height": 2.0, "lowerElevation": 0.0, "relativeCenter": 0.5}
    window = {"id": "o3", "type": "window", "width": 1.2, "height": 1.2, "lowerElevation": 0.9, "relativeCenter": 0.5}
    shared = edge("e-shared", (4, 0), (4, 3), [door])
    rooms = [
        {"id": "roomA", "label": "", "classifications": [{"label": "Entrance"}], "boundary": {"edges": [
            edge("a1", (0, 0), (4, 0), [front]), shared, edge("a3", (4, 3), (0, 3)), edge("a4", (0, 3), (0, 0))]}},
        {"id": "roomB", "label": "", "classifications": [{"label": "Bedroom"}], "boundary": {"edges": [
            edge("b1", (4, 0), (8, 0)), edge("b2", (8, 0), (8, 3), [window]), edge("b3", (8, 3), (4, 3)), shared]}},
    ]
    return {"id": "TESTSCAN", "name": "Test unit", "floors": [{"id": "f0", "label": "Floor 1", "sequence": 0, "rooms": rooms}],
            "geocoordinates": {"rotation": {"z": 0.0, "w": 1.0}}}


def _test_model():
    from ingest.mp3d.model3d import model_from_payload

    try:
        model = model_from_payload(_scan_payload(), "TESTSCAN")
    except Exception as exc:  # noqa: BLE001 - the payload shape is Track A's to define
        pytest.skip(f"Track A could not build the inline payload: {exc}")
    if len(model.rooms) != 2:
        pytest.skip("Track A dropped a room of the inline payload")
    return model


def test_surveyed_scan_lifts_one_to_one() -> None:
    model = _test_model()
    bm = from_model3d(model)
    assert bm.tier.startswith("surveyed") and bm.archetype == "whole_scan"
    assert {s.fidelity for s in bm.spaces} == {"surveyed"} and len(bm.walls) == len(model.walls)
    # the doorway the two rooms share is a link backed by the scan's own opening
    assert [(ln.a, ln.b, ln.evidence) for ln in bm.links] == [("roomA", "roomB", "opening")]
    assert bm.entrances and bm.entrances[0].space_id == "roomA"
    assert bm.to_json() == from_model3d(_test_model()).to_json()
    b = to_building(bm)
    assert nx.is_connected(build_graph(b)) and {r.room_type for r in b.rooms} == {"lobby", "office"}
    assert next(r for r in b.rooms if r.id == "roomB").has_windows


def test_scanned_unit_type_is_stamped_into_every_unit_of_the_type() -> None:
    model = _test_model()
    facts = _facts()
    for u in facts["units"][:6]:
        u["scans"] = ["TESTSCAN"]
    bm = build_property_model(facts, {"TESTSCAN": model})
    assert bm.tier == "template+stamped" and bm.stats["units_with_stamped_interior"] == 6
    stamped = [s for s in bm.spaces if s.fidelity == "stamped"]
    assert len(stamped) == 12 and {s.raw_class for s in stamped} == {"Entrance", "Bedroom"}
    units = {s.id: s for s in bm.spaces if s.kind == "unit"}
    for s in stamped:  # every stamped room lies inside its unit's rectangle
        xs = [p[0] for p in units[s.parent].polygon]
        ys = [p[1] for p in units[s.parent].polygon]
        assert all(min(xs) - 0.01 <= x <= max(xs) + 0.01 and min(ys) - 0.01 <= y <= max(ys) + 0.01 for x, y in s.polygon)
    # the other units of the building keep the generic bed / bath interior
    assert bm.stats["units_with_generic_interior"] == 18
    assert make_interior({})({"scans": ["missing"], "id": "u"}, (0, 0, 5, 9), "N", 1, 0.0, 3.0, None) is None
    assert nx.is_connected(build_graph(to_building(bm)))


def test_legacy_app_building_lifts_for_the_viewer() -> None:
    from ingest.matterport import load_building

    bm = from_building(load_building("ptc-100"))
    payload = viewer_payload(bm)
    assert len(payload["spaces"]) == len(load_building("ptc-100").rooms) and payload["walls"]
    assert all(s["r"] == s["id"] for s in payload["spaces"])


# ---------------------------------------------------------------------------
# With the CoStar kit on disk
# ---------------------------------------------------------------------------

needs_kit = pytest.mark.skipif(not (kit_dir() / "data" / "commercial-property.jsonl").is_file(),
                               reason="CoStar kit not in data/raw/costar")


@needs_kit
def test_lookup_finds_a_property_by_address_name_and_id() -> None:
    import time

    index.connect()
    for query in ("11900 W Olympic Blvd", "11900 West Olympic Boulevard, Los Angeles, CA 90064", "bundy olympic", "246948"):
        started = time.perf_counter()
        hits = index.lookup(query)
        assert hits and hits[0]["property_id"] == 246948, query
        assert time.perf_counter() - started < 0.25, "lookup is an indexed query, not a file scan"
    assert index.get_record(246948)["id"] == 246948


@needs_kit
def test_every_sampled_property_builds_offline() -> None:
    from ingest.unified.record import property_facts

    ids = index.all_property_ids()[::60]
    assert len(ids) > 40
    for pid in ids:
        bm = build_property_model(property_facts(pid))
        b = to_building(bm)
        assert b.rooms and b.entrances, pid
        assert nx.is_connected(build_graph(b)), pid

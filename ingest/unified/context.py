"""Layer 6: the site-context sidecar (street bearing, placement, imagery window).

Kept apart from ``BuildingModel`` on purpose: road data and imagery change over
time, and the model has to stay byte-identical for the same inputs. The sidecar is
one JSON file per building id under ``data/processed/context/``.

What it records
---------------
``geolocation``   the property's (or the scan's) coordinates
``street``        which way the street lies from the building, from OSM centrelines
                  via Track A's ``ingest.mp3d.aerial`` (service roads ranked last)
``yaw_deg``       how the model is turned on the ground. A scan carries its own yaw.
                  A generated building has none, so its front (-y) is turned to face
                  the street; with no street bearing it stays unrotated.
``footprint_latlng`` / ``imagery``   the model's outline on the map and the Vexcel
                  tile window the web app should request for the approach panel

The only network call is one OSM Overpass request carrying the coordinates. Overpass
can be slow (Track A retries it with backoff), and a call in progress cannot wait on
it: ``build_context`` gives it ``deadline_s`` seconds, then carries on without a street
bearing. The request keeps running in the background and fills the sidecar in when it
returns, so the next build of the same building has it.

Motorways and trunk roads are left out when choosing the street: a building does not
front a freeway, however close one runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from ingest.unified.schema import BuildingModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "data" / "processed" / "context"
STREET_DEADLINE_S = 4.0
NOT_A_FRONTAGE = ("motorway", "motorway_link", "trunk", "trunk_link")


def _street_from_osm(lat: float, lon: float) -> tuple[Optional[dict], Optional[str]]:
    """``(street, error)`` from OSM centrelines, ignoring limited-access roads."""
    from ingest.mp3d.aerial import fetch_osm_roads, road_candidates_from_payload, street_bearing_from_roads

    candidates = [c for c in road_candidates_from_payload(fetch_osm_roads(lat, lon), lat, lon)
                  if c.highway not in NOT_A_FRONTAGE]
    sb = street_bearing_from_roads(candidates)
    if sb.bearing_deg is None:
        return None, sb.note or "no street found"
    first = next((c for c in sb.candidates if abs(c.bearing_deg - sb.bearing_deg) < 1e-6), None)
    return {"bearing_deg": round(float(sb.bearing_deg), 1), "source": sb.source, "confidence": sb.confidence,
            "ambiguous": bool(sb.ambiguous), "note": sb.note, "name": first.name if first else "",
            "distance_m": round(first.distance_m, 1) if first else None}, None


def context_path(building_id: str) -> Path:
    return CONTEXT_DIR / f"{building_id}.json"


def load_context(building_id: str) -> Optional[dict]:
    path = context_path(building_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def build_context(bm: BuildingModel, geolocation=None, network: bool = True, refresh: bool = False,
                  deadline_s: float = STREET_DEADLINE_S) -> dict:
    """Build (or reuse) the sidecar for a model and write it to disk."""
    cached = None if refresh else load_context(bm.id)
    if cached is not None and (cached.get("street") or not network):
        return cached
    geo = list(geolocation or bm.geolocation or []) or None
    street: Optional[dict] = None
    error = None
    if network and geo:
        box: dict = {}

        def fetch() -> None:
            try:
                box["street"], box["error"] = _street_from_osm(float(geo[0]), float(geo[1]))
            except Exception as exc:  # noqa: BLE001 - the sidecar is optional
                box["street"], box["error"] = None, str(exc)[:160]
            if box.get("late") and box.get("street"):  # arrived after the deadline: fill the sidecar in now
                _assemble(bm, geo, box["street"], None)

        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        worker.join(deadline_s)
        if worker.is_alive():
            box["late"] = True
            error = f"street lookup still running after {deadline_s:.0f} s; continuing without it"
        else:
            street, error = box.get("street"), box.get("error")
    return _assemble(bm, geo, street, error)


def _assemble(bm: BuildingModel, geo, street: Optional[dict], error: Optional[str]) -> dict:
    scan = bm.archetype in ("whole_scan", "scan_clusters")
    if scan:
        yaw, yaw_source = bm.yaw_deg, "scan"
    elif street:
        # the model's front is -y; turn it so -y points along the street bearing
        yaw, yaw_source = round((180.0 - street["bearing_deg"] + 180.0) % 360.0 - 180.0, 1), "front faces the street"
    else:
        yaw, yaw_source = 0.0, "unknown"
    context = {"id": bm.id, "geolocation": geo, "street": street, "street_error": error, "yaw_deg": yaw,
               "yaw_source": yaw_source}
    from ingest.unified.project import _bounds, placement, to_latlng

    place = placement(bm, context)
    x0, y0, x1, y1 = _bounds(bm)
    context["footprint_latlng"] = [to_latlng(place, x, y) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))] if geo else []
    context["centre_latlng"] = to_latlng(place, (x0 + x1) / 2, (y0 + y1) / 2) if geo else None
    radius = max(45.0, min(160.0, 0.75 * max(x1 - x0, y1 - y0) + 25.0))
    context["imagery"] = {"layer": "urban", "zoom": 20 if radius > 60 else 21, "radius_m": round(radius)}
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    context_path(bm.id).write_text(json.dumps(context, indent=1), encoding="utf-8")
    return context

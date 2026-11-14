"""Layer 6: the site-context sidecar (where the model sits on the ground, and which way it faces).

Kept apart from ``BuildingModel`` on purpose: road data and imagery change over
time, and the model has to stay byte-identical for the same inputs. The sidecar is
one JSON file per building id under ``data/processed/context/``.

What it records
---------------
``geolocation``     the property's (or the scan's) coordinates: an address point, which
                    is often at the kerb or the parcel centre, not the building's middle
``building_outline`` the real building outline(s) on the property and their oriented
                    bounding box. From Vexcel's property data (the structures on the
                    parcel, newest imagery vintage; about a third of a second), else
                    from OpenStreetMap
``street``          which way the street lies from the building, from OSM centrelines
                    via Track A's ``ingest.mp3d.aerial`` (service roads ranked last;
                    motorways and trunk roads left out: no building fronts a freeway)
``anchor_latlng`` / ``yaw_deg`` / ``placement_source``
                    where the model's centre goes and how it is turned:

                    * a scan keeps its own position and yaw;
                    * a generated building is centred on the real outline and turned so
                      its long side lies along the outline's long side. Of the two ways
                      round, the front goes toward the street, or failing that toward
                      the address point (which usually sits on the street side);
                    * with no outline it is centred on the address point with its front
                      toward the street; with no street either, it is left unrotated.
``outlines``        what to draw on the aerial image: the real outline (``kind: building``)
                    and the model's own extent, one rectangle per block (``kind: model``)
``imagery``         centre, radius and zoom of the aerial window that covers all of it
``pending``         true while the OpenStreetMap request is still running

Two network calls, made side by side: Vexcel ``property/extract`` and one OSM Overpass
request, each carrying only the coordinates. Overpass can take a minute, and a call in
progress cannot wait on it: ``build_context`` waits ``deadline_s`` seconds, then carries
on with what it has. Whatever arrives later completes the sidecar, and ``on_late`` is
called so the caller can re-project the building with the better placement. Overpass
requests from this process run one at a time; the public server rejects bursts.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ingest.georegister import latlng_to_local, local_to_latlng
from ingest.unified.schema import BuildingModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "data" / "processed" / "context"
STREET_DEADLINE_S = 4.0
NOT_A_FRONTAGE = ("motorway", "motorway_link", "trunk", "trunk_link")
ROAD_RADIUS_M = 150.0
BUILDING_RADIUS_M = 70.0
BUILDING_MAX_DISTANCE_M = 45.0  # an outline further than this from the address point is another building
BUILDING_MIN_AREA_M2 = 40.0
OVERPASS_ATTEMPTS = 2
OVERPASS_TIMEOUT_S = 45.0
OVERPASS_COOLDOWN_S = 300.0
_overpass_down = {"until": 0.0}
VEXCEL_TIMEOUT_S = 8.0
OSM_GRACE_S = 1.0
CONTEXT_VERSION = 4  # bump when the sidecar's shape changes, so older files are rebuilt
IMAGERY_MARGIN_M = 25.0
_overpass_gate = threading.Semaphore(1)
# aerial window radius -> tile zoom, chosen so the mosaic stays about six tiles across
ZOOM_BY_RADIUS = ((40.0, 21), (85.0, 20), (170.0, 19), (340.0, 18))
MIN_ZOOM = 17


# ---------------------------------------------------------------------------
# OpenStreetMap: roads and building outlines in one request
# ---------------------------------------------------------------------------


def fetch_site(lat: float, lon: float) -> dict:
    """Raw Overpass response with the roads and buildings around a point."""
    from ingest.mp3d.aerial import OVERPASS_URL, ROAD_CLASS_RANK, USER_AGENT

    classes = "|".join(ROAD_CLASS_RANK)
    query = (f"[out:json][timeout:25];("
             f'way(around:{ROAD_RADIUS_M:.0f},{lat},{lon})["highway"~"^({classes})$"];'
             f'way(around:{BUILDING_RADIUS_M:.0f},{lat},{lon})["building"];'
             f'relation(around:{BUILDING_RADIUS_M:.0f},{lat},{lon})["building"];'
             f");out geom tags;")
    body = urllib.parse.urlencode({"data": query}).encode()
    last: Optional[Exception] = None
    with _overpass_gate:  # one request at a time from this process
        for attempt in range(OVERPASS_ATTEMPTS):
            if time.monotonic() < _overpass_down["until"]:
                raise RuntimeError("Overpass is not answering; not asking again for a few minutes")
            req = urllib.request.Request(OVERPASS_URL, data=body, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=OVERPASS_TIMEOUT_S) as resp:
                    return json.loads(resp.read().decode())
            except Exception as exc:  # noqa: BLE001 - the public endpoint 502s and times out under load
                last = exc
                time.sleep(1.0 + 2.0 * attempt)
        # it failed every time: requests already waiting at the gate give up at once instead of
        # each spending minutes on a server that is down or has stopped accepting this address
        _overpass_down["until"] = time.monotonic() + OVERPASS_COOLDOWN_S
    raise RuntimeError(f"Overpass failed: {type(last).__name__}: {str(last)[:80]}")


def street_from_payload(payload: dict, lat: float, lon: float) -> tuple[Optional[dict], Optional[str]]:
    """``(street, error)`` from the roads of a ``fetch_site`` payload."""
    from ingest.mp3d.aerial import road_candidates_from_payload, street_bearing_from_roads

    roads = {"elements": [e for e in payload.get("elements") or [] if (e.get("tags") or {}).get("highway")]}
    candidates = [c for c in road_candidates_from_payload(roads, lat, lon) if c.highway not in NOT_A_FRONTAGE]
    sb = street_bearing_from_roads(candidates)
    if sb.bearing_deg is None:
        return None, sb.note or "no street found"
    first = next((c for c in sb.candidates if abs(c.bearing_deg - sb.bearing_deg) < 1e-6), None)
    return {"bearing_deg": round(float(sb.bearing_deg), 1), "source": sb.source, "confidence": sb.confidence,
            "ambiguous": bool(sb.ambiguous), "note": sb.note, "name": first.name if first else "",
            "distance_m": round(first.distance_m, 1) if first else None}, None


def _area(poly) -> float:
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2.0


def _contains(poly, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _distance_to_outline(poly, x: float, y: float) -> float:
    best = float("inf")
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def oriented_box(points) -> dict:
    """Smallest rectangle around ``points``: ``{"centre", "length_m", "width_m", "axis_deg"}``.

    ``axis_deg`` is the direction of the long side, counter-clockwise from east, in [0, 180).
    """
    from ingest.mp3d.geometry import convex_hull

    hull = convex_hull([(float(x), float(y)) for x, y in points])
    best = None
    for i in range(len(hull)):
        (ax, ay), (bx, by) = hull[i], hull[(i + 1) % len(hull)]
        theta = math.atan2(by - ay, bx - ax)
        c, s = math.cos(-theta), math.sin(-theta)
        xs = [px * c - py * s for px, py in hull]
        ys = [px * s + py * c for px, py in hull]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if best is None or w * h < best[0] - 1e-9:
            mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            cb, sb = math.cos(theta), math.sin(theta)
            centre = (mx * cb - my * sb, mx * sb + my * cb)
            axis = theta if w >= h else theta + math.pi / 2
            best = (w * h, centre, max(w, h), min(w, h), math.degrees(axis) % 180.0)
    if best is None:
        return {"centre": (0.0, 0.0), "length_m": 0.0, "width_m": 0.0, "axis_deg": 0.0}
    return {"centre": best[1], "length_m": round(best[2], 2), "width_m": round(best[3], 2), "axis_deg": round(best[4], 1)}


def _choose(found: list[dict], single: bool) -> tuple[list[dict], str]:
    """Which outlines stand for the property. ``found`` items: ``{"poly", "area", "distance", "inside"}``
    in metres around the address point. ``single``: the model is one building, so pick one outline."""
    found = [f for f in found if f["area"] >= BUILDING_MIN_AREA_M2]
    if not found:
        return [], ""
    if not single:
        return found, f"{len(found)} buildings on the property"
    inside = [f for f in found if f["inside"]]
    if inside:
        return [max(inside, key=lambda f: f["area"])], "contains the address point"
    nearest = min(f["distance"] for f in found)
    if nearest > BUILDING_MAX_DISTANCE_M:
        return [], ""
    # an address point at the kerb is often nearest a shed or a garage: take the largest of the close ones
    close = [f for f in found if f["distance"] <= nearest + 8.0]
    pick = max(close, key=lambda f: f["area"])
    return [pick], f"nearest outline, {pick['distance']:.0f} m from the address point"


def _outline(source: str, picks: list[dict], match: str, lat: float, lon: float, detail: str = "") -> Optional[dict]:
    if not picks:
        return None
    box = oriented_box([p for f in picks for p in f["poly"]])
    return {
        "source": source, "match": match, "detail": detail, "area_m2": round(sum(f["area"] for f in picks), 1),
        "polygons_latlng": [[[round(v, 7) for v in local_to_latlng(x, y, (lat, lon))] for x, y in f["poly"]] for f in picks],
        "centre_latlng": [round(v, 7) for v in local_to_latlng(box["centre"][0], box["centre"][1], (lat, lon))],
        "length_m": box["length_m"], "width_m": box["width_m"], "axis_deg": box["axis_deg"],
    }


def _measure(poly: list) -> dict:
    inside = _contains(poly, 0.0, 0.0)
    return {"poly": poly, "area": _area(poly), "inside": inside,
            "distance": 0.0 if inside else _distance_to_outline(poly, 0.0, 0.0)}


def building_from_payload(payload: dict, lat: float, lon: float, single: bool = True) -> Optional[dict]:
    """The OpenStreetMap building at an address point (one building only: OSM has no notion of the parcel)."""
    if not single:
        return None
    found = []
    for e in payload.get("elements") or []:
        if "building" not in (e.get("tags") or {}):
            continue
        rings = []
        if e.get("type") == "way" and e.get("geometry"):
            rings = [e["geometry"]]
        elif e.get("type") == "relation":
            rings = [m["geometry"] for m in e.get("members") or [] if m.get("role") in ("outer", "") and m.get("geometry")]
        for ring in rings:
            poly = [latlng_to_local(p["lat"], p["lon"], (lat, lon)) for p in ring if "lat" in p]
            if len(poly) > 1 and poly[0] == poly[-1]:
                poly = poly[:-1]
            if len(poly) >= 3:
                found.append(_measure(poly))
    picks, match = _choose(found, single)
    return _outline("OpenStreetMap", picks, match, lat, lon)


def structures_from_property(data: Optional[dict], lat: float, lon: float, single: bool = True,
                             layer: str = "urban") -> Optional[dict]:
    """Building outline(s) from a Vexcel ``property/extract`` response.

    The response has one feature per imagery vintage, each listing the structures on the
    parcel. The newest vintage of the layer the aerial tiles come from is used, so the
    outline matches the picture it is drawn on.
    """
    from ingest.mp3d.aerial import parse_polygon_wkt

    features = [f for f in (data or {}).get("features") or [] if (f.get("properties") or {}).get("structures")]
    if not features:
        return None

    def rank(f: dict) -> tuple:
        p = f.get("properties") or {}
        return (p.get("metadata.layer-name") == layer, str(p.get("metadata.ortho-date-estimate") or ""))

    props = max(features, key=rank).get("properties") or {}
    found = []
    for st in props.get("structures") or []:
        ring = parse_polygon_wkt(str(st.get("footprint.wkt") or ""))  # (lon, lat) pairs
        poly = [latlng_to_local(la, lo, (lat, lon)) for lo, la in ring]
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly = poly[:-1]
        if len(poly) >= 3:
            found.append(_measure(poly))
    picks, match = _choose(found, single)
    if single and not picks and found:  # nothing near the address point: the parcel's main building
        picks, match = [max(found, key=lambda f: f["area"])], "largest building on the parcel"
    detail = f"{props.get('metadata.collection') or ''} {str(props.get('metadata.ortho-date-estimate') or '')[:10]}".strip()
    return _outline("Vexcel", picks, match, lat, lon, detail)


def fetch_vexcel_outline(lat: float, lon: float, single: bool = True) -> Optional[dict]:
    from ingest.mp3d.aerial import VexcelClient

    return structures_from_property(VexcelClient(timeout_s=VEXCEL_TIMEOUT_S).property_extract(lat, lon), lat, lon, single)


# ---------------------------------------------------------------------------
# sidecar
# ---------------------------------------------------------------------------


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
                  deadline_s: float = STREET_DEADLINE_S, on_late: Optional[Callable[[dict], None]] = None) -> dict:
    """Build (or reuse) the sidecar for a model and write it to disk.

    ``on_late(context)`` is called from a background thread whenever a source answers
    after ``deadline_s``; by then the improved sidecar is already on disk.
    """
    cached = None if refresh else load_context(bm.id)
    # a sidecar is reused when it has this version's shape and, if the network may be used,
    # when its sources had all answered (a lookup that was still running is asked again)
    if cached is not None and cached.get("version") == CONTEXT_VERSION and (
            not network or (cached.get("sources_done") and not cached.get("pending"))):
        return cached
    geo = list(geolocation or bm.geolocation or []) or None
    if not (network and geo):
        return _assemble(bm, geo, {}, pending=False)
    lat, lon = float(geo[0]), float(geo[1])
    single = int(bm.stats.get("blocks") or 1) == 1
    state: dict = {"late": False}
    lock = threading.Lock()

    def finish_late() -> None:
        with lock:
            if not state["late"]:
                return
            late = _assemble(bm, geo, state, pending=not state.get("osm_done"))
        if on_late is not None:
            try:
                on_late(late)
            except Exception:  # noqa: BLE001 - a failed re-projection must not kill the thread loudly
                pass

    def vexcel_worker() -> None:
        try:
            outline = fetch_vexcel_outline(lat, lon, single)
            with lock:
                state["vexcel"] = outline
        except Exception as exc:  # noqa: BLE001 - no key, no coverage, timeout: the sidecar is optional
            with lock:
                state["vexcel_error"] = str(exc)[:120]
        with lock:
            state["vexcel_done"] = True
        finish_late()

    def osm_worker() -> None:
        try:
            payload = fetch_site(lat, lon)
            street, error = street_from_payload(payload, lat, lon)
            outline = building_from_payload(payload, lat, lon, single)
            with lock:
                state["street"], state["street_error"], state["osm"] = street, error, outline
        except Exception as exc:  # noqa: BLE001
            with lock:
                state["street_error"] = f"OpenStreetMap lookup failed: {str(exc)[:120]}"
        with lock:
            state["osm_done"] = True
        finish_late()

    workers = [threading.Thread(target=vexcel_worker, daemon=True), threading.Thread(target=osm_worker, daemon=True)]
    started = time.monotonic()
    for w in workers:
        w.start()
    workers[0].join(deadline_s)  # the outline: normally a fraction of a second
    with lock:
        placed = bool(state.get("vexcel"))
    # Once the outline has placed the building, the street only settles which way round it
    # faces, so it gets a short grace period rather than the whole deadline.
    remaining = max(0.0, deadline_s - (time.monotonic() - started))
    workers[1].join(min(remaining, OSM_GRACE_S) if placed else remaining)
    with lock:
        state["late"] = True  # anything finishing from here on re-assembles and reports
        return _assemble(bm, geo, state, pending=not state.get("osm_done"))


def _front_bearing(yaw_deg: float) -> float:
    """Compass bearing the model's front (-y) points at when the model is turned by ``yaw_deg``."""
    return (180.0 - yaw_deg) % 360.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _assemble(bm: BuildingModel, geo, state: dict, pending: bool) -> dict:
    from ingest.unified.project import _bounds, placement, to_latlng

    street = state.get("street")
    outline = state.get("vexcel") or state.get("osm")
    x0, y0, x1, y1 = _bounds(bm)
    scan = bm.archetype in ("whole_scan", "scan_clusters")
    anchor = None
    front_hint = None
    fit_scale = None
    if scan:
        yaw, source = bm.yaw_deg, "the scan's own position and yaw"
    elif outline:
        # long side of the model along the long side of the real outline; of the two ways round,
        # the front goes toward the street, or toward the address point when no street is known
        base = outline["axis_deg"] if (x1 - x0) >= (y1 - y0) else outline["axis_deg"] - 90.0
        options = [base, base + 180.0]
        if street:
            front_hint, hint_name = street["bearing_deg"], "the street"
        else:
            east, north = latlng_to_local(geo[0], geo[1], outline["centre_latlng"])
            if math.hypot(east, north) >= 5.0:
                front_hint, hint_name = math.degrees(math.atan2(east, north)) % 360.0, "the address point"
        if front_hint is not None:
            options.sort(key=lambda a: _gap(_front_bearing(a), front_hint))
        yaw = round((options[0] + 180.0) % 360.0 - 180.0, 1)
        if int(bm.stats.get("blocks") or 1) == 1:
            anchor = outline["centre_latlng"]
            source = f"the {outline['source']} building outline" + (f", front toward {hint_name}" if front_hint is not None else "")
            # The generated layout's size comes from the unit list, not from the building, so on
            # the map it is fitted to the real outline's box: entrances land on the building, not
            # in the street or on the neighbours. Which model axis is the long one decides the pairing.
            w, d = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            along, across = (outline["length_m"], outline["width_m"])
            fit = (along / w, across / d) if (x1 - x0) >= (y1 - y0) else (across / w, along / d)
            fit_scale = [round(min(max(v, 0.1), 10.0), 4) for v in fit]
        else:
            # several buildings but one known outline: it can set the direction the buildings
            # run in, not where each one stands
            source = f"the address point; the {outline['source']} outline shows the building at that point"
    elif street:
        yaw = round((180.0 - street["bearing_deg"] + 180.0) % 360.0 - 180.0, 1)
        source = "the address point, front toward the street"
    else:
        yaw, source = 0.0, "the address point (orientation unknown)"
    context = {
        "version": CONTEXT_VERSION, "id": bm.id, "geolocation": geo, "street": street, "street_error": state.get("street_error"),
        "building_outline": outline, "vexcel_error": state.get("vexcel_error"),
        "sources_done": bool(state.get("vexcel_done") or state.get("osm_done")), "pending": pending,
        "yaw_deg": yaw, "anchor_latlng": anchor, "placement_source": source, "fit_scale": fit_scale,
    }

    # A property with several buildings is laid out on a grid whose positions are invented, so
    # nothing of that grid is put on the map: only the address point and the real outline there.
    placed = scan or int(bm.stats.get("blocks") or 1) == 1
    context["layout_placed"] = placed
    place = placement(bm, context)
    outlines: list[dict] = []
    if geo:
        for poly in (outline or {}).get("polygons_latlng") or []:
            outlines.append({"kind": "building", "source": outline["source"], "latlng": poly})
        blocks: dict[str, list] = {}
        for s in bm.spaces:
            if s.parent is None and s.kind != "walkway":
                blocks.setdefault(s.block, []).extend(s.polygon)
        for key in sorted(blocks):
            if fit_scale or not placed:  # fitted to the real outline (it already shows the place), or not placeable
                break
            pts = blocks[key]
            bx0, by0 = min(p[0] for p in pts), min(p[1] for p in pts)
            bx1, by1 = max(p[0] for p in pts), max(p[1] for p in pts)
            outlines.append({"kind": "model", "source": "scan" if scan else "generated", "block": key,
                             "latlng": [to_latlng(place, x, y) for x, y in ((bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0))]})
    context["outlines"] = outlines
    context["footprint_latlng"] = [to_latlng(place, x, y) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))] if geo else []

    # aerial window: centred on everything that will be drawn, wide enough to hold it
    if geo:
        pts = [p for o in outlines for p in o["latlng"]] + ([] if placed and outlines else [list(geo)])
        lat_c = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
        lng_c = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
        reach = max(math.hypot(*latlng_to_local(p[0], p[1], (lat_c, lng_c))) for p in pts)
        radius = max(35.0, reach + IMAGERY_MARGIN_M)
        zoom = next((z for limit, z in ZOOM_BY_RADIUS if radius <= limit), MIN_ZOOM)
        context["centre_latlng"] = [round(lat_c, 7), round(lng_c, 7)]
        context["imagery"] = {"layer": "urban", "zoom": zoom, "radius_m": round(radius)}
    else:
        context["centre_latlng"] = None
        context["imagery"] = {}
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    context_path(bm.id).write_text(json.dumps(context, indent=1), encoding="utf-8")
    return context

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
                    * a generated building drawn inside the real outline (``shape`` is
                      "outline"; see ``ingest.unified.shape``) carries the outline's own
                      frame, so it sits on the outline exactly and nothing is fitted;
                    * a generated bar (no usable outline for the layout) is centred on the
                      real outline when there is one and turned so its long side lies
                      along the outline's long side. Of the two ways round, the front
                      goes toward the street, or failing that toward the address point
                      (which usually sits on the street side);
                    * with no outline it is centred on the address point with its front
                      toward the street; with no street either, it is left unrotated.
``surroundings``    what stands around the building, from the same OpenStreetMap answer:
                    street centrelines, car parks and neighbouring buildings (houses told
                    apart from the rest), with the modelled building's own outline flagged.
                    ``project.to_building`` casts rays from the windows at these to decide
                    what each room looks at (see ``ingest.unified.surroundings``)
``outlines``        what to draw on the aerial image: the real outline (``kind: building``)
                    and the model's own extent (``kind: model``): the outer edge of its
                    lowest floor when it was drawn to the outline, else one rectangle per block
``shape``           whether the model's plates follow the real outline or an assumed bar, and why
``parcel_outlines`` every structure on the parcel (garages and sheds included; ``main_outlines``
                    leaves those out)
``imagery``         centre, radius and zoom of the aerial window that covers all of it
``pending``         true while the OpenStreetMap request is still running

The outline is needed before the layout is generated, so the pipeline asks the sources first
(``site_sources``), hands the outline to the layout (``layout_site``) and then builds the
sidecar from the same answers (``build_context(..., sources=...)``).

Two lookups, made side by side: Vexcel ``property/extract`` (kept on disk, so a building is
asked about once) and OpenStreetMap, each carrying only the coordinates. OpenStreetMap is read from a local index of the Los Angeles and Houston
extracts when one has been built (``ingest.unified.osm_local``); elsewhere it is one request
to the public Overpass server. Overpass can take a minute, and a call in
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
from ingest.unified.schema import SELF_PLACED, BuildingModel
from ingest.unified.surroundings import SURROUNDINGS_RADIUS_M, mark_own, surroundings_from_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "data" / "processed" / "context"
# Tuned heuristic: how long a live call waits for the site sources before carrying on (s); later answers arrive via
# on_late. Vexcel normally answers in a fraction of a second; Overpass can take a minute (module docstring).
STREET_DEADLINE_S = 4.0
NOT_A_FRONTAGE = ("motorway", "motorway_link", "trunk", "trunk_link")
ROAD_RADIUS_M = 150.0  # assumed: the fronting street lies within 150 m of the address point
BUILDING_MAX_DISTANCE_M = 45.0  # an outline further than this from the address point is another building
BUILDING_MIN_AREA_M2 = 40.0  # assumed: a traced outline under 40 m2 is a kiosk or shed, never the building
# A structure smaller than this is an outbuilding (garage, carport, shed, pool house), not one of
# the property's buildings: ignored when the parcel's outlines are matched to the record.
OUTBUILDING_M2 = 150.0
# The parcel's main outlines are trusted over the record's building count when their total area
# is within this share of the record's floor area. Tuned heuristic: +/-20% leaves room for tracing error and
# record rounding; a much looser band would accept outlines that belong to a different building count.
OUTLINE_AREA_TOL = 0.2
# Overpass: 2 tries with a 45 s socket timeout each (the query itself asks the server for 25 s); after both fail,
# every caller skips Overpass for 5 min (300 s) instead of queueing on a server that is down.
OVERPASS_ATTEMPTS = 2
OVERPASS_TIMEOUT_S = 45.0
OVERPASS_COOLDOWN_S = 300.0
_overpass_down = {"until": 0.0}
VEXCEL_TIMEOUT_S = 8.0  # assumed socket timeout (s) for one property/extract request; answers are cached on disk
OSM_GRACE_S = 1.0  # tuned: once Vexcel has placed the building, wait at most 1 s more for the street (facing only)
CONTEXT_VERSION = 6  # bump when the sidecar's shape changes, so older files are rebuilt
IMAGERY_MARGIN_M = 25.0  # assumed margin (m) of aerial imagery kept around everything drawn
_overpass_gate = threading.Semaphore(1)
# aerial window radius -> tile zoom, chosen so the mosaic stays about six tiles across
# (one zoom level down doubles the metres per tile, so each radius limit is about double the previous)
ZOOM_BY_RADIUS = ((40.0, 21), (85.0, 20), (170.0, 19), (340.0, 18))
MIN_ZOOM = 17


# ---------------------------------------------------------------------------
# OpenStreetMap: roads, building outlines and car parks in one request
# ---------------------------------------------------------------------------


def fetch_site(lat: float, lon: float) -> dict:
    """The roads, buildings, car parks and residential land around a point, as Overpass elements.

    Read from the local OpenStreetMap index when it covers the point (milliseconds; see
    ``ingest.unified.osm_local``), otherwise asked of the public Overpass server.
    """
    from ingest.mp3d.aerial import OVERPASS_URL, ROAD_CLASS_RANK, USER_AGENT
    from ingest.unified import osm_local

    local = osm_local.fetch_site(lat, lon, max(ROAD_RADIUS_M, SURROUNDINGS_RADIUS_M))
    if local is not None:
        return local

    classes = "|".join(ROAD_CLASS_RANK)
    near = f"around:{SURROUNDINGS_RADIUS_M:.0f},{lat},{lon}"
    query = (f"[out:json][timeout:25];("
             f'way(around:{ROAD_RADIUS_M:.0f},{lat},{lon})["highway"~"^({classes})$"];'
             f'way({near})["building"];relation({near})["building"];'
             f'way({near})["amenity"="parking"];relation({near})["amenity"="parking"];'
             f'way({near})["landuse"="residential"];relation({near})["landuse"="residential"];'
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
                time.sleep(1.0 + 2.0 * attempt)  # linear backoff: 1 s, then 3 s
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
    close = [f for f in found if f["distance"] <= nearest + 8.0]  # tuned: "close" = within 8 m of the nearest
    pick = max(close, key=lambda f: f["area"])
    return [pick], f"nearest outline, {pick['distance']:.0f} m from the address point"


def _outline(source: str, picks: list[dict], match: str, lat: float, lon: float, detail: str = "") -> Optional[dict]:
    if not picks:
        return None
    box = oriented_box([p for f in picks for p in f["poly"]])

    def ring(points) -> list:
        return [[round(v, 7) for v in local_to_latlng(x, y, (lat, lon))] for x, y in points]

    out = {
        "source": source, "match": match, "detail": detail, "area_m2": round(sum(f["area"] for f in picks), 1),
        "polygons_latlng": [ring(f["poly"]) for f in picks],
        "centre_latlng": [round(v, 7) for v in local_to_latlng(box["centre"][0], box["centre"][1], (lat, lon))],
        "length_m": box["length_m"], "width_m": box["width_m"], "axis_deg": box["axis_deg"],
    }
    if any(f.get("holes") for f in picks):
        # courtyards and light wells, one list of rings per polygon above (same order)
        out["holes_latlng"] = [[ring(h) for h in f.get("holes") or []] for f in picks]
    return out


def _measure(poly: list, holes: Optional[list] = None) -> dict:
    """An outline ring in metres around the address point. ``holes``: courtyards inside it."""
    inside = _contains(poly, 0.0, 0.0)
    holes = [h for h in holes or [] if len(h) >= 3 and _area(h) >= 4.0]  # holes under 4 m2 are tracing noise
    return {"poly": poly, "holes": holes, "area": max(_area(poly) - sum(_area(h) for h in holes), 0.0), "inside": inside,
            "distance": 0.0 if inside else _distance_to_outline(poly, 0.0, 0.0)}


def building_from_payload(payload: dict, lat: float, lon: float, single: bool = True) -> Optional[dict]:
    """The OpenStreetMap building at an address point (one building only: OSM has no notion of the parcel)."""
    if not single:
        return None
    found = []
    for e in payload.get("elements") or []:
        if "building" not in (e.get("tags") or {}):
            continue
        rings, inner = [], []
        if e.get("type") == "way" and e.get("geometry"):
            rings = [e["geometry"]]
        elif e.get("type") == "relation":
            rings = [m["geometry"] for m in e.get("members") or [] if m.get("role") in ("outer", "") and m.get("geometry")]
            inner = [m["geometry"] for m in e.get("members") or [] if m.get("role") == "inner" and m.get("geometry")]

        def local(ring) -> list:
            poly = [latlng_to_local(p["lat"], p["lon"], (lat, lon)) for p in ring if "lat" in p]
            return poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly

        holes = [h for h in (local(r) for r in inner) if len(h) >= 3]
        for ring in rings:
            poly = local(ring)
            if len(poly) >= 3:
                # a courtyard belongs to the outer ring that holds it
                found.append(_measure(poly, [h for h in holes if _contains(poly, *h[0])]))
    picks, match = _choose(found, single)
    return _outline("OpenStreetMap", picks, match, lat, lon)


def structures_from_property(data: Optional[dict], lat: float, lon: float, single: bool = True,
                             layer: str = "urban") -> Optional[dict]:
    """Building outline(s) from a Vexcel ``property/extract`` response.

    The response has one feature per imagery vintage, each listing the structures on the
    parcel. The newest vintage of the layer the aerial tiles come from is used, so the
    outline matches the picture it is drawn on.
    """
    features = [f for f in (data or {}).get("features") or [] if (f.get("properties") or {}).get("structures")]
    if not features:
        return None

    def rank(f: dict) -> tuple:
        p = f.get("properties") or {}
        return (p.get("metadata.layer-name") == layer, str(p.get("metadata.ortho-date-estimate") or ""))

    props = max(features, key=rank).get("properties") or {}
    found = []
    for st in props.get("structures") or []:
        best = None
        for rings in wkt_polygons(str(st.get("footprint.wkt") or "")):  # (lon, lat) pairs
            local = []
            for ring in rings:
                poly = [latlng_to_local(la, lo, (lat, lon)) for lo, la in ring]
                local.append(poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly)
            if local and len(local[0]) >= 3:
                m = _measure(local[0], local[1:])
                if best is None or m["area"] > best["area"]:  # a footprint in several parts: its main part
                    best = m
        if best is not None:
            found.append(best)
    picks, match = _choose(found, single)
    if single and not picks and found:  # nothing near the address point: the parcel's main building
        picks, match = [max(found, key=lambda f: f["area"])], "largest building on the parcel"
    detail = f"{props.get('metadata.collection') or ''} {str(props.get('metadata.ortho-date-estimate') or '')[:10]}".strip()
    return _outline("Vexcel", picks, match, lat, lon, detail)


def wkt_polygons(wkt: str) -> list[list[list[tuple[float, float]]]]:
    """Every polygon of a WKT POLYGON or MULTIPOLYGON as ``[outer ring, hole, ...]``, points as given. Pure."""
    out: list[list[list[tuple[float, float]]]] = []
    current: list[list[tuple[float, float]]] = []
    ring_depth = 3 if wkt.lstrip().upper().startswith("MULTIPOLYGON") else 2
    depth = start = 0
    for i, ch in enumerate(wkt):
        if ch == "(":
            depth += 1
            if depth == ring_depth:
                start = i + 1
        elif ch == ")":
            if depth == ring_depth:
                ring = []
                for pair in wkt[start:i].split(","):
                    bits = pair.split()
                    if len(bits) >= 2:
                        try:
                            ring.append((float(bits[0]), float(bits[1])))
                        except ValueError:
                            continue
                if len(ring) >= 3:
                    current.append(ring)
            depth -= 1
            if depth == ring_depth - 2:  # the polygon is closed
                if current:
                    out.append(current)
                current = []
    return out


def vexcel_cache_path(lat: float, lon: float) -> Path:
    from ingest.vexcel import vexcel_data_dir

    return vexcel_data_dir() / "property" / f"{lat:.6f}_{lon:.6f}.json"


def vexcel_property(lat: float, lon: float, refresh: bool = False) -> Optional[dict]:
    """Vexcel ``property/extract`` for a point, kept on disk so a building is asked about once.

    An answer of "no parcel here" is kept too; a failed request is not.
    """
    path = vexcel_cache_path(lat, lon)
    if not refresh and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("data")
        except ValueError:
            pass
    from ingest.mp3d.aerial import VexcelClient

    data = VexcelClient(timeout_s=VEXCEL_TIMEOUT_S).property_extract(lat, lon)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"lat": lat, "lon": lon, "data": data}), encoding="utf-8")
    except OSError:  # a read-only data folder only costs the next run a request
        pass
    return data


def fetch_vexcel_outline(lat: float, lon: float, single: bool = True) -> Optional[dict]:
    return structures_from_property(vexcel_property(lat, lon), lat, lon, single)


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


def site_sources(building_id: str, geolocation, single: bool = True, buildings: int = 1, network: bool = True,
                 refresh: bool = False, deadline_s: float = STREET_DEADLINE_S,
                 on_late: Optional[Callable[[dict], None]] = None) -> dict:
    """What Vexcel and OpenStreetMap say about a point: the ``state`` that ``layout_site`` and
    ``build_context`` read.

    Keys: ``vexcel`` / ``osm`` (the outline of the building at the point), ``vexcel_all`` (every
    structure on the parcel, asked for when the record describes ``buildings`` > 1), ``street``,
    ``surroundings``, the errors, and ``vexcel_done`` / ``osm_done``.

    Asked before the layout stage, because the layout is drawn to the outline. Nothing is
    fetched when a finished sidecar of this version is on disk, the Vexcel answer is kept on
    disk (``vexcel_property``) and OpenStreetMap is read from the local index where it covers
    the point, so generating a building again costs no request. A source that has not answered
    after ``deadline_s`` is not waited for: ``on_late(state)`` is called from its thread when
    it does, so the caller can lay the building out again with what arrived.
    """
    geo = list(geolocation or []) or None
    cached = None if refresh else load_context(building_id)
    if cached is not None and cached.get("version") == CONTEXT_VERSION and (
            not network or (cached.get("sources_done") and not cached.get("pending"))):
        outline = cached.get("building_outline") or {}
        return {"vexcel": outline if outline.get("source") == "Vexcel" else None,
                "osm": outline if outline and outline.get("source") != "Vexcel" else None,
                "vexcel_all": cached.get("parcel_outlines"), "street": cached.get("street"),
                "street_error": cached.get("street_error"), "vexcel_error": cached.get("vexcel_error"),
                "surroundings": cached.get("surroundings"), "vexcel_done": bool(cached.get("sources_done")),
                "osm_done": bool(cached.get("sources_done")), "reused": cached}
    if not (network and geo):
        return {}
    lat, lon = float(geo[0]), float(geo[1])
    state: dict = {"late": False}
    lock = threading.Lock()

    def finish_late() -> None:
        with lock:
            if not state["late"]:
                return
            snapshot = dict(state)
        if on_late is not None:
            try:
                on_late(snapshot)
            except Exception:  # noqa: BLE001 - a failed re-projection must not kill the thread loudly
                pass

    def vexcel_worker() -> None:
        try:
            outline = fetch_vexcel_outline(lat, lon, single)
            # every structure on the parcel, even for one building: the record's building count
            # can be wrong, and the parcel's outlines are checked against its floor area
            every = fetch_vexcel_outline(lat, lon, False)
            with lock:
                state["vexcel"], state["vexcel_all"] = outline, every
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
            around = surroundings_from_payload(payload, lat, lon)
            with lock:
                state["street"], state["street_error"], state["osm"] = street, error, outline
                state["surroundings"] = around
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
        state["late"] = True  # anything finishing from here on is reported through on_late
        return dict(state)


def outline_areas(outline: Optional[dict]) -> list[float]:
    """Area in m2 of each polygon of an outline (courtyards taken out), in ``polygons_latlng`` order."""
    out = []
    polys = (outline or {}).get("polygons_latlng") or []
    holes = (outline or {}).get("holes_latlng") or []
    for i, ring in enumerate(polys):
        if len(ring) < 3:
            out.append(0.0)
            continue
        ref = ring[0]
        area = _area([latlng_to_local(la, lo, ref) for la, lo in ring])
        for h in (holes[i] if i < len(holes) else []):
            if len(h) >= 3:
                area -= _area([latlng_to_local(la, lo, ref) for la, lo in h])
        out.append(max(area, 0.0))
    return out


def main_outlines(outline: Optional[dict]) -> Optional[dict]:
    """``outline`` without its outbuildings (polygons under ``OUTBUILDING_M2``: garages, sheds).

    None when nothing is left. Holes stay paired with their polygons.
    """
    if not outline:
        return None
    areas = outline_areas(outline)
    keep = [i for i, a in enumerate(areas) if a >= OUTBUILDING_M2]
    if not keep:
        return None
    if len(keep) == len(areas):
        return outline
    out = dict(outline)
    out["polygons_latlng"] = [outline["polygons_latlng"][i] for i in keep]
    if outline.get("holes_latlng"):
        holes = outline["holes_latlng"]
        out["holes_latlng"] = [holes[i] if i < len(holes) else [] for i in keep]
    out["area_m2"] = round(sum(areas[i] for i in keep), 1)
    out["outbuildings_dropped"] = len(areas) - len(keep)
    return out


def _area_matches(total: float, floor_area_m2: Optional[float]) -> bool:
    return bool(floor_area_m2) and abs(total - floor_area_m2) <= OUTLINE_AREA_TOL * floor_area_m2


def layout_site(state: dict, geolocation, buildings: int = 1, floor_area_m2: Optional[float] = None) -> dict:
    """What the layout stage needs of the site: the outline(s) to draw the plates to.

    One building: the outline at the address point (Vexcel first, OpenStreetMap second). Several
    buildings: the parcel's main structures (outbuildings under ``OUTBUILDING_M2`` left out),
    used when there are exactly as many as the record has buildings.

    The record's building count is not always right (CoStar can say 2 for a complex the imagery
    shows as 3). ``floor_area_m2`` is the record's area of one floor: when the parcel's main
    outlines add up to it (within ``OUTLINE_AREA_TOL``) they are used whatever the count says,
    and ``buildings_from_outlines`` gives the count they imply. The same check catches a record
    of one building whose address-point outline is far too small for it while the parcel's
    outlines together fit. ``polygons_latlng`` is empty, with a ``reason``, when there is
    nothing the layout may use.
    """
    street = (state.get("street") or {}).get("bearing_deg")
    base = {"street_bearing_deg": street, "address_latlng": list(geolocation) if geolocation else None,
            "polygons_latlng": []}
    every = main_outlines(state.get("vexcel_all"))
    found = len((every or {}).get("polygons_latlng") or [])
    total = sum(outline_areas(every))
    by_area = found > 1 and _area_matches(total, floor_area_m2)
    dropped = (every or {}).get("outbuildings_dropped") or 0
    extra = {}
    if buildings <= 1:
        outline = state.get("vexcel") or state.get("osm")
        own = sum(outline_areas(outline)) if outline else 0.0
        if by_area and not (outline and _area_matches(own, floor_area_m2)):
            outline = every
            extra = {"buildings_from_outlines": found,
                     "count_note": f"the record describes one building, but its outline is too small for the "
                                   f"recorded floor area and the parcel's {found} buildings together match it"}
        elif not outline:
            return {**base, "reason": "no building outline was found for this address"}
    elif found == buildings:
        outline = every
    elif by_area:
        outline = every
        extra = {"buildings_from_outlines": found,
                 "count_note": f"the record describes {buildings} buildings, but the parcel has {found} "
                               f"({total:.0f} m2 together, against {floor_area_m2:.0f} m2 recorded per floor)"}
    else:
        side = f" (and {dropped} outbuilding{'' if dropped == 1 else 's'} left out)" if dropped else ""
        why = (f"the record describes {buildings} buildings and {found} outline{'' if found == 1 else 's'} "
               f"{'was' if found == 1 else 'were'} found on the property{side}, so the buildings cannot be "
               "matched to outlines")
        if found > 1 and floor_area_m2:
            why += f"; together they cover {total:.0f} m2 against {floor_area_m2:.0f} m2 recorded per floor"
        return {**base, "reason": why}
    if dropped and outline is every:
        extra["outbuildings_dropped"] = dropped
    return {**base, **extra, "source": outline["source"], "detail": outline.get("detail") or "",
            "polygons_latlng": outline["polygons_latlng"], "holes_latlng": outline.get("holes_latlng") or []}


def build_context(bm: BuildingModel, geolocation=None, network: bool = True, refresh: bool = False,
                  deadline_s: float = STREET_DEADLINE_S, on_late: Optional[Callable[[dict], None]] = None,
                  sources: Optional[dict] = None) -> dict:
    """Build (or reuse) the sidecar for a model and write it to disk.

    ``sources`` is a ``site_sources`` result gathered earlier (the pipeline asks before the
    layout stage); without it the sources are asked here. ``on_late(context)`` is called from a
    background thread whenever a source answers after ``deadline_s``; by then the improved
    sidecar is already on disk.
    """
    geo = list(geolocation or bm.geolocation or []) or None
    if sources is None:
        def late(state: dict) -> None:
            context = _assemble(bm, geo, state, pending=not state.get("osm_done"))
            if on_late is not None:
                on_late(context)

        blocks = int(bm.stats.get("blocks") or 1)
        sources = site_sources(bm.id, geo, blocks == 1, blocks, network, refresh, deadline_s, late)
        # a finished sidecar of this version is reused as it is
        if sources.get("reused") is not None:
            return sources["reused"]
    if not sources:
        return _assemble(bm, geo, {}, pending=False)
    return _assemble(bm, geo, sources, pending=not sources.get("osm_done"))


def _front_bearing(yaw_deg: float) -> float:
    """Compass bearing the model's front (-y) points at when the model is turned by ``yaw_deg``."""
    return (180.0 - yaw_deg) % 360.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _plates(bm: BuildingModel) -> list[tuple[str, list]]:
    """``(block, ring)`` for the outer edge of each building's lowest floor, in model metres."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    lowest = min((s.level for s in bm.spaces), default=1)
    blocks: dict[str, list] = {}
    for s in bm.spaces:
        if s.parent is None and s.kind != "walkway" and s.level == lowest and len(s.polygon) >= 3:
            # grow by 1 cm so touching rooms fuse in the union, then shrink back by the same amount
            blocks.setdefault(s.block, []).append(Polygon(s.polygon).buffer(0.01, join_style="mitre"))
    out = []
    for key in sorted(blocks):
        merged = unary_union(blocks[key]).buffer(-0.01, join_style="mitre")
        for part in getattr(merged, "geoms", [merged]):
            if part.geom_type == "Polygon" and part.area > 1.0:
                out.append((key, [(round(x, 3), round(y, 3)) for x, y in list(part.exterior.simplify(0.02).coords)[:-1]]))
    return out


def _assemble(bm: BuildingModel, geo, state: dict, pending: bool) -> dict:
    from ingest.unified.project import _bounds, placement, to_latlng

    street = state.get("street")
    outline = state.get("vexcel") or state.get("osm")
    x0, y0, x1, y1 = _bounds(bm)
    scan = bm.archetype in SELF_PLACED
    traced = bm.archetype == "floor_plan"
    shape = bm.stats.get("shape") or {}
    drawn = shape.get("kind") == "outline"  # the layout was generated inside the real outline(s)
    if drawn and int(bm.stats.get("blocks") or 1) > 1 and state.get("vexcel_all"):
        outline = main_outlines(state["vexcel_all"]) or state["vexcel_all"]
    anchor = None
    front_hint = None
    fit_scale = None
    if scan:
        yaw, source = bm.yaw_deg, ("building corners matched between the floor plan and the map" if traced
                                   else "the scan's own position and yaw")
    elif drawn:
        # nothing to fit: the model's frame was taken from the outline, so it sits on it exactly
        yaw, anchor = bm.yaw_deg, shape["origin_latlng"]
        source = f"the {shape.get('source')} building outline" + (", front toward the street" if street else "")
    elif outline:
        # long side of the model along the long side of the real outline; of the two ways round,
        # the front goes toward the street, or toward the address point when no street is known
        base = outline["axis_deg"] if (x1 - x0) >= (y1 - y0) else outline["axis_deg"] - 90.0
        options = [base, base + 180.0]
        if street:
            front_hint, hint_name = street["bearing_deg"], "the street"
        else:
            east, north = latlng_to_local(geo[0], geo[1], outline["centre_latlng"])
            if math.hypot(east, north) >= 5.0:  # within 5 m of the centre the address point gives no direction
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
            fit_scale = [round(min(max(v, 0.1), 10.0), 4) for v in fit]  # clamp to 0.1-10x against bad outlines
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
        # every structure on the parcel, kept when the record describes several buildings
        "parcel_outlines": state.get("vexcel_all"),
        # "outline": the plates follow the real outline; "assumed": a bar sized from the unit list
        "shape": {k: shape.get(k) for k in ("kind", "source", "detail", "reason") if shape.get(k)} or None,
    }

    # A property with several buildings is laid out on a grid whose positions are invented, so
    # nothing of that grid is put on the map: only the address point and the real outline there.
    placed = scan or drawn or int(bm.stats.get("blocks") or 1) == 1
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
        if drawn:  # the model's own plates, which should lie on the real outline
            for key, ring in _plates(bm):
                outlines.append({"kind": "model", "source": "generated", "block": key,
                                 "latlng": [to_latlng(place, x, y) for x, y in ring]})
        for key in sorted(blocks):
            if drawn or fit_scale or not placed:  # drawn above, fitted to the real outline, or not placeable
                break
            pts = blocks[key]
            bx0, by0 = min(p[0] for p in pts), min(p[1] for p in pts)
            bx1, by1 = max(p[0] for p in pts), max(p[1] for p in pts)
            outlines.append({"kind": "model", "source": "floor plan" if traced else "scan" if scan else "generated", "block": key,
                             "latlng": [to_latlng(place, x, y) for x, y in ((bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0))]})
    context["outlines"] = outlines
    context["footprint_latlng"] = [to_latlng(place, x, y) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))] if geo else []
    # what the windows look at; the OSM tracings of the modelled building itself are flagged so
    # that no window "sees" its own wall. None until OpenStreetMap has answered.
    middle = to_latlng(place, (x0 + x1) / 2, (y0 + y1) / 2) if geo else None
    context["surroundings"] = mark_own(state.get("surroundings"), (outline or {}).get("polygons_latlng") or [], middle)

    # aerial window: centred on everything that will be drawn, wide enough to hold it
    if geo:
        pts = [p for o in outlines for p in o["latlng"]] + ([] if placed and outlines else [list(geo)])
        lat_c = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
        lng_c = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
        reach = max(math.hypot(*latlng_to_local(p[0], p[1], (lat_c, lng_c))) for p in pts)
        radius = max(35.0, reach + IMAGERY_MARGIN_M)  # 35 m floor: even a tiny building gets some surroundings
        zoom = next((z for limit, z in ZOOM_BY_RADIUS if radius <= limit), MIN_ZOOM)
        context["centre_latlng"] = [round(lat_c, 7), round(lng_c, 7)]
        context["imagery"] = {"layer": "urban", "zoom": zoom, "radius_m": round(radius)}
    else:
        context["centre_latlng"] = None
        context["imagery"] = {}
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    context_path(bm.id).write_text(json.dumps(context, indent=1), encoding="utf-8")
    return context

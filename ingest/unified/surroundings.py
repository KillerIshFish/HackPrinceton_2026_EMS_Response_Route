"""What stands around the building, and so what each window looks at.

A caller says "I can see the car park" or "there are houses out the back". To use that,
every windowed room needs a ``window_facing`` class from the view vocabulary
(``configs/clue_vocab.yaml``): ``street``, ``parking_lot``, ``residential`` or
``adjacent_building``. This module derives it from OpenStreetMap, using the same
Overpass response the site context already fetches for roads and building outlines.

Two steps, both pure:

* ``surroundings_from_payload``: Overpass elements -> a list of ground features, each a
  class plus a line (road centreline) or an area (car park, neighbouring building), and the
  named places nearby (shops, petrol stations, schools). Places are shown on the site map
  and are not yet used for scoring.
* ``facing_for_windows``: from each window, a small fan of rays is cast outward; every
  ray reports the first feature it meets within ``RANGE_M``, and the class most rays
  agree on wins. That is "the nearest thing in front of the glass", which is what a
  caller describes: a shop whose front faces the street across its own car park
  sees the car park.

How a neighbouring building is classed
--------------------------------------
``residential``        tagged as a house (``building=house|detached|...``), or an untyped
                       small building standing on ``landuse=residential``
``parking_lot``        ``amenity=parking`` (not underground) or ``building=parking``
``adjacent_building``  any other building

The building's own outline must not be mistaken for a neighbour. ``mark_own`` flags the
OSM buildings that are the modelled building itself; rays ignore them.

OSM coverage is uneven: where it has mapped nothing in front of a window no ray hits,
this module returns no class for that room, and the caller falls back to the street
bearing. Nothing here is stored in ``BuildingModel``; it lives in the site-context sidecar.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from ingest.georegister import latlng_to_local, local_to_latlng

SURROUNDINGS_RADIUS_M = 120.0   # how far around the address point features are requested
RANGE_M = 60.0                  # how far a window "sees" (assumed: what a caller would call "out the window")
FAN_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)  # assumed 60-degree view cone, 5 rays, so one thin feature can't decide
START_OFFSET_M = 0.5            # rays start just outside the glass
# Assumed: an untagged ("yes") building under 300 m2 standing on residential land is a house.
SMALL_BUILDING_M2 = 300.0
HOUSE_TAGS = frozenset({"house", "detached", "semidetached_house", "terrace", "bungalow", "residential",
                        "cabin", "static_caravan", "farm"})
PARKING_BUILDING_TAGS = frozenset({"parking", "carport"})
PLACE_KEYS = ("amenity", "shop", "tourism", "leisure")
NOT_A_VIEW_ROAD = frozenset({"service"})  # driveways and parking aisles: the car park polygon speaks for them
# ties between classes are settled in this order (the vocabulary's own order)
CLASS_ORDER = ("street", "parking_lot", "residential", "adjacent_building")

Feature = dict  # {"class", "shape": "line" | "area", "name", "latlng": [[lat, lng], ...], "building": bool, "own": bool}


# ---------------------------------------------------------------------------
# small geometry (metres, x east, y north)
# ---------------------------------------------------------------------------


def _area(poly) -> float:
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2.0


def _centroid(poly) -> tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def _contains(poly, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _ray_hit(ox: float, oy: float, dx: float, dy: float, points, closed: bool) -> Optional[float]:
    """Distance along the ray (unit direction) to the nearest edge of a line or ring, or None."""
    best = None
    n = len(points)
    for i in range(n if closed else n - 1):
        (ax, ay), (bx, by) = points[i], points[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:  # ray parallel to the edge
            continue
        t = ((ax - ox) * ey - (ay - oy) * ex) / denom   # along the ray
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom   # along the edge
        if t > 0.0 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


# ---------------------------------------------------------------------------
# Overpass payload -> features
# ---------------------------------------------------------------------------


def _rings(element: dict) -> list[list[dict]]:
    if element.get("type") == "way" and element.get("geometry"):
        return [element["geometry"]]
    if element.get("type") == "relation":
        return [m["geometry"] for m in element.get("members") or []
                if m.get("role") in ("outer", "") and m.get("geometry")]
    return []


def _local_ring(ring: list[dict], origin) -> list[tuple[float, float]]:
    poly = [latlng_to_local(p["lat"], p["lon"], origin) for p in ring if "lat" in p]
    if len(poly) > 1 and poly[0] == poly[-1]:
        poly = poly[:-1]
    return poly


def _within(points, radius: float) -> bool:
    return any(math.hypot(x, y) <= radius for x, y in points) or (len(points) >= 3 and _contains(points, 0.0, 0.0))


def surroundings_from_payload(payload: dict, lat: float, lon: float) -> dict:
    """``{"source", "features", "counts"}`` from a ``context.fetch_site`` Overpass payload.

    Features are in lat/lng, sorted, and rounded, so the same payload gives the same sidecar.
    """
    from ingest.mp3d.aerial import ROAD_CLASS_RANK

    origin = (lat, lon)
    elements = payload.get("elements") or []
    residential_land = []
    for e in elements:
        if (e.get("tags") or {}).get("landuse") == "residential":
            residential_land.extend(p for p in (_local_ring(r, origin) for r in _rings(e)) if len(p) >= 3)

    found: list[tuple[str, str, str, bool, list]] = []  # class, shape, name, is a building, local points
    for e in elements:
        tags = e.get("tags") or {}
        name = str(tags.get("name") or "")
        highway, building = tags.get("highway"), tags.get("building")
        if highway:
            if highway in ROAD_CLASS_RANK and highway not in NOT_A_VIEW_ROAD and e.get("geometry"):
                line = [latlng_to_local(p["lat"], p["lon"], origin) for p in e["geometry"] if "lat" in p]
                # a street can be seen from up to RANGE_M beyond the request radius, so its reach is wider
                if len(line) >= 2 and _within(line, SURROUNDINGS_RADIUS_M + RANGE_M):
                    found.append(("street", "line", name, False, line))
            continue
        parking = tags.get("amenity") == "parking" and tags.get("parking") != "underground"
        if not (parking or building):
            continue
        for poly in (_local_ring(r, origin) for r in _rings(e)):
            if len(poly) < 3 or not _within(poly, SURROUNDINGS_RADIUS_M):
                continue
            if parking or building in PARKING_BUILDING_TAGS:
                cls = "parking_lot"
            elif building in HOUSE_TAGS:
                cls = "residential"
            elif building == "yes" and _area(poly) < SMALL_BUILDING_M2 and any(
                    _contains(land, *_centroid(poly)) for land in residential_land):
                cls = "residential"
            else:
                cls = "adjacent_building"
            found.append((cls, "area", name, bool(building), poly))

    places = []
    for e in elements:
        tags = e.get("tags") or {}
        kind = next((str(tags[k]) for k in PLACE_KEYS if k in tags), None)
        if e.get("type") != "node" or not kind or not tags.get("name") or "lat" not in e:
            continue
        if math.hypot(*latlng_to_local(e["lat"], e["lon"], origin)) <= SURROUNDINGS_RADIUS_M:
            places.append({"name": str(tags["name"]), "kind": kind, "latlng": [round(e["lat"], 7), round(e["lon"], 7)]})
    places.sort(key=lambda p: (p["name"], p["kind"], p["latlng"]))

    features = [{"class": cls, "shape": shape, "name": name, "building": is_building, "own": False,
                 "latlng": [[round(v, 7) for v in local_to_latlng(x, y, origin)] for x, y in points]}
                for cls, shape, name, is_building, points in found]
    features.sort(key=lambda f: (CLASS_ORDER.index(f["class"]), f["shape"], f["latlng"], f["name"]))
    counts = {c: sum(1 for f in features if f["class"] == c) for c in CLASS_ORDER}
    return {"source": str(payload.get("source") or "OpenStreetMap"), "features": features, "places": places,
            "counts": counts}


def mark_own(surroundings: Optional[dict], outline_polygons_latlng: Iterable, centre_latlng) -> Optional[dict]:
    """A copy of ``surroundings`` with ``own`` set on the OSM buildings that are the modelled building.

    A building is the model's own when the model's centre lies in it, or when it holds the
    centroid of one of the chosen outline polygons (from Vexcel or OSM), or that polygon holds
    its centroid: two tracings of the same roof rarely agree to the metre, but their middles do.
    """
    if not surroundings:
        return surroundings
    origin = tuple(centre_latlng) if centre_latlng else None
    features = [dict(f) for f in surroundings.get("features") or []]
    if origin is not None:
        outlines = [[latlng_to_local(la, lo, origin) for la, lo in poly] for poly in outline_polygons_latlng or []]
        outlines = [o for o in outlines if len(o) >= 3]
        for f in features:
            if not f.get("building"):
                continue
            poly = [latlng_to_local(la, lo, origin) for la, lo in f["latlng"]]
            c = _centroid(poly)
            f["own"] = bool(_contains(poly, 0.0, 0.0)
                            or any(_contains(o, *c) or _contains(poly, *_centroid(o)) for o in outlines))
    out = dict(surroundings)
    out["features"] = features
    out["own_buildings"] = sum(1 for f in features if f.get("own"))
    return out


# ---------------------------------------------------------------------------
# windows -> what they face
# ---------------------------------------------------------------------------


def local_features(surroundings: Optional[dict], origin) -> list[tuple[str, bool, list]]:
    """``(class, closed, points in metres around origin)`` for every feature that is not the building itself."""
    out = []
    for f in (surroundings or {}).get("features") or []:
        if f.get("own"):
            continue
        pts = [latlng_to_local(la, lo, origin) for la, lo in f["latlng"]]
        if len(pts) >= 2:
            out.append((f["class"], f.get("shape") == "area", pts))
    return out


def ray_view(x: float, y: float, bearing_deg: float, features: list) -> Optional[tuple[str, float]]:
    """``(class, distance_m)`` of the first feature a ray meets within ``RANGE_M``, or None."""
    b = math.radians(bearing_deg)
    dx, dy = math.sin(b), math.cos(b)
    best: Optional[tuple[float, int, str]] = None
    for cls, closed, pts in features:
        if closed and _contains(pts, x, y):
            if cls != "parking_lot":
                continue  # standing inside a "neighbour": an unmarked tracing of this very building
            hit = 0.0     # the car park is drawn right up to (or under) the wall
        else:
            hit = _ray_hit(x, y, dx, dy, pts, closed)
        if hit is None or hit > RANGE_M:
            continue
        key = (round(hit, 3), CLASS_ORDER.index(cls), cls)
        if best is None or key < best:
            best = key
    return (best[2], best[0]) if best else None


def facing_for_windows(windows: Iterable[tuple[float, float, float]], features: list) -> Optional[str]:
    """The class most rays agree on, over every window of one room.

    ``windows``: ``(x, y, outward bearing)`` per window, in the same metres as ``features``.
    Ties go to the class seen nearest, then to ``CLASS_ORDER``. None when no ray meets anything.
    """
    votes: dict[str, int] = {}
    nearest: dict[str, float] = {}
    for x, y, bearing in windows:
        b = math.radians(bearing)
        sx, sy = x + math.sin(b) * START_OFFSET_M, y + math.cos(b) * START_OFFSET_M
        for off in FAN_DEG:
            seen = ray_view(sx, sy, bearing + off, features)
            if seen is None:
                continue
            cls, dist = seen
            votes[cls] = votes.get(cls, 0) + 1
            nearest[cls] = min(nearest.get(cls, float("inf")), dist)
    if not votes:
        return None
    return min(votes, key=lambda c: (-votes[c], nearest[c], CLASS_ORDER.index(c)))

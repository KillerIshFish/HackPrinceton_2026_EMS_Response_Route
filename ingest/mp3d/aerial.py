"""Aerial sidecar: street bearing, parcel and imagery for a built model.

Deliberately a *sidecar*, not part of :class:`~ingest.mp3d.model3d.Model3D`.
Tiles are network-fetched and time-varying - Vexcel reported 13 capture dates
for the reference house, newest 2026-01-29 - so folding them into the model
would destroy its byte-identical determinism. The model stays reproducible; the
aerial reference is stamped with ``captured_at`` and ``content_sha256`` and
stored beside it, keyed by model id.

What the EMS model actually consumes is ``street_bearing_deg`` plus
``bearing_source``. The image is provenance. The bearing feeds the existing
``--street-bearing`` seam in :mod:`ingest.mp3d.views`; it does not replace the
Entrance heuristic silently - :func:`reconcile_bearings` reports both and the
disagreement between them.

Why bearings come from OSM rather than pixels
---------------------------------------------
Vexcel serves parcel polygons and building footprints as vectors, but **no road
centrelines** (checked against the v2 API: ``/property/extract`` geometry is the
parcel, ``structures[].footprint`` the building). Getting a street direction out
of imagery therefore means detecting roads with ``/elements/detect``, which is
open-vocabulary CV with its own error rate. OSM road centrelines plus the
building's lat/long is one step of arithmetic. So:

* ``street_bearing_deg``  <- OSM centrelines  (``bearing_source="osm-centreline"``)
* aerial imagery          <- the street-vs-parking-lot distinction a caller
  describes, and staging-area placement
* parcel polygon          <- lot extent, for staging

Coverage ceiling
----------------
Only models carrying ``geolocation`` can be looked up at all: 791 of 1,000 in
the join scan, so roughly **79%**. :func:`coverage_report` measures the miss
rate rather than assuming it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ingest.mp3d import geometry as geo

VEXCEL_API = "https://api.vexcelgroup.com/v2"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "ingest.mp3d aerial sidecar/1.0"

# Pass every layer so the API picks the best imagery available at the point.
# urban = 7.5 cm ortho + obliques, urban-r = 7.5 cm ortho, wide-area = 15 cm.
ALL_LAYERS = "urban,urban-r,wide-area,urban-g"
OBLIQUE_DIRECTIONS = ("oblique-north", "oblique-south", "oblique-east", "oblique-west")

MAX_RETRIES = 5
BACKOFF_CAP_S = 30.0

# A service road is a driveway or an internal drive, not a street. Ranked last
# so a real street wins even when a drive is physically closer - which is the
# normal case for a house in a gated complex.
ROAD_CLASS_RANK = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
    "residential": 5, "unclassified": 6, "living_street": 7, "service": 8,
}
DEFAULT_SEARCH_RADIUS_M = 150.0

SOURCE_OSM = "osm-centreline"
SOURCE_ENTRANCE = "entrance-heuristic"
SOURCE_MANUAL = "manual"
SOURCE_UNKNOWN = "unknown"

EARTH_M_PER_DEG_LAT = 110540.0
EARTH_M_PER_DEG_LON = 111320.0


class AerialError(RuntimeError):
    """A transport or coverage failure in the aerial sidecar."""


# ---------------------------------------------------------------------------
# geometry helpers (pure)
# ---------------------------------------------------------------------------
def _local_xy(lat: float, lon: float, mid_lat_rad: float) -> tuple[float, float]:
    return (lon * EARTH_M_PER_DEG_LON * math.cos(mid_lat_rad),
            lat * EARTH_M_PER_DEG_LAT)


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2, degrees east of north."""
    mid = math.radians((lat1 + lat2) / 2.0)
    east = (lon2 - lon1) * EARTH_M_PER_DEG_LON * math.cos(mid)
    north = (lat2 - lat1) * EARTH_M_PER_DEG_LAT
    return math.degrees(math.atan2(east, north)) % 360.0


def point_to_segment(
    plat: float, plon: float,
    alat: float, alon: float, blat: float, blon: float,
) -> tuple[float, float, float]:
    """Distance in metres from P to segment AB, and the closest point on AB."""
    mid = math.radians((alat + blat) / 2.0)
    px, py = _local_xy(plat, plon, mid)
    ax, ay = _local_xy(alat, alon, mid)
    bx, by = _local_xy(blat, blon, mid)
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    cx, cy = ax + t * dx, ay + t * dy
    dist = math.hypot(px - cx, py - cy)
    return dist, cy / EARTH_M_PER_DEG_LAT, cx / (EARTH_M_PER_DEG_LON * math.cos(mid))


def angular_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def square_wkt(lat: float, lon: float, half_m: float) -> str:
    """WKT polygon of a square centred on the point. Longitude first."""
    dlat = half_m / EARTH_M_PER_DEG_LAT
    dlon = half_m / (EARTH_M_PER_DEG_LON * math.cos(math.radians(lat)))
    w, e = lon - dlon, lon + dlon
    s, n = lat - dlat, lat + dlat
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def point_wkt(lat: float, lon: float) -> str:
    """WKT point. Longitude first - transposing it is the classic bug."""
    return f"POINT({lon} {lat})"


# ---------------------------------------------------------------------------
# data shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadCandidate:
    """One nearby road centreline, as a direction from the building."""

    way_id: int
    name: str
    highway: str
    class_rank: int
    distance_m: float
    bearing_deg: float

    @property
    def is_service(self) -> bool:
        return self.highway == "service"


@dataclass(frozen=True)
class StreetBearing:
    """Which way the street lies, and how much to trust it."""

    bearing_deg: Optional[float]
    source: str
    confidence: str                      # "high" | "medium" | "low" | "none"
    ambiguous: bool = False
    note: str = ""
    candidates: tuple[RoadCandidate, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        if self.bearing_deg is None:
            return f"no street bearing ({self.source}): {self.note}"
        flag = " AMBIGUOUS" if self.ambiguous else ""
        return (f"{self.bearing_deg:.1f} deg via {self.source} "
                f"[{self.confidence}{flag}] {self.note}".rstrip())


@dataclass(frozen=True)
class AerialRef:
    """Sidecar record for one model. Never part of Model3D's canonical JSON."""

    model_id: str
    provider: str
    lat: float
    lng: float
    yaw_deg: float
    zoom: Optional[int]
    bbox: Optional[tuple[float, float, float, float]]   # w, s, e, n
    captured_at: Optional[str]
    content_sha256: Optional[str]
    local_path: Optional[str]
    street_bearing_deg: Optional[float]
    bearing_source: str
    # provenance that is not geometry
    layer: Optional[str] = None
    collection: Optional[str] = None
    parcel_wkt: Optional[str] = None
    oblique_directions: tuple[str, ...] = field(default_factory=tuple)
    bearing_confidence: str = SOURCE_UNKNOWN
    bearing_ambiguous: bool = False
    bearing_note: str = ""
    bbox_source: str = "none"
    gsd_m: Optional[float] = None
    road_candidates: tuple[RoadCandidate, ...] = field(default_factory=tuple)
    # lon/lat rings so a viewer can outline exactly which building is modelled
    footprint_lonlat: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # the modelled building's own extent - the highlight a viewer should draw
    outline_lonlat: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    parcel_lonlat: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    fetched_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# ---------------------------------------------------------------------------
# Vexcel client
# ---------------------------------------------------------------------------
class VexcelClient:
    """Vexcel REST API 2.0. Key goes in the X-API-KEY header, per the docs."""

    def __init__(self, api_key: Optional[str] = None, base: str = VEXCEL_API,
                 timeout_s: float = 180.0) -> None:
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except Exception:  # noqa: BLE001
                pass
            api_key = os.environ.get("VEXCEL_API_KEY")
        if not api_key:
            raise AerialError(
                "Missing VEXCEL_API_KEY. Set it in .env (see .env.example). "
                "The key is sent as the X-API-KEY header."
            )
        self.api_key = api_key
        self.base = base.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, params: dict, method: str = "GET",
                 body: Optional[dict] = None) -> tuple[int, str, bytes]:
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"X-API-KEY": self.api_key, "Accept": "*/*",
                   "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(MAX_RETRIES + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return resp.status, resp.headers.get("Content-Type", ""), resp.read()
            except urllib.error.HTTPError as exc:
                payload = exc.read()[:500]
                # 429 is the documented rate limit; back off ~1,2,4,8,16 s.
                if exc.code == 429 and attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, BACKOFF_CAP_S) + 0.25)
                    continue
                return exc.code, exc.headers.get("Content-Type", ""), payload
            except Exception as exc:  # noqa: BLE001
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, BACKOFF_CAP_S) + 0.25)
                    continue
                raise AerialError(f"{type(exc).__name__}: {exc}") from exc
        raise AerialError("request failed after retries")

    def property_extract(self, lat: float, lon: float,
                         layer: str = ALL_LAYERS) -> Optional[dict]:
        """Parcel polygon and attributes, or ``None`` when there is no parcel.

        A 404 here means "No available property data" - a road, water, open
        land, or outside the key's coverage. It is not an error: imagery and
        detection accept any WKT.
        """
        status, _ctype, body = self._request("property/extract", {
            "wkt": point_wkt(lat, lon), "srid": 4326, "layer": layer,
            "latest": "true", "include": "precomputed", "unit-system": "metric",
        })
        if status == 404:
            return None
        if status != 200:
            raise AerialError(f"property/extract HTTP {status}: {body[:200]!r}")
        return json.loads(body)

    def ortho_extract(self, wkt: str, layer: str = ALL_LAYERS,
                      buffer_m: float = 60.0, dimensions: str = "1024,1024",
                      image_format: str = "jpeg") -> bytes:
        """Top-down imagery as raw bytes. The endpoint returns an image, not JSON."""
        status, ctype, body = self._request("ortho/extract", {
            "wkt": wkt, "srid": 4326, "layer": layer,
            "buffer-meter": buffer_m, "crop": "bounding_box",
            "image-format": image_format, "destination-dimensions": dimensions,
        })
        if status != 200:
            raise AerialError(f"ortho/extract HTTP {status}: {body[:200]!r}")
        if "image" not in ctype:
            raise AerialError(f"ortho/extract returned {ctype}, not an image")
        return body

    def oblique_directions(self, wkt: str) -> tuple[tuple[str, ...], Optional[str]]:
        """Available oblique directions and the newest capture date.

        Obliques exist only in the ``urban`` layer, so a 404 means this place
        has none - rural areas covered by urban-r or wide-area never do.
        """
        status, _ctype, body = self._request("oriented/collections", {
            "wkt": wkt, "srid": 4326, "layer": "urban", "product-type": "oblique",
            "include": "product-type,collection,last-capture-date",
        })
        if status == 404:
            return (), None
        if status != 200:
            raise AerialError(f"oriented/collections HTTP {status}: {body[:200]!r}")
        features = (json.loads(body).get("features") or [])
        props = [f.get("properties") or {} for f in features]
        newest = max((p.get("last-capture-date") for p in props if p.get("last-capture-date")),
                     default=None)
        available = {p.get("product-type") for p in props
                     if p.get("last-capture-date") == newest}
        return tuple(d for d in OBLIQUE_DIRECTIONS if d in available), newest


# ---------------------------------------------------------------------------
# OSM road centrelines
# ---------------------------------------------------------------------------
def overpass_query(lat: float, lon: float,
                   radius_m: float = DEFAULT_SEARCH_RADIUS_M) -> str:
    classes = "|".join(ROAD_CLASS_RANK)
    return (f"[out:json][timeout:60];"
            f'way(around:{radius_m:.0f},{lat},{lon})["highway"~"^({classes})$"];'
            f"out geom tags;")


def fetch_osm_roads(lat: float, lon: float,
                    radius_m: float = DEFAULT_SEARCH_RADIUS_M,
                    url: str = OVERPASS_URL) -> dict:
    """Raw Overpass response. Retries: the public endpoint 502s under load."""
    body = urllib.parse.urlencode({"data": overpass_query(lat, lon, radius_m)}).encode()
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, BACKOFF_CAP_S) + 0.25)
                continue
            raise AerialError(f"Overpass failed: {type(exc).__name__}: {exc}") from exc
    raise AerialError("Overpass failed after retries")


def road_candidates_from_payload(payload: dict, lat: float, lon: float,
                                 ) -> list[RoadCandidate]:
    """Nearest approach of every road in an Overpass payload. Pure."""
    out: list[RoadCandidate] = []
    for way in payload.get("elements") or []:
        geometry = way.get("geometry") or []
        tags = way.get("tags") or {}
        highway = tags.get("highway")
        if highway not in ROAD_CLASS_RANK or len(geometry) < 2:
            continue
        best: Optional[tuple[float, float, float]] = None
        for a, b in zip(geometry, geometry[1:]):
            if None in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")):
                continue
            dist, clat, clon = point_to_segment(lat, lon, a["lat"], a["lon"],
                                                b["lat"], b["lon"])
            if best is None or dist < best[0]:
                best = (dist, clat, clon)
        if best is None:
            continue
        out.append(RoadCandidate(
            way_id=int(way.get("id") or 0),
            name=str(tags.get("name") or ""),
            highway=str(highway),
            class_rank=ROAD_CLASS_RANK[highway],
            distance_m=round(best[0], 2),
            bearing_deg=round(bearing_between(lat, lon, best[1], best[2]), 1),
        ))
    out.sort(key=lambda c: (c.class_rank, c.distance_m, c.way_id))
    return out


def street_bearing_from_roads(
    candidates: Sequence[RoadCandidate],
    ambiguity_gap_deg: float = 45.0,
    ambiguity_distance_ratio: float = 1.5,
) -> StreetBearing:
    """Pick the street direction, and say how much to trust it.

    A named, higher-class road wins over a closer service drive: in a gated
    complex the nearest centreline is an internal driveway on whichever side
    the geometry happens to favour. When the two best candidates are of the
    same class, comparably close, and point in materially different
    directions, the answer is flagged ambiguous rather than asserted.
    """
    if not candidates:
        return StreetBearing(None, SOURCE_OSM, "none",
                             note="no road centrelines within the search radius")

    ranked = sorted(candidates, key=lambda c: (c.class_rank, c.distance_m, c.way_id))
    best = ranked[0]

    # a named road of the same class is preferable to an unnamed one nearby
    same_class = [c for c in ranked if c.class_rank == best.class_rank]
    named = [c for c in same_class if c.name]
    if named and not best.name and named[0].distance_m <= best.distance_m * ambiguity_distance_ratio:
        best = named[0]

    rivals = [c for c in same_class
              if c.way_id != best.way_id
              and c.distance_m <= best.distance_m * ambiguity_distance_ratio
              and angular_gap(c.bearing_deg, best.bearing_deg) > ambiguity_gap_deg]
    ambiguous = bool(rivals)

    if best.is_service:
        confidence = "low"
        note = ("nearest centreline is a service road (driveway or internal "
                "drive), not a named street")
    elif best.name:
        confidence = "medium" if ambiguous else "high"
        note = f"{best.highway} {best.name!r} at {best.distance_m:.0f} m"
    else:
        confidence = "low" if ambiguous else "medium"
        note = f"unnamed {best.highway} at {best.distance_m:.0f} m"
    if ambiguous:
        note += (f"; {len(rivals)} comparable road(s) point elsewhere "
                 f"(e.g. {rivals[0].bearing_deg:.0f} deg at "
                 f"{rivals[0].distance_m:.0f} m)")

    return StreetBearing(
        bearing_deg=best.bearing_deg,
        source=SOURCE_OSM,
        confidence=confidence,
        ambiguous=ambiguous,
        note=note,
        candidates=tuple(ranked[:8]),
    )


def street_bearing_from_osm(lat: float, lon: float,
                            radius_m: float = DEFAULT_SEARCH_RADIUS_M) -> StreetBearing:
    """One network call, then pure selection."""
    payload = fetch_osm_roads(lat, lon, radius_m)
    return street_bearing_from_roads(road_candidates_from_payload(payload, lat, lon))


# ---------------------------------------------------------------------------
# reconciliation - never replace the heuristic silently
# ---------------------------------------------------------------------------
def _internal_drive_signature(osm: StreetBearing) -> Optional[RoadCandidate]:
    """The nearest service drive, when it is much closer than the chosen street.

    This is the gated-complex signature: a named public street 50 m away, and an
    unnamed internal drive 15 m away pointing a different way. The reference
    house matched it exactly, and aerial imagery confirmed the unit's door opens
    onto the internal drive while West 26th Street runs along the far side.
    """
    if osm.bearing_deg is None:
        return None
    chosen = next((c for c in osm.candidates
                   if abs(c.bearing_deg - osm.bearing_deg) < 0.05), None)
    services = [c for c in osm.candidates if c.is_service]
    if not chosen or chosen.is_service or not services:
        return None
    nearest_service = min(services, key=lambda c: c.distance_m)
    if nearest_service.distance_m * 2.0 <= chosen.distance_m:
        return nearest_service
    return None


def reconcile_bearings(osm: StreetBearing,
                       entrance_bearing_deg: Optional[float]) -> dict:
    """Compare the two sources and say what the disagreement means.

    Returns a verdict rather than a winner. A large gap usually does **not**
    mean one source is broken: the OSM bearing points at the nearest public
    street (the approach and staging direction) while the Entrance heuristic
    points at the unit's own frontage (what a caller actually sees). In a
    development with internal drives those are different directions and both
    are correct, so the two are reported as separate products.
    """
    result = {
        "osm_bearing_deg": osm.bearing_deg,
        "osm_confidence": osm.confidence,
        "osm_ambiguous": osm.ambiguous,
        "entrance_bearing_deg": entrance_bearing_deg,
        "gap_deg": None,
        "verdict": "",
        "recommended_source": SOURCE_UNKNOWN,
        "recommended_bearing_deg": None,
        # the two products, when they turn out to be different questions
        "approach_bearing_deg": None,   # where responders arrive from
        "frontage_bearing_deg": None,   # what the unit's own face looks onto
        "internal_drive_m": None,
    }
    if osm.bearing_deg is None and entrance_bearing_deg is None:
        result["verdict"] = "neither source produced a bearing"
        return result
    if osm.bearing_deg is None:
        result.update(verdict="only the Entrance heuristic produced a bearing",
                      recommended_source=SOURCE_ENTRANCE,
                      recommended_bearing_deg=entrance_bearing_deg)
        return result
    if entrance_bearing_deg is None:
        result.update(verdict="only OSM produced a bearing (no Entrance room)",
                      recommended_source=SOURCE_OSM,
                      recommended_bearing_deg=osm.bearing_deg)
        return result

    gap = angular_gap(osm.bearing_deg, entrance_bearing_deg)
    result["gap_deg"] = round(gap, 1)

    drive = _internal_drive_signature(osm)
    if drive is not None and gap >= 45.0:
        # Not a contradiction: the public street and the unit's frontage simply
        # face different ways. Keep both, labelled by what each is good for.
        result.update(
            approach_bearing_deg=osm.bearing_deg,
            frontage_bearing_deg=entrance_bearing_deg,
            internal_drive_m=drive.distance_m,
            verdict=(
                f"different questions, not a conflict: the nearest public street "
                f"is {osm.bearing_deg:.0f} deg away while an internal drive sits "
                f"{drive.distance_m:.0f} m off at {drive.bearing_deg:.0f} deg. Use "
                f"the OSM bearing for approach and staging, and the Entrance "
                f"bearing ({entrance_bearing_deg:.0f} deg) for what a caller sees"),
            recommended_source=SOURCE_ENTRANCE,
            recommended_bearing_deg=entrance_bearing_deg)
        return result

    if gap < 45.0:
        result.update(verdict="agree: both point at the same facade",
                      recommended_source=SOURCE_OSM,
                      recommended_bearing_deg=osm.bearing_deg)
    elif gap > 135.0:
        result.update(
            verdict=("opposite facades - one source has front and back swapped; "
                     "resolve before trusting either"),
            recommended_source=SOURCE_UNKNOWN, recommended_bearing_deg=None)
    else:
        result.update(
            verdict=("roughly perpendicular - the sources disagree about which "
                     "side is the street; neither is safe to use unchecked"),
            recommended_source=SOURCE_UNKNOWN, recommended_bearing_deg=None)
    if osm.ambiguous and result["recommended_source"] == SOURCE_OSM:
        result["verdict"] += "; OSM pick is itself ambiguous"
        result["recommended_source"] = SOURCE_UNKNOWN
        result["recommended_bearing_deg"] = None
    return result


# ---------------------------------------------------------------------------
# sidecar storage
# ---------------------------------------------------------------------------
def sidecar_dir(root: str | Path = "data/processed/aerial") -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sidecar_paths(model_id: str, root: str | Path = "data/processed/aerial",
                  image_ext: str = "jpg") -> tuple[Path, Path]:
    base = sidecar_dir(root)
    return base / f"{model_id}.json", base / f"{model_id}.{image_ext}"


def save_sidecar(ref: AerialRef, root: str | Path = "data/processed/aerial") -> Path:
    meta_path, _ = sidecar_paths(ref.model_id, root)
    with open(meta_path, "w", encoding="utf8", newline="\n") as handle:
        handle.write(ref.to_json())
    return meta_path


def load_sidecar(model_id: str,
                 root: str | Path = "data/processed/aerial") -> Optional[AerialRef]:
    meta_path, _ = sidecar_paths(model_id, root)
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf8"))
    data["road_candidates"] = tuple(
        RoadCandidate(**c) for c in (data.get("road_candidates") or []))
    data["oblique_directions"] = tuple(data.get("oblique_directions") or ())
    data["footprint_lonlat"] = tuple(tuple(p) for p in (data.get("footprint_lonlat") or ()))
    data["outline_lonlat"] = tuple(tuple(p) for p in (data.get("outline_lonlat") or ()))
    data["parcel_lonlat"] = tuple(tuple(p) for p in (data.get("parcel_lonlat") or ()))
    if data.get("bbox") is not None:
        data["bbox"] = tuple(data["bbox"])
    return AerialRef(**data)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def coverage_report(models: Iterable[dict]) -> dict:
    """Miss rate over models, where each is ``{"id":…, "geolocation":{lat,long}}``.

    A model with no ``geolocation`` cannot be looked up at all, so the aerial
    sidecar is optional per model by construction.
    """
    total = located = 0
    missing: list[str] = []
    for model in models:
        total += 1
        geo = model.get("geolocation") or {}
        lat, lng = geo.get("lat"), geo.get("long")
        if lat in (None, "", "null") or lng in (None, "", "null"):
            missing.append(str(model.get("id") or ""))
        else:
            located += 1
    return {
        "models": total,
        "with_geolocation": located,
        "without_geolocation": total - located,
        "coverage_share": round(located / total, 4) if total else 0.0,
        "missing_ids": missing[:50],
    }


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------
def build_aerial_ref(
    model_id: str,
    lat: float,
    lng: float,
    yaw_deg: float,
    entrance_bearing_deg: Optional[float] = None,
    client: Optional[VexcelClient] = None,
    root: str | Path = "data/processed/aerial",
    buffer_m: float = 60.0,
    dimensions: str = "1024,1024",
    fetch_image: bool = True,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    model=None,
) -> tuple[AerialRef, dict]:
    """Build the sidecar for one model. Returns ``(ref, reconciliation)``.

    Network: one Overpass call, and up to three Vexcel calls when
    ``fetch_image`` is set. Nothing here touches the deterministic model.
    """
    osm = street_bearing_from_osm(lat, lng, search_radius_m)
    verdict = reconcile_bearings(osm, entrance_bearing_deg)

    parcel_wkt = collection = captured_at = None
    footprint: list = []
    parcel: list = []
    bbox_source = "none"
    gsd_m = None
    obliques: tuple[str, ...] = ()
    sha = local_path = None
    bbox = None
    layer_used = None

    if fetch_image:
        client = client or VexcelClient()
        wkt = point_wkt(lat, lng)
        try:
            prop = client.property_extract(lat, lng)
        except AerialError:
            prop = None
        if prop and prop.get("features"):
            feature = prop["features"][0]
            attrs = feature.get("properties") or {}
            collection = attrs.get("metadata.collection")
            geometry = feature.get("geometry")
            if isinstance(geometry, str):
                parcel_wkt = geometry
            elif isinstance(geometry, dict):
                parcel_wkt = json.dumps(geometry)   # GeoJSON, kept verbatim
        footprint, parcel = select_footprint(prop, lat, lng)
        obliques, captured_at = client.oblique_directions(wkt)
        # Ask for the exact square we are going to record as the bbox, with no
        # extra buffer: then image pixels map to lon/lat by linear interpolation
        # and an overlay can be drawn on it without guesswork.
        frame = square_wkt(lat, lng, buffer_m)
        image = client.ortho_extract(frame, buffer_m=0.0, dimensions=dimensions)
        meta = parse_image_xmp(image)
        sha = hashlib.sha256(image).hexdigest()
        _, img_path = sidecar_paths(model_id, root)
        img_path.write_bytes(image)
        local_path = str(img_path)
        layer_used = ALL_LAYERS
        gsd_m = meta.get("min_gsd_m")
        if meta.get("bbox"):
            # what the file says it covers, not what we asked for
            bbox = tuple(round(v, 9) for v in meta["bbox"])
            bbox_source = "image-xmp"
        else:
            dlat = buffer_m / EARTH_M_PER_DEG_LAT
            dlon = buffer_m / (EARTH_M_PER_DEG_LON * math.cos(math.radians(lat)))
            bbox = (round(lng - dlon, 7), round(lat - dlat, 7),
                    round(lng + dlon, 7), round(lat + dlat, 7))
            bbox_source = "requested-square"
        # the ortho is its own capture, usually a different date to the obliques
        if meta.get("captured_at"):
            captured_at = meta["captured_at"]

    ref = AerialRef(
        model_id=model_id,
        provider="vexcel",
        lat=lat,
        lng=lng,
        yaw_deg=yaw_deg,
        zoom=None,
        bbox=bbox,
        captured_at=captured_at,
        content_sha256=sha,
        local_path=local_path,
        street_bearing_deg=osm.bearing_deg,
        bearing_source=osm.source if osm.bearing_deg is not None else SOURCE_UNKNOWN,
        layer=layer_used,
        collection=collection,
        parcel_wkt=parcel_wkt,
        oblique_directions=obliques,
        bearing_confidence=osm.confidence,
        bearing_ambiguous=osm.ambiguous,
        bearing_note=osm.note,
        bbox_source=bbox_source,
        gsd_m=gsd_m,
        road_candidates=osm.candidates,
        footprint_lonlat=tuple((round(x, 7), round(y, 7)) for x, y in footprint),
        outline_lonlat=tuple(model_outline_lonlat(model, lat, lng, yaw_deg)
                             if model is not None else ()),
        parcel_lonlat=tuple((round(x, 7), round(y, 7)) for x, y in parcel),
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return ref, verdict


def to_viewer_payload(ref: AerialRef, verdict: Optional[dict] = None,
                      image_href: str = "aerial.jpg") -> dict:
    """Compact shape for the browser viewer.

    The artifact CSP blocks ``api.vexcelgroup.com``, so the page can never fetch
    a tile itself. The image ships alongside the page as a published file and
    this payload only describes it.
    """
    return {
        "modelId": ref.model_id,
        "provider": ref.provider,
        "image": image_href,
        "lat": ref.lat,
        "lng": ref.lng,
        "yaw": ref.yaw_deg,
        "bbox": list(ref.bbox) if ref.bbox else None,
        "capturedAt": ref.captured_at,
        "collection": ref.collection,
        "sha256": (ref.content_sha256 or "")[:16],
        "bboxSource": ref.bbox_source,
        "gsd": ref.gsd_m,
        "obliques": list(ref.oblique_directions),
        "streetBearing": ref.street_bearing_deg,
        "bearingSource": ref.bearing_source,
        "bearingConfidence": ref.bearing_confidence,
        "bearingAmbiguous": ref.bearing_ambiguous,
        "bearingNote": ref.bearing_note,
        "outline": [list(p) for p in ref.outline_lonlat],
        "footprint": [list(p) for p in ref.footprint_lonlat],
        "parcel": [list(p) for p in ref.parcel_lonlat],
        "roads": [
            {"name": c.name or "(unnamed)", "highway": c.highway,
             "dist": c.distance_m, "bearing": c.bearing_deg}
            for c in ref.road_candidates[:6]
        ],
        "verdict": verdict or {},
    }


def write_viewer_payload(ref: AerialRef, path: str | Path,
                         verdict: Optional[dict] = None,
                         var: str = "MP3D_AERIAL",
                         image_href: str = "aerial.jpg") -> Path:
    """Write the aerial payload as a JS file assigning ``window.<var>``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(to_viewer_payload(ref, verdict, image_href),
                      separators=(",", ":"), sort_keys=True)
    with open(target, "w", encoding="utf8", newline="\n") as handle:
        handle.write(f"window.{var}={body};\n")
    return target


# ---------------------------------------------------------------------------
# footprint selection - which of the returned structures is *this* building
# ---------------------------------------------------------------------------
def parse_polygon_wkt(wkt: str) -> list[tuple[float, float]]:
    """Outer ring of a WKT POLYGON as [(lon, lat), ...]. Pure."""
    if not wkt or "(" not in wkt:
        return []
    inner = wkt[wkt.index("((") + 2:]
    inner = inner[:inner.index("))")] if "))" in inner else inner
    ring: list[tuple[float, float]] = []
    for pair in inner.split(","):
        bits = pair.strip().split()
        if len(bits) >= 2:
            try:
                ring.append((float(bits[0]), float(bits[1])))
            except ValueError:
                continue
    return ring


def point_in_ring(lon: float, lat: float,
                  ring: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting test in lon/lat space."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < cross:
                inside = not inside
    return inside


def _ring_centroid(ring: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))


def select_footprint(property_json: Optional[dict], lat: float, lon: float,
                     ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """``(footprint_ring, parcel_ring)`` for the structure at this point.

    A point query in a dense terrace returns every nearby parcel - nine for the
    reference house, all flagged ``inside-parcel``. Containment picks the right
    one; the nearest centroid is the fallback when the point falls in a gap
    between footprints.
    """
    if not property_json:
        return [], []
    best_near: Optional[tuple[float, list, list]] = None

    for feature in property_json.get("features") or []:
        attrs = feature.get("properties") or {}
        parcel_ring: list[tuple[float, float]] = []
        geom = feature.get("geometry")
        if isinstance(geom, dict) and geom.get("coordinates"):
            coords = geom["coordinates"]
            # MultiPolygon -> first ring of the first polygon
            while isinstance(coords, list) and coords and isinstance(coords[0], list) \
                    and coords[0] and isinstance(coords[0][0], list):
                coords = coords[0]
            if isinstance(coords, list):
                parcel_ring = [(float(c[0]), float(c[1])) for c in coords
                               if isinstance(c, (list, tuple)) and len(c) >= 2]

        for structure in attrs.get("structures") or []:
            ring = parse_polygon_wkt(structure.get("footprint.wkt") or "")
            if len(ring) < 3:
                continue
            if point_in_ring(lon, lat, ring):
                return ring, parcel_ring
            cx, cy = _ring_centroid(ring)
            gap = math.hypot((cx - lon) * EARTH_M_PER_DEG_LON * math.cos(math.radians(lat)),
                             (cy - lat) * EARTH_M_PER_DEG_LAT)
            if best_near is None or gap < best_near[0]:
                best_near = (gap, ring, parcel_ring)

    if best_near:
        return best_near[1], best_near[2]
    return [], []


# ---------------------------------------------------------------------------
# authoritative image extent, read back out of the returned file
# ---------------------------------------------------------------------------
XMP_NS = b"http://ns.adobe.com/xap/1.0/"


def parse_image_xmp(image_bytes: bytes) -> dict:
    """Georeferencing Vexcel embeds in the image's XMP packet.

    The docs say image metadata rides in the file's XMP headers, and it is the
    only *authoritative* statement of what ground area the pixels cover. A bbox
    computed from the request is an approximation - measured 0.7% narrow in
    longitude on the reference house, about 7 px at 1024 wide, which is enough
    to visibly shift an overlay off a townhouse.
    """
    start = image_bytes.find(XMP_NS)
    if start < 0:
        return {}
    chunk = image_bytes[start:start + 20000].decode("utf8", "ignore")

    def tag(name: str) -> Optional[str]:
        open_tag = f"<Vexcel:{name}>"
        close_tag = f"</Vexcel:{name}>"
        i = chunk.find(open_tag)
        if i < 0:
            return None
        j = chunk.find(close_tag, i)
        return chunk[i + len(open_tag): j] if j > i else None

    out: dict = {}
    geometry = tag("geometry")
    if geometry:
        ring = parse_polygon_wkt(geometry)
        if len(ring) >= 4:
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            out["bbox"] = (min(lons), min(lats), max(lons), max(lats))
            out["geometry_wkt"] = geometry
    for key, name in (("captured_at", "last-capture-date"),
                      ("first_captured_at", "first-capture-date"),
                      ("product_type", "product-type"),
                      ("srid", "srid")):
        value = tag(name)
        if value:
            out[key] = value
    for key, name in (("min_gsd_m", "min-gsd"), ("max_gsd_m", "max-gsd")):
        value = tag(name)
        if value:
            try:
                out[key] = float(value)
            except ValueError:
                pass
    return out


def model_outline_lonlat(model, lat: float, lng: float,
                         yaw_deg: float) -> list[tuple[float, float]]:
    """The modelled building's own outline, placed on the map.

    Preferred over Vexcel's ``structures[].footprint.wkt`` for highlighting.
    A point query returns every nearby detection - nine heavily overlapping
    polygons for the reference terrace, scattered by about 5 m - so picking one
    by containment is close to arbitrary. The scan's own ground-floor extent is
    measured geometry we have already validated, and it matched the best Vexcel
    candidate's size to within 0.5 m.

    Positioning still rests on the scan's ``geolocation``, which is good to a
    few metres; the shape and orientation are exact.
    """
    ground = min((s.floor for s in model.slabs), default=0)
    points: list[tuple[float, float]] = []
    for room in model.rooms:
        if room.floor == ground:
            points.extend(room.polygon)
    if len(points) < 3:
        points = [w.a for w in model.walls if w.floor == ground] + \
                 [w.b for w in model.walls if w.floor == ground]
    hull = geo.convex_hull(points)
    if len(hull) < 3:
        return []

    theta = math.radians(yaw_deg)
    world = [(x * math.cos(theta) - y * math.sin(theta),
              x * math.sin(theta) + y * math.cos(theta)) for x, y in hull]
    cx = sum(p[0] for p in world) / len(world)
    cy = sum(p[1] for p in world) / len(world)
    mlat = math.radians(lat)
    return [(round(lng + (e - cx) / (EARTH_M_PER_DEG_LON * math.cos(mlat)), 7),
             round(lat + (n - cy) / EARTH_M_PER_DEG_LAT, 7))
            for e, n in world]

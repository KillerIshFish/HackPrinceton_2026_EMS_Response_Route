"""Vexcel aerial imagery (API v2): ortho tiles for the site around a building.

What is real here
-----------------
* ``fetch_tile``: one 256 px ortho tile from the Vexcel WMTS endpoint, cached on
  disk under ``<VEXCEL_DATA_DIR>/tiles/<layer>/<z>/<x>/<y>.jpg``. The API key
  (``VEXCEL_API_KEY`` in ``.env``) never leaves the backend: the web client asks
  ``api/server.py`` for tiles and the server reads the cache or calls Vexcel.
* ``site_tiles``: which tiles cover a circle around a lat/lng, plus the
  Web-Mercator maths the UI needs to draw entrances and staging spots on top.
* ``collections_at``: capture metadata (collection name, product type) at a point.
* ``crop_aerial_tile``: stitch those tiles into one image (needs Pillow, which is
  a dev dependency; only used offline to inspect a site).

What is still manual
--------------------
Which facade looks at a street and which at parking, and where the entrances
and staging spots are, is annotated per building in
``configs/real_buildings.yaml`` from this imagery. ``classify_surroundings`` and
``window_ray_intersections`` (automatic ground-class labelling and window rays)
remain stubs.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from core.schema import Building

API_BASE = "https://api.vexcelgroup.com/v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYER = "urban"
TILE_PX = 256  # standard WMTS / Web-Mercator tile size
# Deepest WMTS zoom accepted per layer. At the equator zoom 20 is ~15 cm/px and zoom 21 ~7.5 cm/px, matching the
# wide-area (15 cm) and urban (7.5 cm) resolutions noted in ingest.mp3d.aerial; requests are clamped to these.
MAX_ZOOM = {"urban": 21, "urban-r": 22, "urban-g": 22, "graysky": 21, "wide-area": 20}
EARTH_RADIUS_M = 6378137.0  # WGS-84 equatorial radius, the sphere Web-Mercator is defined on


class VexcelError(RuntimeError):
    pass


def vexcel_data_dir() -> Path:
    raw = os.environ.get("VEXCEL_DATA_DIR", "./data/raw/vexcel")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def api_key() -> str:
    """The key from ``.env`` with any whitespace removed.

    Keys pasted from a wrapped line arrive with spaces in them; Vexcel rejects
    those with 401, so all whitespace is dropped.
    """
    load_dotenv(REPO_ROOT / ".env")
    key = "".join((os.environ.get("VEXCEL_API_KEY") or "").split())
    if not key:
        raise VexcelError("VEXCEL_API_KEY is not set in .env")
    return key


# Assumed: 45 s per request, no retry. Tiles and collections are cached on disk, so a failure is just re-requested.
def _get(path: str, params: dict[str, str], timeout: float = 45.0) -> tuple[bytes, str]:
    url = f"{API_BASE}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"x-api-key": api_key(), "User-Agent": "caller-location-predictor/0.1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raise VexcelError(f"Vexcel API HTTP {exc.code} for {path}") from exc
    except urllib.error.URLError as exc:
        raise VexcelError(f"Vexcel API unreachable: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Web-Mercator tile maths (EPSG:3857, the tile matrix set of every Vexcel layer)
# ---------------------------------------------------------------------------


def latlng_to_tile(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    """Fractional tile column/row of a WGS84 point."""
    n = 2.0**zoom
    x = (lng + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_to_latlng(x: float, y: float, zoom: int) -> tuple[float, float]:
    """WGS84 (lat, lng) of a fractional tile column/row (its north-west corner for integers)."""
    n = 2.0**zoom
    lng = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lng


def metres_per_pixel(lat: float, zoom: int) -> float:
    return 2.0 * math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat)) / (TILE_PX * 2.0**zoom)


def site_tiles(center_latlng: Sequence[float], radius_m: float = 90.0, zoom: int = 20,
               layer: str = DEFAULT_LAYER) -> dict:
    """Tile range covering a circle of ``radius_m`` around ``center_latlng``.

    Returns ``{layer, zoom, x_min, x_max, y_min, y_max, tile_px, m_per_px,
    north_west: [lat, lng], south_east: [lat, lng]}`` (ranges inclusive).
    """
    zoom = min(int(zoom), MAX_ZOOM.get(layer, 20))
    lat, lng = float(center_latlng[0]), float(center_latlng[1])
    mpp = metres_per_pixel(lat, zoom)
    cx, cy = latlng_to_tile(lat, lng, zoom)
    half = radius_m / mpp / TILE_PX
    x_min, x_max = math.floor(cx - half), math.floor(cx + half)
    y_min, y_max = math.floor(cy - half), math.floor(cy + half)
    return {
        "layer": layer,
        "zoom": zoom,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "tile_px": TILE_PX,
        "m_per_px": round(mpp, 5),
        "north_west": list(tile_to_latlng(x_min, y_min, zoom)),
        "south_east": list(tile_to_latlng(x_max + 1, y_max + 1, zoom)),
    }


# ---------------------------------------------------------------------------
# Network calls (all cached on disk)
# ---------------------------------------------------------------------------


def tile_path(layer: str, zoom: int, x: int, y: int) -> Path:
    return vexcel_data_dir() / "tiles" / layer / str(zoom) / str(x) / f"{y}.jpg"


def fetch_tile(layer: str, zoom: int, x: int, y: int) -> bytes:
    """JPEG bytes of one ortho tile; read from the disk cache when present."""
    if layer not in MAX_ZOOM:
        raise VexcelError(f"Unknown Vexcel layer '{layer}'")
    if not 0 <= zoom <= MAX_ZOOM[layer]:
        raise VexcelError(f"Zoom {zoom} is outside 0..{MAX_ZOOM[layer]} for layer '{layer}'")
    path = tile_path(layer, zoom, x, y)
    if path.is_file():
        return path.read_bytes()
    body, ctype = _get(
        "ortho/wmts",
        {
            "service": "wmts", "request": "getTile", "layer": layer, "Style": "RGB",
            "TileMatrixSet": layer, "TileMatrix": str(zoom), "TileRow": str(y), "TileCol": str(x),
            "format": "image/jpeg",
        },
    )
    if "image" not in ctype:
        raise VexcelError(f"Vexcel returned {ctype or 'no content type'} instead of an image for tile {zoom}/{x}/{y}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def collections_at(lat: float, lng: float) -> list[dict]:
    """Ortho collections covering a point: ``[{"collection": ..., "product-type": ...}, ...]``."""
    # 5 decimal places (~1 m) in the cache key: nearby queries for the same building share one file.
    cache = vexcel_data_dir() / "collections" / f"{lat:.5f}_{lng:.5f}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    body, _ = _get("ortho/collections", {"wkt": f"POINT({lng} {lat})", "srid": "4326"})
    features = json.loads(body.decode()).get("features") or []
    out = [f.get("properties") or {} for f in features]  # geometry is large and not needed
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def crop_aerial_tile(center_latlng: Sequence[float], radius_m: float = 120.0, zoom: int = 20,
                     layer: str = DEFAULT_LAYER) -> Path:
    """Stitch the tiles around a building into one JPEG; return its path.

    Written to ``<VEXCEL_DATA_DIR>/crops/``. Needs Pillow (dev dependency).
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise VexcelError("crop_aerial_tile needs Pillow: run `uv sync` (it is in the dev group)") from exc
    import io

    t = site_tiles(center_latlng, radius_m, zoom, layer)
    cols, rows = t["x_max"] - t["x_min"] + 1, t["y_max"] - t["y_min"] + 1
    mosaic = Image.new("RGB", (cols * TILE_PX, rows * TILE_PX))
    for x in range(t["x_min"], t["x_max"] + 1):
        for y in range(t["y_min"], t["y_max"] + 1):
            tile = Image.open(io.BytesIO(fetch_tile(layer, t["zoom"], x, y)))
            mosaic.paste(tile, ((x - t["x_min"]) * TILE_PX, (y - t["y_min"]) * TILE_PX))
    out = vexcel_data_dir() / "crops" / (
        f"{layer}_z{t['zoom']}_{center_latlng[0]:.5f}_{center_latlng[1]:.5f}_r{int(radius_m)}.jpg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out, quality=90)
    return out


# ---------------------------------------------------------------------------
# Still stubs: automatic labelling of what each window looks at
# ---------------------------------------------------------------------------


def classify_surroundings(tile_path: Path) -> dict:
    """STUB - ground-class polygons (parking_lot, street, ...) in lat/lng around the building."""
    raise NotImplementedError(
        "Automatic ground-class labelling is not implemented; facade classes are annotated "
        "per building in configs/real_buildings.yaml."
    )


def window_ray_intersections(building: Building, surroundings: dict, max_range_m: float = 150.0) -> dict[str, str | None]:
    """STUB - ``{room_id: window_facing}`` from window rays x ground classes."""
    raise NotImplementedError("Window-ray intersection needs georegistered windows + labelled tiles.")


def apply_window_facing(building: Building, facing: dict[str, str | None]) -> Building:
    """Return a copy of ``building`` with ``Room.window_facing`` overridden (real helper)."""
    out = building.model_copy(deep=True)
    for room in out.rooms:
        if room.id in facing and room.has_windows:
            room.window_facing = facing[room.id]
    return out


if __name__ == "__main__":
    import sys

    lat, lng = float(sys.argv[1]), float(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    zoom = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    print("collections:", [c.get("collection") for c in collections_at(lat, lng)][:6])
    print("crop:", crop_aerial_tile((lat, lng), radius, zoom))

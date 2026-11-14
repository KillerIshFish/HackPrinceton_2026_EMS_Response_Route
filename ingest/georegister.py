"""Model (x, y, z) <-> lat/lng transforms.

v0: a real, simple local-tangent-plane (equirectangular) transform that is accurate
to centimetres at building scale. Building-local metres are x = east, y = north when
``rotation_deg == 0``; ``rotation_deg`` is the counter-clockwise rotation of the
building's local +x axis from true east.

TODO (needs real data): Matterport model space is arbitrary (origin = first scan
position, arbitrary yaw). ``register_matterport_model`` will fit a similarity
transform from surveyed control points / Vexcel footprint corners.
"""

from __future__ import annotations

import math
from typing import Sequence

EARTH_RADIUS_M = 6_378_137.0  # WGS-84 equatorial radius


def local_to_latlng(
    x: float,
    y: float,
    origin_latlng: Sequence[float],
    rotation_deg: float = 0.0,
) -> tuple[float, float]:
    """Building-local metres -> (lat, lng) in degrees."""
    lat0, lng0 = float(origin_latlng[0]), float(origin_latlng[1])
    theta = math.radians(rotation_deg)
    east = x * math.cos(theta) - y * math.sin(theta)
    north = x * math.sin(theta) + y * math.cos(theta)
    lat = lat0 + math.degrees(north / EARTH_RADIUS_M)
    lng = lng0 + math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return (lat, lng)


def latlng_to_local(
    lat: float,
    lng: float,
    origin_latlng: Sequence[float],
    rotation_deg: float = 0.0,
) -> tuple[float, float]:
    """(lat, lng) in degrees -> building-local metres. Inverse of ``local_to_latlng``."""
    lat0, lng0 = float(origin_latlng[0]), float(origin_latlng[1])
    north = math.radians(lat - lat0) * EARTH_RADIUS_M
    east = math.radians(lng - lng0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    theta = math.radians(rotation_deg)
    x = east * math.cos(theta) + north * math.sin(theta)
    y = -east * math.sin(theta) + north * math.cos(theta)
    return (x, y)


def register_matterport_model(control_points: list[dict]) -> dict:
    """STUB - fit model-space -> building-local/geodetic alignment for a real scan.

    Expected input (to be confirmed against the real dataset)::

        [{"model_xyz": [x, y, z], "latlng": [lat, lng], "elevation_m": float}, ...]

    Should return ``{"origin_latlng": [lat, lng], "rotation_deg": float,
    "scale": float, "z_offset_m": float}`` usable with ``local_to_latlng``.
    """
    raise NotImplementedError(
        "Matterport geo-registration needs real control points (scan pose metadata "
        "or Vexcel footprint corners). Provide the dataset before implementing."
    )

"""Model (x, y, z) <-> lat/lng transforms.

v0: a real, simple local-tangent-plane (equirectangular) transform that is accurate
to centimetres at building scale. Building-local metres are x = east, y = north when
``rotation_deg == 0``; ``rotation_deg`` is the counter-clockwise rotation of the
building's local +x axis from true east.

Matterport model space is arbitrary (origin = first scan position, arbitrary
yaw). ``register_matterport_model`` fits where a scan sits from points matched
between the scan and aerial imagery (building corners read off Vexcel tiles).
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
    """Fit where a Matterport scan sits on the ground from matched points.

    ``control_points`` is ``[{"model_xy": [x, y], "latlng": [lat, lng]}, ...]``:
    points you can identify both in the scan (model-space metres) and on aerial
    imagery, typically building corners. At least two are needed.

    Fits a rigid transform (rotation + translation, scale fixed at 1 because
    scans are metric) by least squares and returns ``{"origin_latlng": [lat, lng],
    "rotation_deg": float, "rms_m": float}``: the lat/lng of model point (0, 0),
    the counter-clockwise angle of the model's +x axis from east, and the residual.
    The result plugs straight into ``local_to_latlng`` / ``latlng_to_local``.
    """
    if len(control_points) < 2:
        raise ValueError("Geo-registration needs at least two control points")
    ref = control_points[0]["latlng"]
    model = [(float(p["model_xy"][0]), float(p["model_xy"][1])) for p in control_points]
    world = [latlng_to_local(p["latlng"][0], p["latlng"][1], ref, 0.0) for p in control_points]  # east, north
    n = len(model)
    mcx, mcy = sum(p[0] for p in model) / n, sum(p[1] for p in model) / n
    wcx, wcy = sum(p[0] for p in world) / n, sum(p[1] for p in world) / n
    # 2D Kabsch: theta maximises sum of dot(R(theta) * m_i, w_i) over centred points
    s_cos = sum((m[0] - mcx) * (w[0] - wcx) + (m[1] - mcy) * (w[1] - wcy) for m, w in zip(model, world))
    s_sin = sum((m[0] - mcx) * (w[1] - wcy) - (m[1] - mcy) * (w[0] - wcx) for m, w in zip(model, world))
    theta = math.atan2(s_sin, s_cos)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # world position of the model origin
    ox = wcx - (mcx * cos_t - mcy * sin_t)
    oy = wcy - (mcx * sin_t + mcy * cos_t)
    sq = 0.0
    for m, w in zip(model, world):
        ex = ox + m[0] * cos_t - m[1] * sin_t - w[0]
        ey = oy + m[0] * sin_t + m[1] * cos_t - w[1]
        sq += ex * ex + ey * ey
    origin = local_to_latlng(ox, oy, ref, 0.0)
    return {
        "origin_latlng": [round(origin[0], 7), round(origin[1], 7)],
        "rotation_deg": round(math.degrees(theta), 2),
        "rms_m": round(math.sqrt(sq / n), 2),
    }

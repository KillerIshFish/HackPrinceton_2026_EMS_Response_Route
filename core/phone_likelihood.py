"""Phone-fix likelihood per room.

    L_phone(r) = L_h(r) * L_z(r)

    L_z(r) = exp(-0.5 * ((z_fix - (floor_height_r + 1.2)) / sigma_z)^2),  sigma_z = 3 / 1.28 ≈ 2.34 m
    L_h(r) = exp(-0.5 * (d(fix_xy, polygon_r) / sigma_h)^2),  d = 0 inside the room

A missing fix, or a missing horizontal / vertical component, contributes a flat
likelihood of 1.0 for that component.
"""

from __future__ import annotations

import math
from typing import Sequence

from core.schema import Building, PhoneFix

# Assumed phone height above the floor slab (held or pocketed); same value as core.features.
PHONE_HEIGHT_ABOVE_FLOOR_M = 1.2
# Assumed, from the initial commit: tighter than the FCC horizontal benchmark (50 m for 80 % of calls,
# cited in eval/benchmark/common.py), which uses 30 m. Real fixes carry their own horizontal_sigma_m.
DEFAULT_HORIZONTAL_SIGMA_M = 15.0  # used when x/y are given without an uncertainty
MIN_LIKELIHOOD = 1e-12  # keep likelihoods strictly positive, so a far-off fix never zeroes a room


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> float:
    """Distance (m) from a point to a polygon; 0 if the point is inside it."""
    if len(polygon) == 0:
        return float("inf")
    if len(polygon) >= 3 and point_in_polygon(x, y, polygon):
        return 0.0
    n = len(polygon)
    return min(
        _point_segment_distance(x, y, *polygon[i], *polygon[(i + 1) % n]) for i in range(n)
    )


def _gaussian(delta: float, sigma: float) -> float:
    return math.exp(-0.5 * (delta / sigma) ** 2)


def phone_likelihood(building: Building, fix: PhoneFix | None) -> dict[str, float]:
    """L_phone(r) in (0, 1] for every room; all 1.0 when there is nothing to score."""
    liks = {room.id: 1.0 for room in building.rooms}
    if fix is None:
        return liks

    has_z = fix.z_m is not None and fix.z_sigma_m is not None and fix.z_sigma_m > 0
    has_xy = fix.x_m is not None and fix.y_m is not None
    sigma_h = fix.horizontal_sigma_m if fix.horizontal_sigma_m and fix.horizontal_sigma_m > 0 else DEFAULT_HORIZONTAL_SIGMA_M

    for room in building.rooms:
        lik = 1.0
        if has_z:
            floor_z = building.floor_heights_m.get(room.floor)
            if floor_z is None:  # fall back to the room's own centroid height
                floor_z = room.centroid[2]
            lik *= _gaussian(fix.z_m - (floor_z + PHONE_HEIGHT_ABOVE_FLOOR_M), fix.z_sigma_m)
        if has_xy:
            lik *= _gaussian(distance_to_polygon(fix.x_m, fix.y_m, room.polygon), sigma_h)
        liks[room.id] = max(lik, MIN_LIKELIHOOD)
    return liks

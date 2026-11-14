"""Room prior before any evidence.

    pi(r) ∝ area_r^alpha * w(type_r, use, hour) * o(suite_r)

with alpha = 0.7, w from configs/priors.yaml (base room-type weight for the
building use x time-of-day multiplier) and o = Room.occupancy_factor.
``P_NOT_IN_MODEL`` (5%) of the mass is reserved for "caller is not in any
modelled room"; room probabilities sum to 1 - P_NOT_IN_MODEL.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from core.schema import Building

ALPHA = 0.7
P_NOT_IN_MODEL = 0.05
DEFAULT_ROOM_WEIGHT = 0.3
PRIORS_PATH = Path(__file__).resolve().parents[1] / "configs" / "priors.yaml"


@lru_cache(maxsize=8)
def _load_priors_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_priors(path: str | Path | None = None) -> dict:
    """Load configs/priors.yaml (or ``path``) as a dict."""
    return _load_priors_cached(str(Path(path) if path is not None else PRIORS_PATH))


def hour_band(hour: int, priors_cfg: dict) -> str:
    """Name of the hour band containing ``hour`` ([start, end), may wrap midnight)."""
    hour = int(hour) % 24
    for name, (start, end) in (priors_cfg.get("hour_bands") or {}).items():
        if start <= end:
            if start <= hour < end:
                return name
        elif hour >= start or hour < end:
            return name
    return "business"


def type_weight(room_type: str, use_type: str, hour: int, priors_cfg: dict) -> float:
    """w(type, use, hour) = base room-type weight for the use x time-of-day multiplier."""
    default_w = float(priors_cfg.get("default_room_weight", DEFAULT_ROOM_WEIGHT))
    weights = priors_cfg.get("room_weights") or {}
    use_table = weights.get(use_type) or weights.get("default") or {}
    base = float(use_table.get(room_type, (weights.get("default") or {}).get(room_type, default_w)))

    mults = priors_cfg.get("time_multipliers") or {}
    band_table = (mults.get(use_type) or mults.get("default") or {}).get(hour_band(hour, priors_cfg)) or {}
    mult = float(band_table.get(room_type, band_table.get("default", 1.0)))
    return base * mult


def compute_prior(
    building: Building, hour: int = 14, priors_cfg: dict | None = None
) -> tuple[dict[str, float], float]:
    """Return (room_probs, p_not_in_model).

    pi(r) ∝ area^alpha * w(type, use, hour) * occupancy_factor, scaled so that
    sum(room_probs) == 1 - p_not_in_model.
    """
    cfg = priors_cfg if priors_cfg is not None else load_priors()
    alpha = float(cfg.get("alpha", ALPHA))
    p_nim = float(cfg.get("p_not_in_model", P_NOT_IN_MODEL))

    raw: dict[str, float] = {}
    for room in building.rooms:
        w = type_weight(room.room_type, building.use_type, hour, cfg)
        raw[room.id] = max(room.area_m2, 0.0) ** alpha * w * max(room.occupancy_factor, 0.0)

    total = sum(raw.values())
    if not raw:
        return {}, 1.0
    if total <= 0.0:  # degenerate config: fall back to uniform over rooms
        raw = {rid: 1.0 for rid in raw}
        total = float(len(raw))
    scale = (1.0 - p_nim) / total
    return {rid: v * scale for rid, v in raw.items()}, p_nim

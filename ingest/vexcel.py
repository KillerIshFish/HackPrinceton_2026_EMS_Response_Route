"""Vexcel aerial imagery - STUB (v0 uses the synthetic building's window_facing).

Planned pipeline once tiles are available under ``VEXCEL_DATA_DIR``:

1. ``crop_aerial_tile``: crop ortho/oblique tiles to the building footprint plus
   a margin (parking lots, streets, entrances, staging areas).
2. ``classify_surroundings``: label what lies on each side of the footprint
   (parking_lot / street / trees / neighbouring building ...) using the view keys
   of ``configs/clue_vocab.yaml``.
3. ``window_ray_intersections``: cast a ray outward from every exterior window
   (from the Matterport model, georegistered) and intersect it with the labelled
   ground classes -> ``Room.window_facing``.

Nothing here touches the network in v0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from core.schema import Building


def vexcel_data_dir() -> Path:
    return Path(os.environ.get("VEXCEL_DATA_DIR", "./data/raw/vexcel"))


def crop_aerial_tile(center_latlng: Sequence[float], radius_m: float = 120.0) -> Path:
    """STUB - crop the aerial tile(s) around a building; return the cropped image path."""
    raise NotImplementedError("Vexcel tile cropping needs the real tiles + their georeference.")


def classify_surroundings(tile_path: Path) -> dict:
    """STUB - ground-class polygons (parking_lot, street, ...) in lat/lng around the building."""
    raise NotImplementedError("Aerial ground-class labelling is not part of v0.")


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

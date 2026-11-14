"""CoStar occupancy / tenant metadata - STUB (v0 uses occupancy_factor from the YAML).

Planned once exports are available under ``COSTAR_DATA_DIR``:

* ``load_costar_records``: read the property + suite export (CSV/XLSX).
* ``occupancy_factors``: map suites -> rooms and turn vacancy / tenant type /
  operating hours into ``Room.occupancy_factor`` (the ``o(suite)`` term of the
  prior) and confirm ``Building.use_type``.

Nothing here touches the network in v0.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.schema import Building


def costar_data_dir() -> Path:
    return Path(os.environ.get("COSTAR_DATA_DIR", "./data/raw/costar"))


def load_costar_records(path: os.PathLike | str | None = None) -> list[dict]:
    """STUB - parse a CoStar export into suite-level records.

    Expected record (to be confirmed against the real export)::

        {"property_id": str, "address": str, "use_type": str, "suite": str,
         "floor": int, "area_m2": float, "vacant": bool, "tenant_type": str,
         "operating_hours": str}
    """
    raise NotImplementedError("CoStar ingestion needs the real export files and their column layout.")


def occupancy_factors(building: Building, records: list[dict]) -> dict[str, float]:
    """STUB - ``{room_id: occupancy_factor in [0, 1]}`` from suite records."""
    raise NotImplementedError("Suite -> room matching needs real CoStar suites and Matterport rooms.")


def apply_occupancy(building: Building, factors: dict[str, float]) -> Building:
    """Return a copy of ``building`` with ``Room.occupancy_factor`` overridden (real helper)."""
    out = building.model_copy(deep=True)
    for room in out.rooms:
        if room.id in factors:
            room.occupancy_factor = min(1.0, max(0.0, float(factors[room.id])))
    return out

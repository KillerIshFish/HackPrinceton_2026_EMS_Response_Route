"""Day-0 dataset inventory.

v0: reports which raw-data drop folders exist and what is in them. No network.

    uv run python -m ingest.inventory

TODO (needs the real datasets): ``check_field_coverage`` should open a sample of
each dataset and report, per required field, the fraction of records that have
it - so we know before building ingestion which model inputs are actually
available (room polygons? window annotations? suite-level occupancy?).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

try:  # optional: pick up a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

DATASETS: dict[str, dict] = {
    "matterport": {
        "env": "MATTERPORT_DATA_DIR",
        "default": "./data/raw/matterport",
        "feeds": "Room polygons, floors, doors/stairs/elevators, window openings",
        "required_fields": ["room polygons", "room labels", "floor index", "door openings",
                            "floor elevations", "window openings", "scan pose / georeference"],
    },
    "vexcel": {
        "env": "VEXCEL_DATA_DIR",
        "default": "./data/raw/vexcel",
        "feeds": "Aerial tiles -> what each window faces, entrances, staging areas",
        "required_fields": ["ortho tiles", "oblique tiles", "tile georeference", "building footprint"],
    },
    "costar": {
        "env": "COSTAR_DATA_DIR",
        "default": "./data/raw/costar",
        "feeds": "Building use type, suite occupancy / tenants -> occupancy_factor",
        "required_fields": ["property id / address", "building use type", "suite list",
                            "suite occupancy / vacancy", "tenant type", "operating hours"],
    },
}


def dataset_dir(name: str) -> Path:
    spec = DATASETS[name]
    return Path(os.environ.get(spec["env"], spec["default"]))


def inventory_report() -> dict[str, dict]:
    """Existence + file counts for each raw dataset folder."""
    report: dict[str, dict] = {}
    for name, spec in DATASETS.items():
        path = dataset_dir(name)
        entry = {"env": spec["env"], "path": str(path), "exists": path.is_dir(),
                 "n_files": 0, "extensions": {}, "feeds": spec["feeds"]}
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            entry["n_files"] = len(files)
            entry["extensions"] = dict(Counter(p.suffix.lower() or "<none>" for p in files).most_common(8))
        report[name] = entry
    return report


def check_field_coverage(name: str) -> dict[str, float]:
    """STUB - fraction of sampled records carrying each of ``required_fields``."""
    raise NotImplementedError(
        f"Field-coverage check for '{name}' needs the real dataset layout. "
        f"Required fields to verify: {DATASETS[name]['required_fields']}"
    )


def main() -> None:
    print("Dataset inventory (raw data and API caches under data/raw/)\n")
    for name, entry in inventory_report().items():
        if not entry["exists"]:
            status = "MISSING"
        elif entry["n_files"] == 0:
            status = "EMPTY"
        else:
            status = f"{entry['n_files']} files {entry['extensions']}"
        print(f"  {name:<11} {entry['env']:<20} {entry['path']:<28} {status}")
        print(f"  {'':<11} feeds: {entry['feeds']}")
        print(f"  {'':<11} fields to verify: {', '.join(DATASETS[name]['required_fields'])}\n")


if __name__ == "__main__":
    main()

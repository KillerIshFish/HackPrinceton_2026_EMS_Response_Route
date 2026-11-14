"""Loader for configs/situations.yaml (shared by lanes A, B and the API).

Every numeric setting in that file is a mapping::

    {value: <number>, range: [low, high] | null, label: measured | derived | judgement | unverified,
     source: "<citation or reason>"}

``setting_value`` returns the ``value`` of such an entry (or a bare number as is).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SITUATIONS_PATH = Path(__file__).resolve().parents[1] / "configs" / "situations.yaml"
LABELS = ("measured", "derived", "judgement", "unverified")


@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_situations(path: str | Path | None = None) -> dict:
    """The parsed situations config (cached; do not mutate the result)."""
    return _load_cached(str(path or SITUATIONS_PATH))


def section(name: str, path: str | Path | None = None) -> dict:
    """One top-level section (``incident_types``, ``travel``, ``survival``, ``search``, ``policy``); {} if empty."""
    return load_situations(path).get(name) or {}


def setting_value(entry: Any, default: Any = None) -> Any:
    """``entry["value"]`` for a labelled setting, the entry itself for a bare value, ``default`` for None."""
    if entry is None:
        return default
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry

"""The one network call that the 3D build needs.

Separated from assembly so the whole pipeline downstream of here can be tested
against a saved payload with no credentials and no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ingest.mp3d.client import MatterportClient, MatterportError
from ingest.mp3d.queries import GEOMETRY


def fetch_geometry(
    model_id: str, client: Optional[MatterportClient] = None
) -> dict:
    """Raw GEOMETRY payload for ``model_id`` (median ~225 ms, ~49 KiB)."""
    client = client or MatterportClient()
    data = client.query(GEOMETRY, {"id": model_id})
    model = data.get("model")
    if not model:
        raise MatterportError(f"no model returned for id {model_id!r}")
    return model


def save_payload(payload: dict, path: str | Path) -> Path:
    """Write a payload to disk as a test fixture."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    return target


def load_payload(path: str | Path) -> dict:
    """Read a payload saved by :func:`save_payload`."""
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)

"""Matterport Model API (GraphQL) client with an on-disk cache.

Credentials come from ``.env``: ``MATTERPORT_API_KEY`` is the token id and
``MATTERPORT_API_TOKEN`` the token secret (HTTP basic auth). Every model is
fetched once and stored as ``<MATTERPORT_DATA_DIR>/<model_id>.json``; later
calls read the file and never touch the network unless ``refresh=True``.

    uv run python -m ingest.matterport_api MocRC6tyWqy Gq5yHCcffyL
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

GRAPH_URL = "https://api.matterport.com/api/models/graph"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Everything the building assembler needs from one scan. Lengths are metres.
MODEL_QUERY = """
query Model($id: ID!) {
  model(id: $id) {
    id name description created modified visibility state
    address { address locality administrativeArea postalCode }
    geocoordinates {
      latitude longitude altitude source
      rotation { x y z w }
      translation { x y z }
    }
    publication { published url address }
    floors {
      id label sequence
      dimensions(units: metric) { areaFloor areaFloorIndoor }
      dimensionEstimates(units: metric) { area areaIndoor lowerElevation height }
    }
    rooms {
      id label tags keywords
      floor { id }
      classifications { id label confidence }
      dimensions(units: metric) { areaFloor areaFloorIndoor width depth height }
      ceiling(units: metric) { minHeight maxHeight }
      boundary {
        edges {
          id type
          vertices { id position { x y } }
          openings { id type width height lowerElevation relativeCenter label }
        }
      }
    }
    locations { id label position { x y z } floor { id } room { id } neighbors }
    mattertags { id label description position { x y z } floor { id } }
    objectAnnotations { id label confidence classification { id label } room { id } }
  }
}
"""


class MatterportError(RuntimeError):
    pass


def data_dir() -> Path:
    raw = os.environ.get("MATTERPORT_DATA_DIR", "./data/raw/matterport")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _auth_header() -> str:
    load_dotenv(REPO_ROOT / ".env")
    token_id = (os.environ.get("MATTERPORT_API_KEY") or "").strip()
    secret = (os.environ.get("MATTERPORT_API_TOKEN") or "").strip()
    if not token_id or not secret:
        raise MatterportError("MATTERPORT_API_KEY / MATTERPORT_API_TOKEN are not set in .env")
    return "Basic " + base64.b64encode(f"{token_id}:{secret}".encode()).decode()


def graphql(query: str, variables: dict[str, Any] | None = None, timeout: float = 60.0) -> dict:
    """POST one GraphQL request; return ``data`` or raise ``MatterportError``."""
    req = urllib.request.Request(
        GRAPH_URL,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": _auth_header(),
            "User-Agent": "caller-location-predictor/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise MatterportError(f"Matterport API HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise MatterportError(f"Matterport API unreachable: {exc.reason}") from exc
    if payload.get("errors"):
        raise MatterportError("Matterport API error: " + json.dumps(payload["errors"])[:500])
    return payload.get("data") or {}


def fetch_model(model_id: str, refresh: bool = False) -> dict:
    """One scan as a dict (the ``model`` object of ``MODEL_QUERY``), cached on disk."""
    path = data_dir() / f"{model_id}.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    model = graphql(MODEL_QUERY, {"id": model_id}).get("model")
    if not model:
        raise MatterportError(f"Model '{model_id}' not found or not visible to this API token")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1), encoding="utf-8")
    return model


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        m = fetch_model(mid)
        print(f"{mid}  {m.get('name')!r}  rooms={len(m.get('rooms') or [])}  sweeps={len(m.get('locations') or [])}")

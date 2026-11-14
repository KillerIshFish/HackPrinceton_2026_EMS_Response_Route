"""Matterport geometry -> 3D model pipeline.

Deterministic: the same ``model_id`` yields byte-identical output on every run.
Everything between :mod:`ingest.mp3d.fetch` and :mod:`ingest.mp3d.model3d` is a
pure function over the fetched payload, so it is testable without a network.

    from ingest.mp3d import preflight, build_model

    cap = preflight("zhKWys61pww")      # ~160 ms, 2.4 KiB
    if cap.buildable:
        model = build_model("zhKWys61pww")   # ~225 ms
        model.to_json_file("out.json")

Only part of the corpus carries the vector structural layer this needs
(measured: 62% of multi-floor models, 39% overall; commercial models only
13%), so always check ``preflight`` first rather than assuming a model will
build.

The structural layer is all-or-nothing. Across 70 models, room outlines, wall
thickness, dimensioned openings, ``Edge.type`` and room classifications were
present in *exactly* the same 37 models, and the remaining 33 had zero boundary
edges across 147 rooms. For those, :func:`build_scan_model` offers a weaker
model built from scan positions and the anchor graph - positions, areas and
measured connectivity, but no walls and no window views.
"""

from ingest.mp3d.client import MatterportClient, MatterportError
from ingest.mp3d.clusters import ScanModel, ScanRoom, build_scan_model
from ingest.mp3d.layers import LayerReport, layer_report
from ingest.mp3d.model3d import Model3D, Opening, RoomPlate, Slab, Wall, build_model
from ingest.mp3d.preflight import Capability, preflight
from ingest.mp3d.views import RoomView, WindowView, room_views, view_summary

__all__ = [
    "MatterportClient",
    "MatterportError",
    # full structural build
    "Capability",
    "preflight",
    "build_model",
    "Model3D",
    "Slab",
    "Wall",
    "Opening",
    "RoomPlate",
    # per-layer probe
    "LayerReport",
    "layer_report",
    # window views - the data that tells rooms apart
    "RoomView",
    "WindowView",
    "room_views",
    "view_summary",
    # scan-cluster fallback, for models with no wall geometry
    "ScanModel",
    "ScanRoom",
    "build_scan_model",
]

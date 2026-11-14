"""Per-layer capability probe.

:mod:`ingest.mp3d.preflight` answers one question - can the full structural
build run? - from one cheap scalar. This module answers the finer question:
*which* layers a model carries, so a reduced build can be offered to models
that lack wall thickness but still have usable room outlines.

The layers, in the order a build depends on them:

``outlines``    room boundary rings close into simple polygons
``positions``   pano scan positions with z (floor elevation fallback)
``sills``       doorway ``lowerElevation`` (the accurate floor elevation source)
``thickness``   ``Edge.thickness`` (measured wall thickness)
``openings``    openings carrying all four dimension fields
``edge_types``  ``Edge.type``, which marks open boundaries
``classes``     room classifications

A model's :class:`LayerReport` maps to a build level:

``full``      outlines + thickness + openings + sills  (the existing build)
``reduced``   outlines, but missing thickness or openings - walls get default
              thicknesses and no holes; elevations may come from pano z
``none``      no usable outlines, so there is no geometry to build
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ingest.mp3d import geometry as geo
from ingest.mp3d.client import MatterportClient

LEVEL_FULL = "full"
LEVEL_REDUCED = "reduced"
LEVEL_NONE = "none"

# Heavier than PREFLIGHT because deciding whether outlines exist needs the
# vertices themselves. Used for corpus surveys and for choosing a build level,
# not on a hot path where PREFLIGHT's single scalar is enough.
LAYERS_QUERY = """
query Layers($id: ID!) {
  model(id: $id) {
    id
    name
    state
    propertyClassifications { classification type }
    floors {
      sequence
      panoLocations { position { z } }
      rooms {
        id
        classifications { label }
        boundary {
          edges {
            type
            thickness
            vertices { id position { x y } }
            openings { type width height lowerElevation relativeCenter }
          }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class LayerReport:
    """Which data layers a model carries, and what can be built from them."""

    model_id: str
    name: str = ""
    state: str = ""
    property_class: str = ""
    floors: int = 0
    rooms: int = 0

    # layer presence
    rooms_with_outline: int = 0
    floors_with_outline: int = 0
    floors_with_positions: int = 0
    floors_with_sills: int = 0
    walls_total: int = 0
    walls_with_thickness: int = 0
    walls_with_edge_type: int = 0
    openings_total: int = 0
    openings_dimensioned: int = 0
    rooms_with_class: int = 0

    level: str = LEVEL_NONE
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def outline_coverage(self) -> float:
        return self.rooms_with_outline / self.rooms if self.rooms else 0.0

    def describe(self) -> str:
        head = (
            f"{self.model_id} {self.name[:34]!r}: build level {self.level}"
            f" | {self.floors} floors, {self.rooms_with_outline}/{self.rooms} outlines"
        )
        layers = (
            f"  outlines={self.rooms_with_outline} positions={self.floors_with_positions}"
            f"/{self.floors} sills={self.floors_with_sills}/{self.floors}"
            f" thickness={self.walls_with_thickness}/{self.walls_total}"
            f" openings={self.openings_dimensioned}/{self.openings_total}"
            f" classes={self.rooms_with_class}/{self.rooms}"
        )
        out = [head, layers]
        if self.missing:
            out.append("  missing: " + ", ".join(self.missing))
        return "\n".join(out)


def report_from_payload(model_id: str, model: Optional[dict]) -> LayerReport:
    """Classify a model's layers from a LAYERS payload. Pure; no network."""
    if not model:
        return LayerReport(model_id=model_id, missing=("model not found",))

    floor_list = model.get("floors") or []
    n_floors = len(floor_list)
    prop = ""
    for entry in model.get("propertyClassifications") or []:
        if entry.get("classification"):
            prop = str(entry["classification"])
            break

    rooms = outlines = with_class = 0
    floors_outline = floors_pos = floors_sill = 0
    walls = thick = typed = 0
    ops = ops_dim = 0

    for floor in floor_list:
        any_outline = False
        any_sill = False
        for room in floor.get("rooms") or []:
            rooms += 1
            if room.get("classifications"):
                with_class += 1
            edges = (room.get("boundary") or {}).get("edges") or []
            for edge in edges:
                walls += 1
                if edge.get("thickness") is not None:
                    thick += 1
                if edge.get("type") is not None:
                    typed += 1
                for opening in edge.get("openings") or []:
                    ops += 1
                    if all(
                        opening.get(k) is not None
                        for k in ("relativeCenter", "width", "lowerElevation", "height")
                    ):
                        ops_dim += 1
                    if (
                        opening.get("type") == "doorway"
                        and opening.get("lowerElevation") is not None
                    ):
                        any_sill = True
            if geo.ring_from_edges(edges) is not None:
                outlines += 1
                any_outline = True
        if any_outline:
            floors_outline += 1
        if any_sill:
            floors_sill += 1
        zs = [
            (p.get("position") or {}).get("z")
            for p in (floor.get("panoLocations") or [])
        ]
        if [z for z in zs if z is not None]:
            floors_pos += 1

    missing: list[str] = []
    if not outlines:
        missing.append("outlines")
    if not thick:
        missing.append("thickness")
    if not ops_dim:
        missing.append("openings")
    if n_floors and floors_sill < n_floors:
        missing.append("sills")
    if n_floors and floors_pos < n_floors:
        missing.append("positions")

    if not outlines or not n_floors:
        level = LEVEL_NONE
    elif thick and ops_dim and floors_sill == n_floors:
        level = LEVEL_FULL
    elif floors_sill == n_floors or floors_pos == n_floors:
        level = LEVEL_REDUCED
    else:
        level = LEVEL_NONE  # outlines but no way to place the floors in z

    return LayerReport(
        model_id=model_id,
        name=str(model.get("name") or ""),
        state=str(model.get("state") or ""),
        property_class=prop,
        floors=n_floors,
        rooms=rooms,
        rooms_with_outline=outlines,
        floors_with_outline=floors_outline,
        floors_with_positions=floors_pos,
        floors_with_sills=floors_sill,
        walls_total=walls,
        walls_with_thickness=thick,
        walls_with_edge_type=typed,
        openings_total=ops,
        openings_dimensioned=ops_dim,
        rooms_with_class=with_class,
        level=level,
        missing=tuple(missing),
    )


def layer_report(
    model_id: str, client: Optional[MatterportClient] = None
) -> LayerReport:
    """Probe every layer of one model (~225 ms, ~49 KiB)."""
    client = client or MatterportClient()
    data = client.query(LAYERS_QUERY, {"id": model_id})
    return report_from_payload(model_id, data.get("model"))

"""GraphQL documents used by the pipeline.

Kept apart from the logic so the exact field set is reviewable in one place,
and so a schema change shows up as a diff here rather than buried in code.
"""

from __future__ import annotations

# Cheapest document that can classify a model. Edge.thickness is present exactly
# when the full structural layer is (measured: the two sets were identical across
# 70 models), so one scalar per wall is enough to decide buildability.
PREFLIGHT = """
query Preflight($id: ID!) {
  model(id: $id) {
    id
    name
    state
    floors {
      sequence
      rooms {
        id
        boundary { edges { thickness } }
      }
    }
  }
}
"""

# Everything the 3D build needs, in one round trip.
GEOMETRY = """
query Geometry($id: ID!) {
  model(id: $id) {
    id
    name
    state
    address { address locality administrativeArea postalCode }
    geolocation { lat long }
    geocoordinates { rotation { w x y z } }
    floors {
      sequence
      label
      panoLocations { position { z } }
      rooms {
        id
        classifications { label confidence }
        boundary {
          edges {
            id
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

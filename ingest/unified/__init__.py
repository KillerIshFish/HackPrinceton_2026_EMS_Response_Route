"""Unified building pipeline: an address, a property id or a Matterport model id -> ``BuildingModel``.

Two sources meet here:

* ``ingest.mp3d`` (Track A) turns one Matterport scan into a surveyed ``Model3D``
  or, when the scan has no structural layer, a coarse ``ScanModel``.
* The property generator (Track B, this package) builds a whole building from a
  property record: floor stack, unit / tenant roster, an archetype layout, and
  unit interiors stamped from scans of the unit type where one exists.

Both produce a :class:`ingest.unified.schema.BuildingModel`. Everything the rest
of the app consumes (``core.schema.Building`` for the posterior and the search
plan, the 3D viewer payload) is a projection of that model.

    uv run python -m ingest.unified "11900 W Olympic Blvd"     # address
    uv run python -m ingest.unified 246948                      # property id
    uv run python -m ingest.unified mp:1FTnLNf1sBz              # one whole-building scan

Modules: ``index`` (address lookup), ``record`` (property facts), ``roster``
(units and tenants on floors), ``layout`` (archetype geometry), ``stamp`` (scanned
unit interiors), ``lift`` (Model3D / ScanModel / Building -> BuildingModel),
``project`` (BuildingModel -> Building, viewer payload), ``context`` (aerial and
street-bearing sidecar), ``pipeline`` (the stages in order).
"""

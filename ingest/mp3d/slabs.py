"""Floor slab elevations.

Matterport's ``floors.dimensionEstimates.lowerElevation`` is null on every model
we have seen, so the floor plane has to be derived. Two sources, in order:

1. **Median doorway ``lowerElevation``** - a door sill physically rests on the
   slab, and the per-floor values cluster tightly (n=15/6/20 within 0.24 m on
   the reference model). This is the accurate source.
2. **Minimum pano z minus the tripod height** - a fallback. Pano z is *camera*
   height, about 1.5 m above the floor, so using it directly understates
   floor-to-floor spacing badly (2.80 m vs the true 3.50 m on the reference
   model, which materially changed downstream floor probabilities).

All elevations are metres on one datum, shifted so the lowest floor sits at 0.
Opening elevations from the API share that datum before the shift, so the same
``datum`` offset must be applied to them - see :func:`shift_to_datum`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

PANO_ABOVE_FLOOR_M = 1.5  # Matterport tripod height
DEFAULT_FLOOR_HEIGHT_M = 3.0
MIN_STACK_GAP_M = 0.3     # below this, two "floors" are the same plane
MIN_PLAUSIBLE_STOREY_M = 2.0  # a real storey is not shorter than this
ELEV_DP = 3

SOURCE_DOORWAY = "doorway-median"
SOURCE_PANO = "pano-z-fallback"
SOURCE_ASSUMED = "assumed"


@dataclass(frozen=True)
class FloorElevation:
    floor: int
    label: str
    raw_z: float
    source: str
    sample_size: int


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def floor_elevations(floors: Sequence[dict]) -> list[FloorElevation]:
    """Raw (unshifted) slab elevation per floor, in API order of ``sequence``.

    Pure: takes the ``floors`` list straight off a GEOMETRY payload.
    """
    out: list[FloorElevation] = []
    for floor in sorted(floors, key=lambda f: f.get("sequence") or 0):
        sequence = int(floor.get("sequence") or 0)
        label = str(floor.get("label") or f"Floor {sequence + 1}")

        doorways: list[float] = []
        for room in floor.get("rooms") or []:
            for edge in ((room.get("boundary") or {}).get("edges") or []):
                for opening in edge.get("openings") or []:
                    if (
                        opening.get("type") == "doorway"
                        and opening.get("lowerElevation") is not None
                    ):
                        doorways.append(float(opening["lowerElevation"]))
        if doorways:
            out.append(
                FloorElevation(sequence, label, _median(doorways),
                               SOURCE_DOORWAY, len(doorways))
            )
            continue

        panos = [
            float((p.get("position") or {}).get("z"))
            for p in (floor.get("panoLocations") or [])
            if (p.get("position") or {}).get("z") is not None
        ]
        if panos:
            out.append(
                FloorElevation(sequence, label, min(panos) - PANO_ABOVE_FLOOR_M,
                               SOURCE_PANO, len(panos))
            )
            continue

        out.append(FloorElevation(sequence, label, 0.0, SOURCE_ASSUMED, 0))
    return out


def datum(elevations: Sequence[FloorElevation]) -> float:
    """The raw elevation treated as z = 0 (the lowest floor's slab)."""
    if not elevations:
        return 0.0
    return min(e.raw_z for e in elevations)


def shift_to_datum(raw_z: float, datum_z: float) -> float:
    """Convert a raw API elevation onto the pipeline's z = 0 datum."""
    return round(raw_z - datum_z, ELEV_DP)


def floor_heights(elevations: Sequence[FloorElevation]) -> dict[int, float]:
    """Slab-to-slab height per floor, metres.

    Each floor takes the measured gap to the slab above, so the stack has no
    overlaps and no voids. The topmost floor has nothing above it to measure,
    so it takes the mean of the *plausible* spacings below - spacings under
    :data:`MIN_PLAUSIBLE_STOREY_M` are excluded, because a sub-2 m gap means
    Matterport split one storey into two floors (a split-level, a mezzanine, or
    a segmentation artifact) and averaging it in would give the top floor an
    impossible height. With no plausible spacing at all it falls back to 3.0 m.
    """
    if not elevations:
        return {}
    ordered = sorted(elevations, key=lambda e: e.floor)
    base = datum(ordered)
    zs = [shift_to_datum(e.raw_z, base) for e in ordered]
    gaps = [zs[i + 1] - zs[i] for i in range(len(zs) - 1)]
    plausible = [g for g in gaps if g >= MIN_PLAUSIBLE_STOREY_M]
    mean = (
        round(sum(plausible) / len(plausible), ELEV_DP)
        if plausible
        else DEFAULT_FLOOR_HEIGHT_M
    )
    heights: dict[int, float] = {}
    for i, elevation in enumerate(ordered):
        gap = gaps[i] if i < len(gaps) else None
        if gap is not None and gap > MIN_STACK_GAP_M:
            heights[elevation.floor] = round(gap, ELEV_DP)
        else:
            heights[elevation.floor] = mean
    return heights


def implausible_gaps(
    elevations: Sequence[FloorElevation],
) -> list[tuple[int, int, float]]:
    """``(lower_floor, upper_floor, gap_m)`` for each sub-storey floor spacing.

    A non-empty result means the model's floor segmentation does not correspond
    to storeys, so the stacked geometry is honest about the measured planes but
    should not be read as "one floor per storey".
    """
    ordered = sorted(elevations, key=lambda e: e.floor)
    base = datum(ordered)
    zs = [shift_to_datum(e.raw_z, base) for e in ordered]
    out: list[tuple[int, int, float]] = []
    for i in range(len(zs) - 1):
        gap = zs[i + 1] - zs[i]
        if gap < MIN_PLAUSIBLE_STOREY_M:
            out.append((ordered[i].floor, ordered[i + 1].floor, round(gap, ELEV_DP)))
    return out

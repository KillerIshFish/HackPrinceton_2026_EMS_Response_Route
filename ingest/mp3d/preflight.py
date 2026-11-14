"""Cheap buildability check.

Only part of the corpus carries the vector structural layer the 3D build needs.
Measured across a stratified sample of 70 models:

* 62% of multi-floor models and 39% overall are buildable
* ``Edge.thickness`` is present exactly when dimensioned openings are - the two
  sets were *identical*, so one scalar per wall decides it
* failures are whole-model, never partial: every unbuildable model had **zero**
  closed rings rather than a damaged subset

That makes buildability a single boolean rather than a confidence score, which
is why this check is reliable enough to gate the pipeline on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ingest.mp3d.client import MatterportClient
from ingest.mp3d.queries import PREFLIGHT

TIER_FULL = "A"       # rings + per-floor doorway elevations + dimensioned openings
TIER_DEGRADED = "B"   # rings, but elevations fall back to pano z
TIER_UNBUILDABLE = "C"


@dataclass(frozen=True)
class Capability:
    """What the pipeline will be able to make of a model."""

    model_id: str
    name: str = ""
    state: str = ""
    buildable: bool = False
    tier: str = TIER_UNBUILDABLE
    floors: int = 0
    rooms: int = 0
    walls_with_thickness: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        verdict = f"tier {self.tier}" if self.buildable else "not buildable"
        head = (
            f"{self.model_id} {self.name!r}: {verdict} "
            f"({self.floors} floors, {self.rooms} rooms)"
        )
        if self.reasons:
            head += "\n  " + "\n  ".join(self.reasons)
        return head


def capability_from_payload(model_id: str, model: Optional[dict]) -> Capability:
    """Classify a model from a PREFLIGHT payload. Pure; no network."""
    if not model:
        return Capability(
            model_id=model_id,
            reasons=("model not found, or not visible to these credentials",),
        )

    floors = model.get("floors") or []
    rooms = 0
    thick = 0
    for floor in floors:
        for room in floor.get("rooms") or []:
            rooms += 1
            for edge in ((room.get("boundary") or {}).get("edges") or []):
                if edge.get("thickness") is not None:
                    thick += 1

    reasons: list[str] = []
    state = model.get("state") or ""
    if state and state != "active":
        reasons.append(f"model state is {state!r}, not 'active'")
    if not floors:
        reasons.append("no floors returned")
    if not rooms:
        reasons.append("no rooms returned")
    if not thick:
        reasons.append(
            "no Edge.thickness on any wall - this model predates the structural "
            "annotation layer, so it carries no room rings or dimensioned openings"
        )

    buildable = bool(floors and rooms and thick and state in ("", "active"))
    return Capability(
        model_id=model_id,
        name=model.get("name") or "",
        state=state,
        buildable=buildable,
        tier=TIER_FULL if buildable else TIER_UNBUILDABLE,
        floors=len(floors),
        rooms=rooms,
        walls_with_thickness=thick,
        reasons=tuple(reasons),
    )


def preflight(
    model_id: str, client: Optional[MatterportClient] = None
) -> Capability:
    """One small query (~160 ms, ~2.4 KiB) deciding whether a model will build.

    The returned ``tier`` is provisional: it distinguishes buildable from not.
    :func:`ingest.mp3d.model3d.build_model` downgrades to tier B if a floor
    turns out to need the pano-z elevation fallback.
    """
    client = client or MatterportClient()
    data = client.query(PREFLIGHT, {"id": model_id})
    return capability_from_payload(model_id, data.get("model"))

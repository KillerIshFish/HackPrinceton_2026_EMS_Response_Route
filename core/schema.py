"""FROZEN CONTRACT - shared data models for every module.

Do not rename or remove fields without telling the whole team: ingest/, core/,
parser/, planning/, api/ and web/ all depend on these exact shapes.

Conventions
-----------
* Coordinates are building-local metres: x = east, y = north, z = up. z = 0 is
  the finished floor of the lowest floor in ``Building.floors``.
* ``Room.floor`` is the integer floor index; ``Room.floor_label`` is what a
  human says ("1", "2", "M", "B1").
* ``Building.floor_heights_m[floor]`` is the finished-floor elevation of that
  floor above the z = 0 datum.
* Probabilities: ``sum(Posterior.room_probs.values()) + p_not_in_model == 1``.

Free-form dict shapes (kept as dicts in the frozen contract, keys by convention)
--------------------------------------------------------------------------------
Building.edges[i]::

    {"u": room_id, "v": room_id,
     "kind": "door" | "corridor" | "stairs" | "elevator",
     "length_m": float,          # walking distance between the two centroids
     "floors": int}              # abs floor delta (0 unless stairs/elevator)

    Optional keys (read by core/route_clues.py; absent = fall back to room outlines):
     "xy": [x, y]                          # door / corridor edge: where the opening is
     "top_xy": [x, y], "heading": [dx, dy]            # stairs: arrival point and travel direction going up
     "bottom_xy": [x, y], "bottom_heading": [dx, dy]  # stairs: the same going down

Building.entrances[i]::

    {"id": str, "name": str, "room_id": str,     # room the door opens into
     "xy": [x, y], "latlng": [lat, lng],
     "facing": "street" | "parking_lot"}

Building.staging_spots[i]::

    {"id": str, "name": str, "xy": [x, y], "latlng": [lat, lng],
     "kind": "street" | "parking_lot", "near_entrance_id": str}
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# "route" was added after the freeze (additive): how the caller reached the room, e.g.
# "turned right after coming up the stairs". Values are keys of `routes` in clue_vocab.yaml.
ClueType = Literal["room_type", "floor", "landmark", "view", "sound", "smell", "negative", "route"]


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    floor: int
    floor_label: str
    polygon: list[tuple[float, float]]
    centroid: tuple[float, float, float]
    area_m2: float
    room_type: str
    has_windows: bool
    window_facing: Optional[str] = None
    near_elevator: bool
    near_stairs: bool
    occupancy_factor: float


class Building(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    use_type: str
    floors: list[int]
    floor_heights_m: dict[int, float]
    rooms: list[Room]
    edges: list[dict]
    entrances: list[dict]
    staging_spots: list[dict]


class Clue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ClueType
    value: str
    polarity: bool
    confidence: float = Field(ge=0.0, le=1.0)
    span: str


class Posterior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_probs: dict[str, float]
    p_not_in_model: float
    p_floor: dict[str, float]
    credible_set_90: list[str]
    entropy_bits: float
    peak_count: int


class Stop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str
    room_name: str
    floor_label: str
    eta_s: float
    cumulative_prob: float


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teams: list[list[Stop]]
    expected_time_to_find_s: float
    baseline_nearest_s: float
    baseline_highest_prob_s: float
    entrance_id: str
    staging_latlng: tuple[float, float]


# ---------------------------------------------------------------------------
# Supplementary model (not one of the six frozen contract models above, but
# shared by sim/, core/phone_likelihood.py and api/).
# ---------------------------------------------------------------------------

Z_SIGMA_M = 3.0 / 1.28  # +/-3 m is the 80% z-axis bound -> sigma ~= 2.34 m


class PhoneFix(BaseModel):
    """Device location fix in building-local metres. Any component may be missing."""

    model_config = ConfigDict(extra="forbid")

    x_m: Optional[float] = None
    y_m: Optional[float] = None
    horizontal_sigma_m: Optional[float] = None
    z_m: Optional[float] = None  # barometric height above the z = 0 datum
    z_sigma_m: float = Z_SIGMA_M

"""Window views: the data that tells one room from another.

A caller rarely knows which room they are in, but they can describe what they
see. "A small window up high" is a bathroom; "I can see the parking lot" picks
a facade. This module turns each room's openings into the kind of description a
caller would give, and then measures how far that description narrows things
down.

Everything here is derived from three measured fields per window -
``relativeCenter``, ``width``, ``lowerElevation``/``height`` - plus the model's
yaw quaternion, which turns a wall normal into a true compass bearing. The one
unmeasured step is naming what a bearing *faces*: that needs a reference
bearing for the street, which :func:`classify_facing` takes as an explicit
argument rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ingest.mp3d import geometry as geo
from ingest.mp3d.model3d import Model3D, RoomPlate, Wall

# A caller says "small window", not "0.61 m". These thresholds come from the
# glazing sizes actually present in the corpus.
SMALL_MAX_M2 = 0.8
LARGE_MIN_M2 = 2.2

# Sill height above finished floor. A high sill is the bathroom/utility tell;
# a floor-level sill is a patio door or full-height glazing. A sill measurably
# below the slab is not glazing at all - it is an opening whose scan elevation
# puts it on the wrong floor, and saying "floor-level" about it would be a lie.
# Tuned heuristic values: -0.1 m leaves 10 cm for sill/median-slab scan noise before a sill counts as below-floor;
# <= 0.4 m reads as floor-level glazing, >= 1.4 m sits above a standing adult's chest (the bathroom tell).
SILL_BELOW_FLOOR_M = -0.1
SILL_LOW_MAX_M = 0.4
SILL_HIGH_MIN_M = 1.4

SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
FACING_STREET = "street"
FACING_LOT = "parking_lot"
FACING_UNKNOWN = "unknown"


def sector(bearing: float) -> str:
    """Eight-point compass sector containing ``bearing``."""
    index = int((bearing % 360.0 + 22.5) // 45.0) % 8  # 45-degree sectors centred on N, NE, ...
    return SECTORS[index]


def size_class(area_m2: float) -> str:
    if area_m2 <= SMALL_MAX_M2:
        return "small"
    if area_m2 >= LARGE_MIN_M2:
        return "large"
    return "medium"


def sill_class(sill_aff_m: float) -> str:
    if sill_aff_m < SILL_BELOW_FLOOR_M:
        return "below-floor"
    if sill_aff_m <= SILL_LOW_MAX_M:
        return "floor-level"
    if sill_aff_m >= SILL_HIGH_MIN_M:
        return "high"
    return "standard"


@dataclass(frozen=True)
class WindowView:
    """One window, described the way a caller would experience it."""

    wall_id: str
    bearing_deg: float
    sector: str
    width_m: float
    height_m: float
    area_m2: float
    sill_aff_m: float
    size: str
    sill: str

    def describe(self) -> str:
        return (
            f"{self.size} {self.sill} window facing {self.sector} "
            f"({self.bearing_deg:.0f}°, {self.area_m2:.2f} m², "
            f"sill {self.sill_aff_m:.2f} m)"
        )


@dataclass(frozen=True)
class RoomView:
    """A room's whole window exposure."""

    room_id: str
    floor: int
    classification: str
    windows: tuple[WindowView, ...] = field(default_factory=tuple)

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def has_windows(self) -> bool:
        return bool(self.windows)

    @property
    def glazed_area_m2(self) -> float:
        return round(sum(w.area_m2 for w in self.windows), 3)

    @property
    def suspect_windows(self) -> tuple["WindowView", ...]:
        """Windows whose sill sits below the floor - mis-elevated in the scan.

        A caller cannot see through these, so they should not drive a view
        clue. They are kept rather than dropped so the count stays visible.
        """
        return tuple(w for w in self.windows if w.sill == "below-floor")

    @property
    def sectors(self) -> tuple[str, ...]:
        return tuple(sorted({w.sector for w in self.windows}, key=SECTORS.index))

    @property
    def dominant_bearing(self) -> Optional[float]:
        """Width-weighted circular mean of the window bearings."""
        if not self.windows:
            return None
        spread: list[float] = []
        for window in self.windows:
            # weight by width, in whole decimetres, so wider glazing counts more
            for _ in range(max(1, int(round(window.width_m * 10)))):
                spread.append(window.bearing_deg)
        mean = geo.circular_mean(spread)
        return None if mean is None else round(mean, 1)

    @property
    def corner_room(self) -> bool:
        """Windows on more than one facade - a strong discriminator."""
        if len(self.sectors) < 2:
            return False
        return any(
            # More than one 45-degree sector apart, so two adjacent-sector windows on one bent facade do not count.
            geo.angular_gap(a.bearing_deg, b.bearing_deg) > 45.0
            for a in self.windows
            for b in self.windows
        )

    def signature(self) -> tuple:
        """What a caller could plausibly convey, as a comparable key.

        Deliberately coarse: a caller reports roughly how many windows, how big,
        how high the sill, and which way they face - never exact metres.
        """
        return (
            self.classification,
            min(self.window_count, 3),  # a caller says "one", "two" or "lots", so 3+ windows share a bucket
            tuple(sorted({w.size for w in self.windows})),
            tuple(sorted({w.sill for w in self.windows})),
            self.sectors,
        )

    def describe(self) -> str:
        if not self.windows:
            return f"{self.classification} (floor {self.floor}): no windows"
        bits = ", ".join(
            f"{w.size} {w.sill} -> {w.sector}" for w in self.windows[:4]
        )
        extra = "" if self.window_count <= 4 else f" +{self.window_count - 4} more"
        corner = " [corner]" if self.corner_room else ""
        return (
            f"{self.classification} (floor {self.floor}): {self.window_count} window(s), "
            f"{self.glazed_area_m2} m²{corner} - {bits}{extra}"
        )


def _wall_bearing(wall: Wall, room: RoomPlate, yaw_deg: float) -> float:
    return geo.outward_bearing(list(room.polygon), wall.a, wall.b, yaw_deg)


def room_views(model: Model3D) -> list[RoomView]:
    """Build a :class:`RoomView` for every room plate in ``model``.

    A window belongs to the room whose boundary carries it. ``Wall.room_ids``
    already records that, so no spatial guessing is needed.
    """
    rooms = {room.id: room for room in model.rooms}
    slabs = {slab.floor: slab for slab in model.slabs}
    collected: dict[str, list[WindowView]] = {rid: [] for rid in rooms}

    for wall in model.walls:
        windows = [o for o in wall.openings if o.kind == "window"]
        if not windows:
            continue
        slab = slabs.get(wall.floor)
        for room_id in wall.room_ids:
            room = rooms.get(room_id)
            if room is None:
                continue
            bearing = _wall_bearing(wall, room, model.yaw_deg)
            for opening in windows:
                width = round((opening.t1 - opening.t0) * wall.length_m, 3)
                height = round(opening.z1 - opening.z0, 3)
                if width <= 0 or height <= 0:
                    continue
                # product of the rounded values at full precision, so that a
                # reader multiplying the published width by the published
                # height gets exactly the published area
                area = round(width * height, 6)
                sill_aff = round(opening.z0 - (slab.z0 if slab else 0.0), 3)
                collected[room_id].append(
                    WindowView(
                        wall_id=wall.id,
                        bearing_deg=round(bearing, 1),
                        sector=sector(bearing),
                        width_m=width,
                        height_m=height,
                        area_m2=area,
                        sill_aff_m=sill_aff,
                        size=size_class(area),
                        sill=sill_class(sill_aff),
                    )
                )

    out: list[RoomView] = []
    for room_id, room in sorted(rooms.items()):
        windows = sorted(
            collected[room_id],
            key=lambda w: (w.bearing_deg, w.wall_id, -w.area_m2),
        )
        out.append(
            RoomView(
                room_id=room_id,
                floor=room.floor,
                classification=room.classification,
                windows=tuple(windows),
            )
        )
    out.sort(key=lambda v: (v.floor, v.room_id))
    return out


def street_bearing_from_entrances(model: Model3D) -> Optional[float]:
    """Guess which way the street lies, from the exterior walls of an Entrance.

    This is a heuristic and the weakest link in facing classification: it
    assumes the front door faces the street. Aerial imagery or a road
    centreline would replace it. Returns ``None`` when no Entrance room exists,
    which is the common case - only 2 of 22 models carried one.
    """
    entrances = [r for r in model.rooms if r.classification == "Entrance"]
    if not entrances:
        return None
    biggest = max(entrances, key=lambda r: r.area_m2)
    bearings = [
        _wall_bearing(wall, biggest, model.yaw_deg)
        for wall in model.walls
        if wall.exterior and biggest.id in wall.room_ids
    ]
    return None if not bearings else round(geo.circular_mean(bearings) or 0.0, 1)


def classify_facing(
    view: RoomView, street_bearing: Optional[float], tolerance_deg: float = 90.0
) -> str:
    """Name what a room's windows face, given a reference street bearing.

    The default ``tolerance_deg`` of 90 splits the compass into front and back halves, so every bearing except
    one exactly side-on to the street resolves to one of the two.

    Returns ``"street"``, ``"parking_lot"`` or ``"unknown"``. The split is a
    front/back dichotomy: it says which side of the building a room is on, not
    literally what is outside the glass.
    """
    bearing = view.dominant_bearing
    if bearing is None or street_bearing is None:
        return FACING_UNKNOWN
    if geo.angular_gap(bearing, street_bearing) < tolerance_deg:
        return FACING_STREET
    if geo.angular_gap(bearing, (street_bearing + 180.0) % 360.0) < tolerance_deg:
        return FACING_LOT
    return FACING_UNKNOWN


def distinctiveness(views: Sequence[RoomView]) -> dict[str, int]:
    """How many rooms share each room's caller-describable signature.

    ``1`` means a caller describing that room's view has identified it
    uniquely. Higher numbers are rooms a view clue cannot separate. This is the
    measure of how much the window data is actually worth for locating someone.
    """
    counts: dict[tuple, int] = {}
    for view in views:
        key = view.signature()
        counts[key] = counts.get(key, 0) + 1
    return {view.room_id: counts[view.signature()] for view in views}


def view_summary(model: Model3D, street_bearing: Optional[float] = None) -> dict:
    """Corpus-style summary of a model's window data and its discriminating power."""
    views = room_views(model)
    if street_bearing is None:
        street_bearing = street_bearing_from_entrances(model)
    shared = distinctiveness(views)
    windowed = [v for v in views if v.has_windows]
    unique = [v for v in views if shared[v.room_id] == 1]
    facings: dict[str, int] = {}
    for view in views:
        name = classify_facing(view, street_bearing)
        facings[name] = facings.get(name, 0) + 1
    sector_counts: dict[str, int] = {}
    for view in windowed:
        for name in view.sectors:
            sector_counts[name] = sector_counts.get(name, 0) + 1
    return {
        "rooms": len(views),
        "rooms_with_windows": len(windowed),
        "windows": sum(v.window_count for v in views),
        "glazed_area_m2": round(sum(v.glazed_area_m2 for v in views), 2),
        "corner_rooms": sum(1 for v in windowed if v.corner_room),
        "street_bearing_deg": street_bearing,
        "facing": dict(sorted(facings.items())),
        "sectors": dict(sorted(sector_counts.items(), key=lambda kv: SECTORS.index(kv[0]))),
        "uniquely_identified_rooms": len(unique),
        "uniquely_identified_share": round(len(unique) / len(views), 3) if views else 0.0,
        "size_classes": _tally(w.size for v in views for w in v.windows),
        "sill_classes": _tally(w.sill for v in views for w in v.windows),
    }


def _tally(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def views_payload(
    model: Model3D, street_bearing: Optional[float] = None
) -> dict[str, dict]:
    """Per-room view data, keyed by room id, for a renderer or the EMS model."""
    views = room_views(model)
    if street_bearing is None:
        street_bearing = street_bearing_from_entrances(model)
    shared = distinctiveness(views)
    return {
        v.room_id: {
            "n": v.window_count,
            "glazed_m2": v.glazed_area_m2,
            "sectors": list(v.sectors),
            "bearing": v.dominant_bearing,
            "facing": classify_facing(v, street_bearing),
            "corner": v.corner_room,
            "shared_with": shared[v.room_id],
            "windows": [
                {
                    "b": w.bearing_deg, "sec": w.sector, "w": w.width_m,
                    "h": w.height_m, "sill": w.sill_aff_m,
                    "size": w.size, "sillClass": w.sill,
                }
                for w in v.windows
            ],
        }
        for v in views
    }

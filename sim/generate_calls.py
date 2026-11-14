"""Scripted demo calls.

Each call is a list of turns::

    {"turn": int, "title": str, "speaker": "system" | "caller" | "dispatcher",
     "text": str, "phone_fix": PhoneFix | None}

``phone_fix`` is set on the turn where a (new) device fix arrives; consumers keep
the most recent non-null fix. Caller text is run through
``parser.extract.extract_clues`` by the API - no clues are hard-coded here.
"""

from __future__ import annotations

from core.schema import PhoneFix

DEMO_BUILDING_ID = "ptc-100"
DEMO_TRUE_ROOM_ID = "f4-restroom-north"  # where the scripted demo caller actually is


def _turns(rows: list[tuple[str, str, str, PhoneFix | None]]) -> list[dict]:
    return [
        {"turn": i, "title": title, "speaker": speaker, "text": text, "phone_fix": fix}
        for i, (title, speaker, text, fix) in enumerate(rows)
    ]


def demo_call_script() -> list[dict]:
    """The 4-turn headline demo: prior -> z-axis fix -> "bathroom" -> "parking lot"."""
    return _turns(
        [
            (
                "Call connected",
                "system",
                "911 call connected. Dispatchable address: 100 Princeton Tech Center. "
                "No device location yet - showing the building prior.",
                None,
            ),
            (
                "Phone z-axis fix",
                "system",
                "Device barometric z-axis fix received: 12.6 m above ground (±3 m). "
                "No usable horizontal fix indoors.",
                PhoneFix(z_m=12.6),
            ),
            (
                "Caller: room type",
                "caller",
                "I'm locked in a bathroom",
                None,
            ),
            (
                "Caller: view",
                "caller",
                "I can see the parking lot out a small window",
                None,
            ),
        ]
    )


BUNDY_BUILDING_ID = "bundy-olympic"
BUNDY_TRUE_ROOM_ID = "f8-s860-r2"  # the windowed private office in scanned Suite 860


def bundy_olympic_call_script() -> list[dict]:
    """Demo call for the real building (11900 W Olympic Blvd, Los Angeles).

    The caller is in a private office of Suite 860 on the 8th floor, which looks
    out over the parking deck. Floor 8 is 25.6 m above the ground floor (12 ft
    slab to slab), so a held phone reads about 26.8 m.
    """
    return _turns(
        [
            (
                "Call connected",
                "system",
                "911 call connected. Dispatchable address: 11900 W Olympic Blvd, Los Angeles "
                "(Bundy Olympic). No device location yet - showing the building prior.",
                None,
            ),
            (
                "Phone z-axis fix",
                "system",
                "Device barometric z-axis fix received: 26.4 m above ground (±3 m). "
                "No usable horizontal fix indoors.",
                PhoneFix(z_m=26.4),
            ),
            (
                "Caller: room and floor",
                "caller",
                "I'm in an office on the top floor, I locked the door",
                None,
            ),
            (
                "Caller: view",
                "caller",
                "I can see the parking structure out the window",
                None,
            ),
        ]
    )


WASHINGTON_BUILDING_ID = "washington-5285"
WASHINGTON_TRUE_ROOM_ID = "f2-r35"  # "Office 9": a rear office on the second floor


def washington_call_script() -> list[dict]:
    """Demo call for 5285 W Washington Blvd, the building with a whole-building scan.

    The caller is in a second-floor office at the back, overlooking the houses
    behind. The second floor is only 3.0 m above the first, so the z-axis fix
    (accurate to +/-3 m) cannot separate the two floors by itself; what the
    caller says has to.
    """
    return _turns(
        [
            (
                "Call connected",
                "system",
                "911 call connected. Dispatchable address: 5285 W Washington Blvd, Los Angeles. "
                "No device location yet - showing the building prior.",
                None,
            ),
            (
                "Phone z-axis fix",
                "system",
                "Device barometric z-axis fix received: 4.4 m above ground (±3 m). "
                "The floors are 3 m apart, so this leans to the second floor but cannot rule out the first.",
                PhoneFix(z_m=4.4),
            ),
            (
                "Caller: room and floor",
                "caller",
                "I'm on the second floor in an office, I locked the door",
                None,
            ),
            (
                # A route clue: the building has two stairs, and the offices are on the
                # right of one and on the left of the other.
                "Caller: route",
                "caller",
                "I came up the stairs and turned right to get to this room",
                None,
            ),
            (
                "Caller: view",
                "caller",
                "I can see houses and a backyard out the window",
                None,
            ),
        ]
    )


_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
             8: "eighth", 9: "ninth", 10: "tenth"}


def generic_call_script(building) -> list[dict]:
    """A short call for a building generated at call time: address, a height fix, the floor.

    There is no scripted ground truth for such a building, so the "caller" is put in a
    unit picked deterministically (two thirds of the way through the unit list) and only
    says which floor they are on. Everything else is typed in by the dispatcher.
    """
    rows: list[tuple[str, str, str, PhoneFix | None]] = [(
        "Call connected", "system",
        f"911 call connected. Dispatchable address: {building.name}. "
        "No device location yet - showing the building prior.", None,
    )]
    units = sorted((r for r in building.rooms if r.room_type in ("unit", "suite", "office", "room")), key=lambda r: r.id)
    if units and len(building.floors) > 1:
        room = units[(2 * len(units)) // 3]
        z = round(float(building.floor_heights_m.get(room.floor, 0.0)) + 1.2, 1)
        rows.append(("Phone z-axis fix", "system",
                     f"Device barometric z-axis fix received: {z} m above ground (±3 m). "
                     "No usable horizontal fix indoors.", PhoneFix(z_m=z)))
        said = f"the {_ORDINALS[room.floor]} floor" if room.floor in _ORDINALS else f"floor {room.floor}"
        rows.append(("Caller: floor", "caller", f"I'm on {said}", None))
    return _turns(rows)


def demo_script_for(building_id: str, building=None) -> list[dict]:
    """The scripted demo call that belongs to a building (synthetic script by default).

    A building generated at call time (``prop-...`` / ``mp-...``) gets ``generic_call_script``.
    """
    if building_id == BUNDY_BUILDING_ID:
        return bundy_olympic_call_script()
    if building_id == WASHINGTON_BUILDING_ID:
        return washington_call_script()
    if building is not None and building_id.startswith(("prop-", "mp-")):
        return generic_call_script(building)
    return demo_call_script()


def scripted_calls() -> dict[str, list[dict]]:
    """All scripted scenarios, keyed by name. ``"demo"`` is ``demo_call_script()``."""
    return {
        "demo": demo_call_script(),
        # Caller trapped in the elevator car; coarse horizontal fix, floor guess is hedged.
        "elevator_stuck": _turns(
            [
                ("Call connected", "system",
                 "911 call connected. Dispatchable address: 100 Princeton Tech Center.", None),
                ("Phone fix", "system",
                 "Device fix received: z 5.4 m above ground (±3 m), horizontal uncertainty 20 m.",
                 PhoneFix(x_m=24.0, y_m=12.0, horizontal_sigma_m=20.0, z_m=5.4)),
                ("Caller: room type", "caller",
                 "I'm stuck in the elevator, it stopped and the doors won't open", None),
                ("Caller: floor", "caller",
                 "I think it was passing the second floor when it stopped", None),
            ]
        ),
        # Whispering caller hiding in an interior room: mostly negative evidence.
        "hiding_no_windows": _turns(
            [
                ("Call connected", "system",
                 "911 call connected. Dispatchable address: 100 Princeton Tech Center.", None),
                ("Phone z-axis fix", "system",
                 "Device barometric z-axis fix received: 9.0 m above ground (±3 m).",
                 PhoneFix(z_m=9.0)),
                ("Caller: room type", "caller",
                 "I'm hiding in a supply closet, I have to be quiet", None),
                ("Caller: negatives", "caller",
                 "There's no window in here but I can hear the elevator dinging", None),
            ]
        ),
        # Ground-floor medical call with a street view and no z fix at all.
        "lobby_collapse": _turns(
            [
                ("Call connected", "system",
                 "911 call connected. Dispatchable address: 100 Princeton Tech Center. "
                 "No device location available.", None),
                ("Caller: floor", "caller",
                 "He collapsed, we're on the ground floor", None),
                ("Caller: landmark + view", "caller",
                 "We're by the reception desk and I can see the street through the windows", None),
            ]
        ),
    }

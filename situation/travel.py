"""Travel profile for the crew in this situation (step B1).

Numbers come from configs/situations.yaml (``travel`` section), each with its label and
source. Built-in fallbacks equal the values there, so a missing or partial config still
gives the same profile.

* Walking 1.2 m/s in clear air.
* Stairs 15 s per floor for a light load, 29 s per floor for ``load == "heavy"``; 90 s per
  floor above floor 18 for any load (judgement, not timed).
* ``smoke_reported``: movement 0.5 m/s and rooms searched at a search-advance rate of
  0.040 m/s, with a note on the profile saying so.
* Elevator option (45 s per ride + 2 s per floor) for medical incident types only, never for
  ``fire_smoke`` or when smoke is reported. Where it is offered, the planner graph routes by
  elevator when a ride is faster than the stairs: the vertical leg alone favours the ride from
  4 floors with a light load and from 2 with a heavy one (walks to the elevator and the stairs
  can shift that); ``planning.approach.recommend_approach`` also shows stair and elevator
  times side by side.

On a building with fewer than 18 floors, a light-load profile gives today's stair and walking
times (1.2 m/s, 15 s per floor).
"""

from __future__ import annotations

from typing import Any

from core.schema import Situation, TravelProfile
from situation.config import section, setting_value

_FALLBACK: dict[str, float] = {
    "walk_mps": 1.2,
    "stair_light": 15.0,
    "stair_heavy": 29.0,
    "stair_high": 90.0,
    "high_floor_from": 18,
    "smoke_search_advance_mps": 0.040,
    "smoke_walk_mps": 0.5,
    "elevator_fixed_s": 45.0,
    "elevator_s_per_floor": 2.0,
}


def _get(cfg: dict, path: tuple[str, ...], key: str) -> Any:
    node: Any = cfg
    for p in path:
        node = node.get(p) if isinstance(node, dict) else None
    return setting_value(node, _FALLBACK[key])


def travel_settings() -> dict[str, Any]:
    """The travel numbers in use, flat (config values, falling back to the built-in ones)."""
    cfg = section("travel")
    return {
        "walk_mps": float(_get(cfg, ("walk_mps",), "walk_mps")),
        "stair_light": float(_get(cfg, ("stair_s_per_floor", "light"), "stair_light")),
        "stair_heavy": float(_get(cfg, ("stair_s_per_floor", "heavy"), "stair_heavy")),
        "stair_high": float(_get(cfg, ("stair_s_per_floor_high",), "stair_high")),
        "high_floor_from": int(_get(cfg, ("high_floor_from",), "high_floor_from")),
        "smoke_search_advance_mps": float(_get(cfg, ("smoke", "search_advance_mps"), "smoke_search_advance_mps")),
        "smoke_walk_mps": float(_get(cfg, ("smoke", "walk_mps"), "smoke_walk_mps")),
        "elevator_fixed_s": float(_get(cfg, ("elevator", "fixed_s"), "elevator_fixed_s")),
        "elevator_s_per_floor": float(_get(cfg, ("elevator", "s_per_floor"), "elevator_s_per_floor")),
    }


def is_medical(incident_type: str) -> bool:
    """True for the incident types configs/situations.yaml marks ``medical`` (fire and threat are not)."""
    if incident_type in ("fire_smoke", "active_threat"):
        return False
    entry = (section("incident_types") or {}).get(incident_type) or {}
    return bool(entry.get("medical", True))


def elevator_allowed_for(situation: Situation) -> bool:
    """Elevator option: medical incident types only; never fire_smoke, never with smoke reported."""
    if situation.incident_type == "fire_smoke" or situation.smoke_reported:
        return False
    return is_medical(situation.incident_type)


def profile_for(situation: Situation) -> TravelProfile:
    """Movement rates for the crew in ``situation`` (see the module docstring)."""
    s = travel_settings()
    heavy = situation.load == "heavy"
    notes: list[str] = []

    walk = s["walk_mps"]
    search_advance = None
    if situation.smoke_reported:
        walk = s["smoke_walk_mps"]
        search_advance = s["smoke_search_advance_mps"]
        notes.append(
            f"Estimate uses smoke rates: rooms searched at a search-advance rate of {search_advance:.3f} m/s "
            f"(firefighters with a hose in live fire) and movement at {walk:.1f} m/s."
        )
    stair = s["stair_heavy"] if heavy else s["stair_light"]
    if heavy:
        notes.append(f"Heavy load: stairs at {stair:.0f} s per floor.")

    elevator = elevator_allowed_for(situation)
    return TravelProfile(
        walk_mps=walk,
        stair_s_per_floor=stair,
        stair_s_per_floor_high=s["stair_high"],
        high_floor_from=s["high_floor_from"],
        search_advance_mps=search_advance,
        elevator_allowed=elevator,
        elevator_fixed_s=s["elevator_fixed_s"] if elevator else None,
        elevator_s_per_floor=s["elevator_s_per_floor"] if elevator else None,
        notes=notes,
    )

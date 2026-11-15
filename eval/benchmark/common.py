"""Shared pieces of the benchmark: call types, buildings, the three search arms, one trial.

Arms (every arm starts at the same entrance, searches the same set of rooms, and uses the
same travel and room-search times, so only the ORDER of the search differs):

  A  address only        nearest-first sweep of the whole building
  B  informed dispatcher the caller's stated floor and room type are relayed by voice; the
                         crew checks rooms of that type on that floor first, then the rest of
                         that floor, then that type elsewhere, then everything (each tier
                         nearest-first)
  C  app                 the app's search plan on the fused posterior (phone z-axis fix +
                         parsed caller clues); for cardiac arrest it optimises survival

Time to patient = call-to-curb (shared, drawn per trial) + curb-to-entrance (shared
constant) + in-building time until the search of the TRUE room completes (per arm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Shared timing assumptions
# ---------------------------------------------------------------------------

# Call-to-curb (911 call to EMS arrival on scene), urban US: median 7.0 min, IQR 5.0-10.0
# (Mell et al. 2017, JAMA Surgery 152:983, NEMSIS). Lognormal with that median and the IQR's
# log-width: sigma = ln(10/5) / (2 * 0.6745) = 0.514 (0.6745 = standard-normal 75th percentile).
CURB_MEDIAN_S = 420.0  # 7.0 min
CURB_SIGMA = math.log(10.0 / 5.0) / (2 * 0.6745)
# Curb to the building entrance, shared by all arms. Judgement, not measured: it cancels in every
# A/B/C time difference, but for cardiac arrest it moves where on the survival curve contact falls.
ENTRY_S = 60.0  # 1 min, as stated in REPORT.md

# Phone fix: z-axis present 90% of the time (FCC z-axis: +-3 m for 80% of calls, the
# model's own sigma); horizontal fix present half the time with sigma 30 m, deliberately
# looser than the simulator's 15 m default (FCC horizontal benchmark is 50 m for 80%).
# The two availability rates (0.9, 0.5) are assumptions; no measured rate is cited in the repo.
# They equal sim.sample_caller.sample_phone_fix's defaults; the no_phone variant drops the fix.
P_Z = 0.9
P_HORIZONTAL = 0.5
HORIZONTAL_SIGMA_M = 30.0

BUILDINGS = {
    "ptc-100": "Synthetic office (4 floors, 100 rooms)",
    "washington-5285": "5285 W Washington Blvd (real scan, 3 levels, 38 rooms)",
    "bundy-olympic": "Bundy Olympic tower (8 storeys, 148 rooms)",
    "jrr-princeton": "JRR Building, Princeton (5 levels, 208 rooms)",
}
BUILDING_SHORT = {
    "ptc-100": "Synthetic office",
    "washington-5285": "5285 W Washington",
    "bundy-olympic": "Bundy Olympic",
    "jrr-princeton": "JRR Princeton",
}


@dataclass(frozen=True)
class CallType:
    key: str
    label: str
    situation: Mapping[str, Any]  # core.schema.Situation fields; drive the travel profile and survival curve
    personas: Mapping[str, float]  # sim/personas.yaml name -> sampling weight (normalised in _persona_for)
    # curve: cardiac survival curve on the call clock; linear_call: rate x minutes from the call;
    # linear_arrival: rate x minutes after reaching the entrance; time_only: no outcome model.
    outcome: str  # "curve" | "linear_call" | "linear_arrival" | "time_only"
    rate_pp_min: Optional[tuple[float, float, float]] = None  # low, mid, high (linear only)
    outcome_label: str = ""
    complaint: str = ""  # for the speech generator


# Assumptions shared by every call type below:
# - load="heavy": the crew carries kit, so stairs cost 29 s per floor (ODPM 2/2005, configs/situations.yaml).
# - Persona mixes are judgement (who plausibly makes that call); no call-mix data is cited in the repo.
# - rate_pp_min is copied by hand from configs/situations.yaml survival.<type>.stake_per_min as
#   (range low, value, range high); it is not read from that file, so the two must be kept in sync.
# - Cardiac curves are not set here: situation.survival picks Valenzuela 1997 when witnessed is True
#   and De Maio 2003 otherwise, with coefficients from configs/situations.yaml.
CALL_TYPES: list[CallType] = [
    CallType(
        "cardiac_cpr", "Cardiac arrest, witnessed, bystander CPR",
        dict(incident_type="cardiac_arrest", witnessed=True, cpr_in_progress=True, load="heavy"),
        {"bystander": 0.5, "calm_employee": 0.3, "panicked": 0.2}, "curve",
        outcome_label="Survival to discharge (Valenzuela 1997)",
        complaint="a coworker collapsed in front of you and is not breathing; you are doing chest compressions",
    ),
    CallType(
        "cardiac_nocpr", "Cardiac arrest, witnessed, no CPR",
        dict(incident_type="cardiac_arrest", witnessed=True, cpr_in_progress=False, load="heavy"),
        {"panicked": 0.5, "bystander": 0.3, "elderly": 0.2}, "curve",
        outcome_label="Survival to discharge (Valenzuela 1997)",
        complaint="someone just collapsed in front of you and is not breathing; you do not know CPR",
    ),
    CallType(
        "cardiac_unwitnessed", "Cardiac arrest, unwitnessed",
        dict(incident_type="cardiac_arrest", witnessed=None, cpr_in_progress=None, load="heavy"),
        {"bystander": 0.5, "panicked": 0.3, "elderly": 0.2}, "curve",
        outcome_label="Survival to discharge (De Maio 2003)",
        complaint="you just found someone on the floor, unresponsive, not breathing",
    ),
    CallType(
        "fire_smoke", "Fire / smoke, trapped occupant",
        dict(incident_type="fire_smoke", smoke_reported=True, load="heavy"),
        {"panicked": 0.6, "calm_employee": 0.2, "elderly": 0.2}, "linear_arrival",
        # Voluntary survey in trade press, not a fitted curve (configs/situations.yaml survival.fire_smoke).
        rate_pp_min=(2.5, 3.75, 5.0),
        outcome_label="Survival points lost per minute after FD arrival (2.5-5 pp/min)",
        complaint="there is smoke filling the room and you cannot get out",
    ),
    CallType(
        "stroke", "Stroke",
        dict(incident_type="stroke", load="heavy"),
        {"elderly": 0.4, "bystander": 0.4, "calm_employee": 0.2}, "linear_call",
        # Labelled "derived" from the project's evidence report (configs/situations.yaml survival.stroke).
        rate_pp_min=(0.03, 0.08, 0.13),
        outcome_label="Good functional outcome lost per minute (0.03-0.13 pp/min)",
        complaint="the person with you suddenly can't speak properly and one side of their face is drooping",
    ),
    CallType(
        "internal_bleeding", "Internal bleeding",
        dict(incident_type="internal_bleeding", load="heavy"),
        {"injured": 0.6, "calm_employee": 0.2, "panicked": 0.2}, "linear_call",
        # Labelled "derived" from the evidence report (configs/situations.yaml survival.internal_bleeding).
        rate_pp_min=(0.2, 0.35, 0.5),
        outcome_label="Mortality added per minute (0.2-0.5 pp/min)",
        complaint="you fell hard, your belly hurts badly and you feel faint and dizzy",
    ),
    CallType(
        "overdose", "Overdose",
        dict(incident_type="overdose", load="heavy"),
        {"bystander": 0.5, "panicked": 0.4, "child": 0.1}, "time_only",
        outcome_label="No published survival curve: time only",
        complaint="someone with you took too many pills and is barely breathing",
    ),
]
CALL_TYPE_BY_KEY = {c.key: c for c in CALL_TYPES}

# Robustness variants (main = the default world). uniform_truth draws the true room uniformly instead of
# from the occupancy prior; error_x2 / error_x4 scale each persona's error_rate (see _persona_for);
# no_phone drops the fix (only arm C uses it); no_clues drops the clues for arms B and C alike, so arm B
# has no stated floor or type and its order collapses to arm A's nearest-first sweep.
VARIANTS = {
    "main": "Main benchmark",
    "uniform_truth": "Caller anywhere, equal odds",
    "error_x2": "Callers 2x as error-prone",
    "error_x4": "Callers 4x as error-prone",
    "no_phone": "No phone location fix",
    "no_clues": "No caller clues (phone fix only)",
}


# ---------------------------------------------------------------------------
# Per-process engine state
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def engine() -> dict[str, Any]:
    if _STATE:
        return _STATE
    from core.prior import compute_prior
    from ingest.matterport import generate_synthetic_office, load_building

    buildings = {}
    for bid in BUILDINGS:
        b = generate_synthetic_office() if bid == "ptc-100" else load_building(bid)
        # hour=14 throughout the benchmark: the same 2 pm business-hours prior as api.session.DEMO_HOUR.
        prior, _ = compute_prior(b, hour=14)
        # The searchable set: every room the prior gives any mass (all three arms sweep exactly these).
        universe = [r.id for r in b.rooms if prior.get(r.id, 0.0) > 0.0]
        buildings[bid] = {"b": b, "rooms": {r.id: r for r in b.rooms}, "universe": universe}
    _STATE["buildings"] = buildings
    return _STATE


# ---------------------------------------------------------------------------
# Search orders
# ---------------------------------------------------------------------------


def _t(travel, a, b) -> float:
    if a == b:
        return 0.0
    # Pair missing from the travel table (no path in the graph): charge 1 h so it sorts last in a
    # nearest-first chain. Placeholder penalty, not a measured time; the same for every arm.
    return travel.get(a, {}).get(b, 3600.0)


def nearest_chain(members: Sequence[str], here: str, travel) -> tuple[list[str], str]:
    """Greedy nearest-neighbour walk over ``members`` from ``here``; returns (order, last room)."""
    left = list(members)
    out = []
    while left:
        nxt = min(left, key=lambda r: (_t(travel, here, r), r))  # ties broken by room id, for determinism
        out.append(nxt)
        left.remove(nxt)
        here = nxt
    return out, here


def informed_order(universe, start, travel, rooms, stated_floor: Optional[int], stated_type: Optional[str]) -> list[str]:
    """Arm B: tier 0 = stated type on stated floor, 1 = rest of that floor, 2 = that type elsewhere,
    3 = everything else; each tier is walked nearest-first, continuing from where the last one ended.
    With neither a floor nor a type stated every room is tier 3, i.e. arm A's sweep."""
    def tier(rid: str) -> int:
        r = rooms[rid]
        on_floor = stated_floor is not None and r.floor == stated_floor
        is_type = stated_type is not None and r.room_type == stated_type
        if on_floor and is_type:
            return 0
        if on_floor:
            return 1
        if is_type:
            return 2
        return 3

    order: list[str] = []
    here = start
    for k in range(4):
        members = [rid for rid in universe if tier(rid) == k]
        chain, here = nearest_chain(members, here, travel)
        order += chain
    return order


def completion_time(order: Sequence[str], target: str, start: str, travel, search: Mapping[str, float]) -> float:
    """Seconds from the entrance until ``target``'s search is finished, walking ``order`` (travel + look per room).

    Every look finds a person who is present (no missed detections), for all three arms.
    """
    t, here = 0.0, start
    for rid in order:
        t += _t(travel, here, rid) + search[rid]
        if rid == target:
            return t
        here = rid
    raise KeyError(target)


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


def _persona_for(ct: CallType, rng: np.random.Generator, variant: str) -> dict:
    from sim.sample_caller import resolve_persona

    names = list(ct.personas)
    w = np.array([ct.personas[n] for n in names], dtype=float)
    name = names[int(rng.choice(len(names), p=w / w.sum()))]
    p = resolve_persona(name)
    if variant in ("error_x2", "error_x4"):
        k = 2.0 if variant == "error_x2" else 4.0
        # 0.9 cap keeps it a probability; never binds with the shipped personas (max 0.15 x 4 = 0.6).
        p = {**p, "error_rate": min(0.9, float(p["error_rate"]) * k)}
    return p


def sample_world(bid: str, ct: CallType, variant: str, rng: np.random.Generator) -> dict:
    """Ground-truth room, phone fix and structured clues for one caller."""
    from sim.sample_caller import sample_phone_fix, sample_true_room, truthful_clues

    E = engine()["buildings"][bid]
    b = E["b"]
    persona = _persona_for(ct, rng, variant)
    if variant == "uniform_truth":
        room = E["rooms"][E["universe"][int(rng.integers(len(E["universe"])))]]
    else:
        room = sample_true_room(b, rng, hour=14)
    fix = sample_phone_fix(b, room, rng, p_z=P_Z, p_horizontal=P_HORIZONTAL, horizontal_sigma_m=HORIZONTAL_SIGMA_M)
    clues = truthful_clues(b, room, rng, persona=persona)
    if variant == "no_phone":
        fix = None
    if variant == "no_clues":
        clues = []
    return {"room": room, "fix": fix, "clues": clues, "persona": persona.get("name", "custom")}


def stated_floor_and_type(b, clues) -> tuple[Optional[int], Optional[str]]:
    """What arm B's dispatcher relays: the first positive floor clue (mapped from its label to a floor
    index, so a wrong floor is relayed as said) and the first positive room-type clue. Negatives,
    views and landmarks are not relayed."""
    label_to_floor = {}
    for r in b.rooms:
        label_to_floor.setdefault(str(r.floor_label), r.floor)
    floor = rtype = None
    for c in clues:
        if not getattr(c, "polarity", True):
            continue
        if c.type == "floor" and floor is None:
            floor = label_to_floor.get(str(c.value))
        elif c.type == "room_type" and rtype is None:
            rtype = str(c.value)
    return floor, rtype


def run_arms(bid: str, ct: CallType, room, fix, clues, t_curb_s: float, *, evidence=None) -> dict:
    """Times for arms A, B and C given one caller's evidence.

    ``evidence`` (optional) replaces ``clues`` for the app's posterior only (used by the speech
    test, where the app reads parsed evidence but the informed dispatcher still hears the
    structured clues - the same information, so arm B is unchanged).
    """
    from core.fusion import posterior_probs
    from core.schema import Situation
    from planning.search_order import _graph_for, pairwise_travel, plan_search, search_time_s
    from situation.survival import survival_from
    from situation.travel import profile_for

    E = engine()["buildings"][bid]
    b, rooms, universe = E["b"], E["rooms"], E["universe"]
    situation = Situation(**ct.situation)
    profile = profile_for(situation)
    elapsed = t_curb_s + ENTRY_S
    # Cardiac only: the planner gets the survival curve on a clock starting at the entrance, so it
    # maximises survival; every other type is planned on expected time to find.
    survival = survival_from(situation, elapsed) if ct.outcome == "curve" else None

    room_probs, p_nim = posterior_probs(b, hour=14, fix=fix, clues=evidence if evidence is not None else clues)
    plan = plan_search(b, room_probs, p_nim, profile=profile, survival=survival)
    start = plan.entrance_id
    G = _graph_for(b, profile)
    travel = pairwise_travel(G, [start, *universe])
    search = {rid: search_time_s(rooms[rid], profile) for rid in universe}

    # C: the app's plan, then (if the caller is outside the 99% plan) a nearest-first sweep of the rest.
    # One crew: only the first team's stops are used (plan_search default n_teams=1).
    plan_order = list(dict.fromkeys(s.room_id for s in plan.teams[0] if s.room_id in search))
    in_plan = room.id in plan_order
    rest = [rid for rid in universe if rid not in set(plan_order)]
    tail, _ = nearest_chain(rest, plan_order[-1] if plan_order else start, travel)
    order_c = plan_order + tail

    order_a, _ = nearest_chain(universe, start, travel)
    sf, st = stated_floor_and_type(b, clues)
    order_b = informed_order(universe, start, travel, rooms, sf, st)

    ranked = sorted(room_probs, key=lambda r: -room_probs[r])
    return {
        "t_A": completion_time(order_a, room.id, start, travel, search),
        "t_B": completion_time(order_b, room.id, start, travel, search),
        "t_C": completion_time(order_c, room.id, start, travel, search),
        "in_plan": in_plan,
        "plan_len": len(plan_order),
        "rank_true": ranked.index(room.id) + 1 if room.id in room_probs else len(ranked) + 1,
        "p_true": float(room_probs.get(room.id, 0.0)),
        "stated_floor": sf,
        "stated_floor_correct": (sf == room.floor) if sf is not None else None,
        "stated_type": st,
        "objective": plan.objective,
        "entrance": start,
    }


def curb_time(seed_parts: Sequence[int]) -> float:
    """One lognormal call-to-curb draw (s). The seed omits the variant, so a given building, call type
    and trial index gets the same drive time in every variant; 9999 is an arbitrary tag that keeps this
    stream apart from the world streams seeded in ``trial``."""
    rng = np.random.default_rng([9999, *seed_parts])
    return float(CURB_MEDIAN_S * math.exp(CURB_SIGMA * rng.standard_normal()))


def trial(bid: str, ct_key: str, variant: str, i: int) -> dict:
    ct = CALL_TYPE_BY_KEY[ct_key]
    b_idx = list(BUILDINGS).index(bid)
    c_idx = [c.key for c in CALL_TYPES].index(ct_key)
    v_idx = list(VARIANTS).index(variant)
    # One independent, reproducible stream per (variant, building, call type, trial): reruns and any
    # worker split give identical rows.
    rng = np.random.default_rng([v_idx, b_idx, c_idx, i])
    w = sample_world(bid, ct, variant, rng)
    t_curb = curb_time([b_idx, c_idx, i])
    out = run_arms(bid, ct, w["room"], w["fix"], w["clues"], t_curb)
    return {
        "building": bid, "call_type": ct_key, "variant": variant, "trial": i,
        "persona": w["persona"], "true_room": w["room"].id, "true_type": w["room"].room_type,
        "true_floor": w["room"].floor, "n_clues": len(w["clues"]),
        "has_z": bool(w["fix"] is not None and w["fix"].z_m is not None),
        "has_h": bool(w["fix"] is not None and w["fix"].x_m is not None),
        "t_curb": t_curb, **out,
    }

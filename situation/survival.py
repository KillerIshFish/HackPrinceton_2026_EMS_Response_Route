"""Survival functions and the stake (what a minute costs) per incident type.

Numbers live in configs/situations.yaml (``survival`` section), each with its label and
source. Wording lives here.

Time conventions
----------------
* ``survival_for(situation)`` returns ``S(t_s)``: the probability the patient can still be
  saved if the crew reaches them ``t_s`` seconds after the call started (the call clock).
  The Valenzuela curve already includes the assumed minute from patient side to first
  shock, so a caller should not add treatment time for it again.
  Use ``survival_from(situation, elapsed_s)`` for a curve on a clock that starts now.
* ``stake_for(situation, elapsed_s)`` reports the per-minute loss as a range. For cardiac
  arrest the range covers reaching the patient anywhere between now and
  ``stake_horizon_min`` minutes from now (and, when the CPR state is not known, both CPR
  assumptions). The other quantified types use fixed published ranges.

Rules (docs/BUILD_PLAN.md section 3): a stake is never zero. A quantified stake always has
both bounds strictly above 0; a value under 0.1 is displayed as "under 0.1"
(``format_per_min`` / ``stake_text``). Active threat has no stake card (``None``).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from core.schema import Situation, Stake
from situation.config import section, setting_value

# Maps seconds since the time origin (the call clock) to the probability the patient is still savable.
SurvivalFn = Callable[[float], float]

UNQUANTIFIED_CRITICAL = "Unquantified, time-critical"
TIME_CRITICAL = "Time-critical"
SURVIVAL_UNIT = "percentage points of survival per minute"
# Smallest per-minute value ever reported. The curve never reaches zero, but floating point can.
MIN_REPORTED_PER_MIN = 0.01
DISPLAY_FLOOR = 0.1

_UNQUANTIFIED = {
    "external_bleeding": (UNQUANTIFIED_CRITICAL, "No published curve for this incident type."),
    "overdose": (UNQUANTIFIED_CRITICAL, "No published curve for this incident type."),
    "unknown": (TIME_CRITICAL, "Set the incident type to see what a minute costs."),
    "other_medical": (TIME_CRITICAL, "No published curve for this incident type."),
}

# Fixed ranges: (label, unit, origin, note). The numbers come from the config.
_FIXED = {
    "stroke": (
        "Chance of a good outcome lost per minute of delay",
        "percentage points of good outcome per minute",
        "Per minute of delay to treatment",
        "Linear estimate from stroke time-to-treatment studies. A linear cost leaves the search order unchanged.",
    ),
    "internal_bleeding": (
        "Mortality added per minute of delay",
        "percentage points of mortality per minute",
        "Per minute of delay to treatment",
        "Linear estimate. A linear cost leaves the search order unchanged.",
    ),
    "fire_smoke": (
        "Survival chance lost per minute",
        SURVIVAL_UNIT,
        "Counted from fire-department arrival to reaching the victim",
        "From a voluntary survey reported in trade press, not a fitted curve.",
    ),
}


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _cfg(*path: str) -> object:
    node: object = section("survival")
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"configs/situations.yaml: survival.{'.'.join(path)} is missing")
        node = node[key]
    return node


def _num(*path: str) -> float:
    return float(setting_value(_cfg(*path)))


def _range(*path: str) -> tuple[float, float]:
    entry = _cfg(*path)
    low, high = entry["range"]  # type: ignore[index]
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def _logistic_of_neg(x: float) -> float:
    """1 / (1 + exp(x)), without overflow."""
    if x >= 0:
        e = math.exp(-x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(x))


def de_maio(t_min: float) -> float:
    """De Maio: S = 1 / (1 + exp(0.679 + 0.262 t)), t in minutes from the call."""
    return _logistic_of_neg(_num("cardiac_arrest", "de_maio", "intercept") + _num("cardiac_arrest", "de_maio", "per_min") * t_min)


def valenzuela(t_cpr_min: float, t_defib_min: float) -> float:
    """Valenzuela: S = 1 / (1 + exp(-0.260 + 0.106 t_cpr + 0.139 t_defib)), minutes from collapse."""
    v = ("cardiac_arrest", "valenzuela")
    x = _num(*v, "intercept") + _num(*v, "per_min_to_cpr") * t_cpr_min + _num(*v, "per_min_to_defib") * t_defib_min
    return _logistic_of_neg(x)


def _valenzuela_at_contact(contact_min_from_call: float, bystander_cpr: bool) -> float:
    """Valenzuela survival when the crew reaches the patient this many minutes after the call.

    Collapse is assumed 1.2 min before the call and the first shock 1 min after patient
    contact (both modelling assumptions). With bystander CPR, CPR is assumed under way from
    the start of the call; without it, the crew starts CPR on arrival.
    """
    to_call = _num("cardiac_arrest", "collapse_to_call_min")
    to_shock = _num("cardiac_arrest", "patient_to_shock_min")
    contact = to_call + max(0.0, contact_min_from_call)
    t_defib = contact + to_shock
    cpr_start = to_call if bystander_cpr else contact
    return valenzuela(min(cpr_start, t_defib), t_defib)


def _use_valenzuela(situation: Situation) -> bool:
    return situation.witnessed is True


def survival_for(situation: Situation) -> Optional[SurvivalFn]:
    """Survival function for the planner's objective, or None for a linear cost.

    Only cardiac arrest has a curve. The returned function takes seconds since the call
    started (when the crew reaches the patient) and returns a probability in (0, 1).
    Witnessed arrest: Valenzuela (bystander CPR assumed only when cpr_in_progress is True).
    Otherwise: De Maio.
    """
    if situation.incident_type != "cardiac_arrest":
        return None
    if _use_valenzuela(situation):
        bystander = situation.cpr_in_progress is True

        def s_valenzuela(t_s: float) -> float:
            return _valenzuela_at_contact(max(0.0, t_s) / 60.0, bystander)

        return s_valenzuela

    def s_de_maio(t_s: float) -> float:
        return de_maio(max(0.0, t_s) / 60.0)

    return s_de_maio


def survival_from(situation: Situation, elapsed_s: float) -> Optional[SurvivalFn]:
    """``survival_for`` shifted so its clock starts now: ``S_now(t) = S(elapsed_s + t)``."""
    base = survival_for(situation)
    if base is None:
        return None
    offset = _clean_elapsed(elapsed_s)
    return lambda t_s: base(offset + max(0.0, t_s))


# ---------------------------------------------------------------------------
# Stake
# ---------------------------------------------------------------------------


def _clean_elapsed(elapsed_s: float) -> float:
    try:
        e = float(elapsed_s)
    except (TypeError, ValueError):
        return 0.0
    return e if math.isfinite(e) and e > 0 else 0.0


def _per_min_losses(curve: Callable[[float], float], now_min: float, horizon_min: float) -> list[float]:
    """Percentage points lost by one more minute, for contact times now .. now + horizon (half-minute steps)."""
    steps = max(1, int(round(horizon_min * 2)))
    out = []
    for k in range(steps + 1):
        c = now_min + k * horizon_min / steps
        out.append(100.0 * (curve(c) - curve(c + 1.0)))
    return out


def _floor_value(v: float) -> float:
    return max(round(v, 2), MIN_REPORTED_PER_MIN)


def _cardiac_stake(situation: Situation, elapsed_s: float) -> Stake:
    now_min = _clean_elapsed(elapsed_s) / 60.0
    horizon = _num("cardiac_arrest", "stake_horizon_min")
    window = f"Range covers reaching the patient between now and {horizon:g} minutes from now."
    if _use_valenzuela(situation):
        if situation.cpr_in_progress is True:
            options, cpr_note = [True], "CPR assumed under way from the start of the call"
        elif situation.cpr_in_progress is False:
            options, cpr_note = [False], "no bystander CPR; the crew starts CPR on arrival"
        else:
            options, cpr_note = [True, False], "CPR state not recorded, so the range covers both with and without bystander CPR"
        losses = [
            v for b in options
            for v in _per_min_losses(lambda c, b=b: _valenzuela_at_contact(c, b), now_min, horizon)
        ]
        to_call = _num("cardiac_arrest", "collapse_to_call_min")
        to_shock = _num("cardiac_arrest", "patient_to_shock_min")
        origin = f"Counted from the call, plus an assumed {to_call:g} minutes from collapse to call"
        note = (
            f"Valenzuela curve for a witnessed arrest; {cpr_note}. Assumes {to_shock:g} minute from "
            f"patient side to first shock (modelling assumption). {window}"
        )
    else:
        losses = _per_min_losses(de_maio, now_min, horizon)
        fitted = _num("cardiac_arrest", "de_maio", "fitted_to_min")
        origin = "Counted from the call"
        note = (
            f"De Maio curve (arrest not known to be witnessed). It is extrapolated beyond about "
            f"{fitted:g} minutes from the call. {window}"
        )
        if now_min + horizon > fitted:
            note += " Part or all of this range lies past that point, so those values are extrapolated."
    low, high = _floor_value(min(losses)), _floor_value(max(losses))
    if low < DISPLAY_FLOOR:
        note += " Values below 0.1 points per minute are shown as under 0.1, never as zero."
    return Stake(
        label="Survival chance lost per minute of delay",
        quantified=True,
        per_min_low=low,
        per_min_high=high,
        unit=SURVIVAL_UNIT,
        origin=origin,
        note=note,
    )


def stake_for(situation: Situation, elapsed_s: float) -> Optional[Stake]:
    """What a minute costs at ``elapsed_s`` seconds since the call started.

    Cardiac arrest: derived from its curve at the current time. Stroke, internal bleeding,
    fire: fixed published ranges. External bleeding, overdose: "Unquantified, time-critical".
    Unknown, other medical: "Time-critical". Active threat: None (no stake card).
    Never returns a quantified stake with a bound of 0.
    """
    kind = situation.incident_type
    if kind == "cardiac_arrest":
        return _cardiac_stake(situation, elapsed_s)
    if kind in _FIXED:
        label, unit, origin, note = _FIXED[kind]
        low, high = _range(kind, "stake_per_min")
        return Stake(
            label=label, quantified=True,
            per_min_low=max(low, MIN_REPORTED_PER_MIN), per_min_high=max(high, MIN_REPORTED_PER_MIN),
            unit=unit, origin=origin, note=note,
        )
    if kind in _UNQUANTIFIED:
        label, note = _UNQUANTIFIED[kind]
        return Stake(label=label, quantified=False, note=note)
    return None  # active_threat: the incident board replaces the stake card


# ---------------------------------------------------------------------------
# Display (the web console applies the same rule)
# ---------------------------------------------------------------------------


def format_per_min(v: float) -> str:
    """One bound as text: "under 0.1" below 0.1, two decimals below 1, one decimal otherwise."""
    if v < DISPLAY_FLOOR:
        return "under 0.1"
    if v < 1.0:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.1f}"


def stake_text(stake: Optional[Stake]) -> str:
    """The stake as one line of text; '' when there is no stake card."""
    if stake is None:
        return ""
    if not stake.quantified or stake.per_min_low is None or stake.per_min_high is None:
        return stake.label
    low, high = format_per_min(stake.per_min_low), format_per_min(stake.per_min_high)
    span = low if low == high else f"{low} to {high}"
    return f"{span} {stake.unit or ''}".strip()

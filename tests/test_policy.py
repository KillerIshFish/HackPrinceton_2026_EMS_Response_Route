"""Step B2: incident policy, survival functions and stake (docs/BUILD_PLAN.md sections 3 and 6).

One test (or more) per rule. These rules carry the console's safety behaviour.
"""

from __future__ import annotations

import itertools
import math
import re
from datetime import datetime, timezone

import pytest

import situation.policy as policy_mod
from core.schema import INCIDENT_TYPES, Fact, Situation, TravelProfile
from ingest.matterport import load_building
from situation.config import LABELS, section
from situation.policy import (
    ACTIVE_THREAT_TEXT,
    CPR_TEXT,
    FIRE_ELEVATOR_TEXT,
    FIRE_HAZARD_CLASS_TEXT,
    RULES,
    policy_for,
)
from situation.survival import (
    de_maio,
    format_per_min,
    stake_for,
    stake_text,
    survival_for,
    survival_from,
    valenzuela,
)

MEDICAL = ("unknown", "cardiac_arrest", "stroke", "internal_bleeding", "external_bleeding", "overdose", "other_medical")
FORBIDDEN = ("safe", "empty", "unsurvivable", "no known hazard", "none known")
ELEVATOR_PROFILE = TravelProfile(elevator_allowed=True, elevator_fixed_s=75.0, elevator_s_per_floor=2.0)


@pytest.fixture(scope="module")
def building():
    return load_building("ptc-100")


@pytest.fixture
def elevator_travel(monkeypatch):
    """A travel profile that offers an elevator, so the policy rule alone decides (B1 independent)."""
    monkeypatch.setattr(policy_mod, "profile_for", lambda situation: ELEVATOR_PROFILE.model_copy())


CALL_TIME = datetime(2026, 11, 14, 19, 2, 31, tzinfo=timezone.utc)


def hazard(text: str, *, at: datetime | None = CALL_TIME, source: str = "agent") -> Fact:
    return Fact(id=f"f-{abs(hash(text))}", category="hazard", text=text, source=source, observed_at=at)


def texts(pol) -> list[str]:
    return [b.text for b in pol.banners]


# ---------------------------------------------------------------------------
# The display table
# ---------------------------------------------------------------------------


def test_rules_cover_every_incident_type() -> None:
    assert set(RULES) == set(INCIDENT_TYPES)


@pytest.mark.parametrize("kind", INCIDENT_TYPES)
def test_policy_follows_the_table(building, elevator_travel, kind: str) -> None:
    route, elevator, hazard_class, board = RULES[kind]
    pol = policy_for(building, Situation(incident_type=kind, incident_confirmed=True), [])
    assert (pol.route_enabled, pol.elevator_allowed, pol.show_hazard_class, pol.show_incident_board) == (
        route, elevator, hazard_class, board)
    expected = {
        "fire_smoke": (True, False, True, False),
        "active_threat": (False, False, False, True),
    }.get(kind, (True, True, False, False))
    assert (route, elevator, hazard_class, board) == expected


# ---------------------------------------------------------------------------
# Active threat: no route
# ---------------------------------------------------------------------------


def test_active_threat_has_route_off(building, elevator_travel) -> None:
    pol = policy_for(building, Situation(incident_type="active_threat", incident_confirmed=True), [])
    assert pol.route_enabled is False
    assert pol.show_incident_board is True
    assert pol.elevator_allowed is False and pol.travel.elevator_allowed is False
    assert pol.travel.elevator_fixed_s is None
    assert pol.protocol_prompts == []
    assert pol.stake is None
    assert pol.banners[0].text == ACTIVE_THREAT_TEXT and pol.banners[0].level == "critical"


@pytest.mark.parametrize("current", ["unknown", "cardiac_arrest", "fire_smoke"])
def test_unconfirmed_active_threat_suggestion_has_route_off(building, elevator_travel, current: str) -> None:
    sit = Situation(incident_type=current, incident_confirmed=current != "unknown",
                    suggested_type="active_threat", suggested_span="someone is shooting")
    pol = policy_for(building, sit, [])
    assert pol.route_enabled is False
    assert pol.elevator_allowed is False
    assert pol.protocol_prompts == []
    pending = [b for b in pol.banners if b.level == "critical"]
    assert pending and "confirms" in pending[0].text and "incident type" in pending[0].text
    assert '"someone is shooting"' in pending[0].text


def test_dismissed_active_threat_suggestion_restores_the_route(building) -> None:
    sit = Situation(incident_type="cardiac_arrest", incident_confirmed=True,
                    suggested_type="active_threat", dismissed_types=["active_threat"])
    pol = policy_for(building, sit, [])
    assert pol.route_enabled is True
    assert not any(b.level == "critical" for b in pol.banners)


def test_active_threat_never_has_protocol_prompts(building, monkeypatch) -> None:
    real = section("policy")
    fake = {**real, "protocol_prompts": {**(real.get("protocol_prompts") or {}), "active_threat": ["Go to the caller."]}}
    monkeypatch.setattr(policy_mod, "section", lambda name: fake if name == "policy" else section(name))
    pol = policy_for(building, Situation(incident_type="active_threat", incident_confirmed=True), [])
    assert pol.protocol_prompts == []
    assert section("policy")["protocol_prompts"]["active_threat"] == []


@pytest.mark.parametrize("suggested", ["cardiac_arrest", "fire_smoke", "overdose"])
def test_other_suggestions_change_nothing_until_confirmed(building, elevator_travel, suggested: str) -> None:
    base = policy_for(building, Situation(), [])
    pol = policy_for(building, Situation(suggested_type=suggested, suggested_span="he collapsed"), [])
    flags = lambda p: (p.route_enabled, p.elevator_allowed, p.show_hazard_class, p.show_incident_board,
                       p.stake, p.travel, p.protocol_prompts)
    assert flags(pol) == flags(base)
    extra = [b for b in pol.banners if b not in base.banners]
    assert len(extra) == 1 and extra[0].level == "info" and "until the dispatcher confirms" in extra[0].text


# ---------------------------------------------------------------------------
# Elevator: never with fire, smoke or an alarm
# ---------------------------------------------------------------------------


def test_fire_has_no_elevator_option(building, elevator_travel) -> None:
    pol = policy_for(building, Situation(incident_type="fire_smoke", incident_confirmed=True), [])
    assert pol.elevator_allowed is False
    assert pol.travel.elevator_allowed is False
    assert pol.travel.elevator_fixed_s is None and pol.travel.elevator_s_per_floor is None
    assert pol.show_hazard_class is True
    assert texts(pol)[:2] == [FIRE_ELEVATOR_TEXT, FIRE_HAZARD_CLASS_TEXT]


@pytest.mark.parametrize("kind", MEDICAL)
def test_smoke_turns_the_elevator_off_for_a_medical_call(building, elevator_travel, kind: str) -> None:
    assert policy_for(building, Situation(incident_type=kind), []).elevator_allowed is True
    pol = policy_for(building, Situation(incident_type=kind, smoke_reported=True), [])
    assert pol.elevator_allowed is False
    assert pol.travel.elevator_allowed is False and pol.travel.elevator_fixed_s is None
    assert FIRE_ELEVATOR_TEXT in texts(pol)
    assert pol.route_enabled is True


@pytest.mark.parametrize("words", ["The fire alarm is going off", "I can smell smoke", "flames in the kitchen"])
def test_hazard_report_of_fire_smoke_or_alarm_turns_the_elevator_off(building, elevator_travel, words: str) -> None:
    pol = policy_for(building, Situation(incident_type="cardiac_arrest"), [hazard(words)])
    assert pol.elevator_allowed is False and pol.travel.elevator_allowed is False


def test_other_hazard_report_keeps_the_elevator(building, elevator_travel) -> None:
    pol = policy_for(building, Situation(incident_type="cardiac_arrest"), [hazard("Broken glass on the stairs")])
    assert pol.elevator_allowed is True


def test_elevator_offered_only_with_elevator_timings(building, monkeypatch) -> None:
    monkeypatch.setattr(policy_mod, "profile_for", lambda situation: TravelProfile())
    assert policy_for(building, Situation(incident_type="cardiac_arrest"), []).elevator_allowed is False


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------


def test_cpr_banner_appears(building) -> None:
    pol = policy_for(building, Situation(incident_type="cardiac_arrest", cpr_in_progress=True), [])
    assert CPR_TEXT in texts(pol)
    assert CPR_TEXT == "CPR in progress does not lower urgency."
    for cpr in (False, None):
        assert CPR_TEXT not in texts(policy_for(building, Situation(incident_type="cardiac_arrest", cpr_in_progress=cpr), []))


def test_required_banner_wording() -> None:
    assert ACTIVE_THREAT_TEXT == "Route planning is off. Caller reports below are for relay to unified command."
    assert FIRE_ELEVATOR_TEXT == "Elevator routes are not offered while fire or smoke is reported."
    assert FIRE_HAZARD_CLASS_TEXT == ("Hazard class is shown for information. Whether a space can be searched "
                                      "is the incident commander's decision.")


@pytest.mark.parametrize("kind", INCIDENT_TYPES)
def test_hazard_fact_banner_has_text_source_and_time(building, kind: str) -> None:
    facts = [hazard("Smoke in the third-floor hallway"),
             Fact(id="p", category="patient", text="Not breathing", source="agent"),
             hazard("Gas smell by the boiler", source="dispatcher", at=None)]
    pol = policy_for(building, Situation(incident_type=kind), facts)
    by_text = {b.text: b for b in pol.banners}
    b = by_text["Smoke in the third-floor hallway"]
    assert b.source == "Caller, 19:02:31 UTC, unverified" and b.level == "warning"
    assert by_text["Gas smell by the boiler"].source == "Dispatcher, time not recorded, unverified"
    assert "Not breathing" not in by_text


def test_hazard_fact_with_a_console_forbidden_word_is_marked_as_reported(building) -> None:
    pol = policy_for(building, Situation(), [hazard("the stairs are not safe")])
    b = [b for b in pol.banners if "stairs" in b.text][0]
    assert b.text == 'Caller reported: "the stairs are not safe"'


# ---------------------------------------------------------------------------
# Stake: never zero
# ---------------------------------------------------------------------------

ELAPSED = (0.0, 30.0, 300.0, 600.0, 1800.0, 7200.0, 86400.0, 1e9, -5.0, float("nan"))


@pytest.mark.parametrize("kind", INCIDENT_TYPES)
def test_no_stake_is_zero(kind: str) -> None:
    for witnessed, cpr, elapsed in itertools.product((True, False, None), (True, False, None), ELAPSED):
        sit = Situation(incident_type=kind, witnessed=witnessed, cpr_in_progress=cpr)
        stake = stake_for(sit, elapsed)
        if kind == "active_threat":
            assert stake is None
            continue
        assert stake is not None
        text = stake_text(stake)
        assert text and not re.search(r"(^|\s)0(\.0+)?(\s|$)", text), text
        if stake.quantified:
            assert stake.per_min_low is not None and stake.per_min_high is not None
            assert 0 < stake.per_min_low <= stake.per_min_high
            assert stake.unit and stake.origin and stake.note
        else:
            assert stake.per_min_low is None and stake.per_min_high is None
            assert stake.label in ("Unquantified, time-critical", "Time-critical")


def test_stake_labels_by_type() -> None:
    def s(kind: str):
        return stake_for(Situation(incident_type=kind), 0.0)

    for kind in ("external_bleeding", "overdose"):
        assert not s(kind).quantified and s(kind).label == "Unquantified, time-critical"
    for kind in ("unknown", "other_medical"):
        assert not s(kind).quantified and s(kind).label == "Time-critical"
    assert (s("stroke").per_min_low, s("stroke").per_min_high) == (0.03, 0.13)
    assert (s("internal_bleeding").per_min_low, s("internal_bleeding").per_min_high) == (0.2, 0.5)
    fire = s("fire_smoke")
    assert (fire.per_min_low, fire.per_min_high) == (2.5, 5.0)
    assert "fire-department arrival" in fire.origin and "trade press" in fire.note


def test_display_floor_is_under_point_one() -> None:
    assert format_per_min(0.004) == "under 0.1"
    assert format_per_min(0.13) == "0.13"
    assert format_per_min(2.345) == "2.3"
    late = stake_for(Situation(incident_type="cardiac_arrest"), 7200.0)
    assert late.quantified and stake_text(late).startswith("under 0.1")
    assert "never as zero" in late.note


def test_cardiac_stake_is_derived_from_the_curve() -> None:
    sit = Situation(incident_type="cardiac_arrest")  # witnessed unknown: De Maio
    stake = stake_for(sit, 0.0)
    assert stake.per_min_high == pytest.approx(100 * (de_maio(0) - de_maio(1)), abs=0.01)
    assert stake.per_min_low == pytest.approx(100 * (de_maio(5) - de_maio(6)), abs=0.01)
    later = stake_for(sit, 300.0)
    assert later.per_min_high == pytest.approx(stake.per_min_low, abs=0.01)
    assert "extrapolated beyond about 10 minutes" in stake.note
    assert stake.origin == "Counted from the call"
    wit = stake_for(Situation(incident_type="cardiac_arrest", witnessed=True, cpr_in_progress=True), 0.0)
    assert "1.2 minutes from collapse to call" in wit.origin and "Valenzuela" in wit.note
    assert "1 minute from patient side to first shock" in wit.note


# ---------------------------------------------------------------------------
# Survival functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("minutes,percent", [(5, 12.0), (6, 9.5), (7, 7.5), (8, 5.9)])
def test_de_maio_check_values(minutes: int, percent: float) -> None:
    assert round(100 * de_maio(minutes), 1) == percent
    for witnessed in (None, False):
        s = survival_for(Situation(incident_type="cardiac_arrest", witnessed=witnessed))
        assert round(100 * s(minutes * 60.0), 1) == percent


def test_valenzuela_formula_and_assumptions() -> None:
    def expected(t_cpr: float, t_defib: float) -> float:
        return 1 / (1 + math.exp(-0.260 + 0.106 * t_cpr + 0.139 * t_defib))

    assert valenzuela(3.0, 7.0) == pytest.approx(expected(3.0, 7.0))
    # Crew reaches the patient 4 min after the call: collapse 1.2 min before, shock 1 min after contact.
    with_cpr = survival_for(Situation(incident_type="cardiac_arrest", witnessed=True, cpr_in_progress=True))
    without = survival_for(Situation(incident_type="cardiac_arrest", witnessed=True, cpr_in_progress=False))
    assert with_cpr(240.0) == pytest.approx(expected(1.2, 6.2))
    assert without(240.0) == pytest.approx(expected(5.2, 6.2))
    assert with_cpr(240.0) > without(240.0)


@pytest.mark.parametrize("kind", INCIDENT_TYPES)
def test_survival_function_only_for_cardiac_arrest(kind: str) -> None:
    for witnessed in (True, False, None):
        s = survival_for(Situation(incident_type=kind, witnessed=witnessed))
        if kind != "cardiac_arrest":
            assert s is None
            continue
        values = [s(t) for t in (0.0, 60.0, 300.0, 600.0, 3600.0)]
        assert all(0.0 <= v < 1.0 for v in values) and values == sorted(values, reverse=True)
        assert s(1e9) >= 0.0  # no overflow


def test_survival_from_starts_the_clock_now() -> None:
    sit = Situation(incident_type="cardiac_arrest")
    assert survival_from(sit, 120.0)(60.0) == pytest.approx(survival_for(sit)(180.0))
    assert survival_from(Situation(incident_type="stroke"), 120.0) is None


def test_survival_settings_are_labelled() -> None:
    def leaves(node, path=()):
        if isinstance(node, dict) and "value" in node:
            yield path, node
        elif isinstance(node, dict):
            for k, v in node.items():
                yield from leaves(v, (*path, k))

    entries = list(leaves(section("survival")))
    assert len(entries) >= 11
    for path, entry in entries:
        assert entry.get("label") in LABELS and entry.get("source"), path


# ---------------------------------------------------------------------------
# Wording: never safe, empty or unsurvivable; never hold or enter
# ---------------------------------------------------------------------------


def _all_policies(building):
    facts_options = ([], [hazard("Smoke in the hallway"), hazard("Door to the stairwell is locked")])
    for kind, smoke, cpr, witnessed, suggested, facts in itertools.product(
            INCIDENT_TYPES, (False, True), (True, None), (True, None),
            (None, "active_threat", "cardiac_arrest", "fire_smoke"), facts_options):
        sit = Situation(incident_type=kind, smoke_reported=smoke, cpr_in_progress=cpr, witnessed=witnessed,
                        suggested_type=suggested, suggested_span="help, please hurry" if suggested else None)
        for elapsed in (0.0, 600.0, 7200.0):
            yield policy_for(building, sit, facts, elapsed_s=elapsed)


def test_policy_never_uses_forbidden_words(building) -> None:
    n = 0
    for pol in _all_policies(building):
        dump = pol.model_dump_json().lower()
        for word in FORBIDDEN:
            assert word not in dump, (word, dump)
        assert not re.search(r"\b(hold|enter)\b", dump), dump
        n += 1
    assert n > 1000


def test_configured_prompts_never_use_forbidden_words() -> None:
    prompts = section("policy")["protocol_prompts"]
    assert set(prompts) == set(INCIDENT_TYPES)
    for kind, items in prompts.items():
        for p in items:
            low = p.lower()
            assert not any(w in low for w in FORBIDDEN), p
            assert not re.search(r"\b(hold|enter|go to|run|move toward)\b", low), p

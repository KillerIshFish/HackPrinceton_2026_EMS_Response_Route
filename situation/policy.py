"""What the console may show for this situation (step B2).

The server computes the Policy; the UI only renders it. The display rules below are
requirements from docs/BUILD_PLAN.md section 3, not settings, so they live in code:

| incident_type        | route | elevator option | hazard class | incident board |
| unknown, other       | yes   | yes             | no           | no             |
| cardiac_arrest       | yes   | yes             | no           | no             |
| stroke               | yes   | yes             | no           | no             |
| internal_bleeding    | yes   | yes             | no           | no             |
| external_bleeding    | yes   | yes             | no           | no             |
| overdose             | yes   | yes             | no           | no             |
| fire_smoke           | yes   | no              | yes          | no             |
| active_threat        | no    | no              | no           | yes            |

* Smoke reported, or a hazard fact that mentions fire, smoke or an alarm, turns the elevator
  option off for any type (ground rule 4). The travel profile handed to the planner then
  carries no elevator either.
* Active threat is fail-safe: an active-threat suggestion the dispatcher has neither
  confirmed nor dismissed turns the route off and raises a banner asking for a decision.
  Every other suggestion changes nothing until confirmed.
* The elevator option is offered only when the travel profile (``profile_for``) also has
  elevator timings.

Protocol prompts per incident type come from configs/situations.yaml (``policy`` section);
active threat never has any.
"""

from __future__ import annotations

from datetime import timezone
from typing import Sequence

from core.schema import Banner, Building, Fact, Policy, Situation, TravelProfile
from situation.config import load_situations, section
from situation.survival import stake_for
from situation.travel import profile_for

# incident_type -> (route_enabled, elevator option, show_hazard_class, show_incident_board)
RULES: dict[str, tuple[bool, bool, bool, bool]] = {
    "unknown": (True, True, False, False),
    "other_medical": (True, True, False, False),
    "cardiac_arrest": (True, True, False, False),
    "stroke": (True, True, False, False),
    "internal_bleeding": (True, True, False, False),
    "external_bleeding": (True, True, False, False),
    "overdose": (True, True, False, False),
    "fire_smoke": (True, False, True, False),
    "active_threat": (False, False, False, True),
}

ACTIVE_THREAT_TEXT = "Route planning is off. Caller reports below are for relay to unified command."
ACTIVE_THREAT_PENDING_TEXT = (
    "The caller's words suggest an active threat. Route planning is withheld until the "
    "dispatcher confirms or dismisses the incident type."
)
FIRE_ELEVATOR_TEXT = "Elevator routes are not offered while fire or smoke is reported."
FIRE_HAZARD_CLASS_TEXT = (
    "Hazard class is shown for information. Whether a space can be searched is the incident "
    "commander's decision."
)
HAZARD_ELEVATOR_TEXT = (
    "Elevator routes are not offered: a caller report mentions fire, smoke or an alarm (unverified)."
)
CPR_TEXT = "CPR in progress does not lower urgency."
UNKNOWN_TEXT = "Incident type not set yet. Setting it adjusts travel times, the stake and what the console shows."

# Words that, in a hazard fact, withdraw the elevator option. Overridable in the config.
DEFAULT_ELEVATOR_OFF_WORDS = ("fire", "smoke", "flame", "burning", "alarm")
# The console never uses these words in its own text (ground rule 1).
FORBIDDEN_WORDS = ("safe", "empty", "unsurvivable")

_SPEAKER = {"rules": "Caller", "llm": "Caller", "agent": "Caller", "dispatcher": "Dispatcher", "record": "Premise record"}


def _type_label(incident_type: str) -> str:
    entry = (load_situations().get("incident_types") or {}).get(incident_type) or {}
    return str(entry.get("label") or incident_type.replace("_", " "))


def active_threat_pending(situation: Situation) -> bool:
    """An active-threat suggestion the dispatcher has neither confirmed nor dismissed."""
    return (
        situation.suggested_type == "active_threat"
        and situation.incident_type != "active_threat"
        and "active_threat" not in situation.dismissed_types
    )


def _elevator_off_words() -> tuple[str, ...]:
    words = section("policy").get("elevator_off_words")
    return tuple(str(w).lower() for w in words) if words else DEFAULT_ELEVATOR_OFF_WORDS


def _hazard_mentions_fire(facts: Sequence[Fact]) -> bool:
    words = _elevator_off_words()
    return any(f.category == "hazard" and any(w in f.text.lower() for w in words) for f in facts)


def _fact_source(fact: Fact) -> str:
    who = _SPEAKER.get(fact.source, "Caller")
    if fact.observed_at is not None:
        at = fact.observed_at
        if at.tzinfo is not None:
            at = at.astimezone(timezone.utc)
        when = at.strftime("%H:%M:%S UTC")
    else:
        when = "time not recorded"
    parts = [who, when]
    if not fact.verified:
        parts.append("unverified")
    return ", ".join(parts)


def _fact_text(fact: Fact) -> str:
    """Hazard text as reported. Reported speech is marked as such when it uses a word the
    console itself never uses, so the banner cannot read as the console's own claim."""
    text = fact.text.strip()
    if any(w in text.lower() for w in FORBIDDEN_WORDS):
        who = _SPEAKER.get(fact.source, "Caller")
        return f'{who} reported: "{text}"'
    return text


def _hazard_banners(facts: Sequence[Fact]) -> list[Banner]:
    out: list[Banner] = []
    seen: set[tuple[str, str]] = set()
    for f in facts:
        if f.category != "hazard" or not f.text.strip():
            continue
        b = Banner(level="warning", text=_fact_text(f), source=_fact_source(f))
        if (b.text, b.source) not in seen:
            seen.add((b.text, b.source))
            out.append(b)
    return out


def _protocol_prompts(incident_type: str) -> list[str]:
    if incident_type == "active_threat":
        return []
    prompts = (section("policy").get("protocol_prompts") or {}).get(incident_type) or []
    return [str(p) for p in prompts]


def _without_elevator(travel: TravelProfile) -> TravelProfile:
    if not travel.elevator_allowed and travel.elevator_fixed_s is None and travel.elevator_s_per_floor is None:
        return travel
    return travel.model_copy(update={"elevator_allowed": False, "elevator_fixed_s": None, "elevator_s_per_floor": None})


def policy_for(building: Building, situation: Situation, facts: Sequence[Fact], elapsed_s: float = 0.0) -> Policy:
    """Policy for this call. ``elapsed_s`` is seconds since ``situation.call_started_at`` (for the stake)."""
    kind = situation.incident_type
    route, elevator, hazard_class, board = RULES.get(kind, RULES["unknown"])
    pending_threat = active_threat_pending(situation)
    banners: list[Banner] = []

    if kind == "active_threat":
        banners.append(Banner(level="critical", text=ACTIVE_THREAT_TEXT, source="policy: active_threat"))
    if pending_threat:
        route = False
        span = (situation.suggested_span or "").strip()
        text = ACTIVE_THREAT_PENDING_TEXT + (f' Caller\'s words: "{span}"' if span else "")
        banners.append(Banner(level="critical", text=text, source="policy: active_threat suggested, unconfirmed"))

    if kind == "fire_smoke":
        banners.append(Banner(level="warning", text=FIRE_ELEVATOR_TEXT, source="policy: fire_smoke"))
        banners.append(Banner(level="info", text=FIRE_HAZARD_CLASS_TEXT, source="policy: fire_smoke"))
    elif situation.smoke_reported:
        elevator = False
        if route:
            banners.append(Banner(level="warning", text=FIRE_ELEVATOR_TEXT, source="policy: smoke reported"))
    if elevator and _hazard_mentions_fire(facts):
        elevator = False
        if route:
            banners.append(Banner(level="warning", text=HAZARD_ELEVATOR_TEXT, source="policy: hazard report"))
    if not route:
        elevator = False

    if kind == "cardiac_arrest" and situation.cpr_in_progress is True:
        banners.append(Banner(level="warning", text=CPR_TEXT, source="policy: cardiac_arrest"))
    if kind == "unknown":
        banners.append(Banner(level="info", text=UNKNOWN_TEXT, source="policy: unknown"))
    suggested = situation.suggested_type
    if (suggested and suggested != kind and suggested != "active_threat"
            and suggested not in situation.dismissed_types):
        span = (situation.suggested_span or "").strip()
        text = f"The caller's words suggest {_type_label(suggested).lower()}."
        if span:
            text += f' Caller\'s words: "{span}".'
        text += " Nothing changes until the dispatcher confirms it."
        banners.append(Banner(level="info", text=text, source="interpreter suggestion, unconfirmed"))

    banners.extend(_hazard_banners(facts))

    travel = profile_for(situation)
    if not elevator:
        travel = _without_elevator(travel)
    elevator = elevator and travel.elevator_allowed

    return Policy(
        route_enabled=route,
        elevator_allowed=elevator,
        show_hazard_class=hazard_class,
        show_incident_board=board,
        stake=stake_for(situation, elapsed_s),
        banners=banners,
        travel=travel,
        protocol_prompts=[] if pending_threat else _protocol_prompts(kind),
    )

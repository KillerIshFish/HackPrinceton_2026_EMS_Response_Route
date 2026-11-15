"""Sample synthetic callers: a ground-truth room, a noisy phone fix and truthful clues.

Used by ``eval/run_eval.py`` to measure how well the posterior and the search
plan do when the world behaves the way the model assumes (plus persona noise).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from core.prior import compute_prior
from core.schema import Z_SIGMA_M, Building, Clue, PhoneFix, Room
from parser.extract import load_vocab

PERSONAS_PATH = Path(__file__).resolve().parent / "personas.yaml"

# Same 1.2 m as core.features.PHONE_HEIGHT_ABOVE_FLOOR_M, so the simulated fix and the model agree on
# where a phone sits (assumed held or pocketed height, not measured).
PHONE_HEIGHT_M = 1.2  # phone is held ~1.2 m above the finished floor
# Assumed default, not sourced: tighter than the FCC 50 m / 80% horizontal benchmark. The benchmark
# (eval/benchmark/common.py) overrides it with a looser 30 m.
HORIZONTAL_SIGMA_M = 15.0  # typical indoor horizontal uncertainty

# Fills keys a persona leaves out; every persona in personas.yaml sets them all, so in practice this
# serves custom dict personas. Judgement: values sit inside the range
# spanned by sim/personas.yaml (e.g. clue_rate 0.40-0.85, error_rate 0.03-0.15); not fitted to calls.
_FALLBACK_PERSONA: dict[str, Any] = {
    "clue_rate": 0.7, "error_rate": 0.05, "hedge_rate": 0.15, "hedge_factor": 0.7,
    "negative_rate": 0.25, "knows_floor": 0.85, "max_clues": 4,
}

# The wrong side a mistaken caller names for a view (used when error_rate fires on a view clue).
_VIEW_OPPOSITE = {"parking_lot": "street", "street": "parking_lot"}


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


@lru_cache(maxsize=1)
def load_personas(path: str | None = None) -> dict:
    """Parsed ``sim/personas.yaml``: ``{"default": name, "personas": {name: params}}``."""
    with open(path or PERSONAS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_persona(persona: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Persona name / dict / None -> full parameter dict (missing keys filled in)."""
    if isinstance(persona, Mapping):
        return {**_FALLBACK_PERSONA, **persona}
    cfg = load_personas()
    name = persona or cfg.get("default")
    personas = cfg.get("personas") or {}
    if name not in personas:
        raise KeyError(f"unknown persona {name!r}; available: {sorted(personas)}")
    return {**_FALLBACK_PERSONA, **personas[name], "name": name}


# ---------------------------------------------------------------------------
# Ground truth + phone fix
# ---------------------------------------------------------------------------


def sample_true_room(
    building: Building, rng: np.random.Generator | int | None = None, hour: int = 14,
    priors_cfg: dict | None = None,
) -> Room:
    """Draw the caller's true room from the prior (conditioned on being in the model)."""
    rng = _rng(rng)
    prior, _ = compute_prior(building, hour=hour, priors_cfg=priors_cfg)
    rooms = [r for r in building.rooms if prior.get(r.id, 0.0) > 0.0]
    weights = np.array([prior[r.id] for r in rooms], dtype=float)
    return rooms[int(rng.choice(len(rooms), p=weights / weights.sum()))]


def sample_phone_fix(
    building: Building,
    room: Room,
    rng: np.random.Generator | int | None = None,
    p_z: float = 0.9,
    p_horizontal: float = 0.5,
    z_sigma_m: float = Z_SIGMA_M,
    horizontal_sigma_m: float = HORIZONTAL_SIGMA_M,
) -> PhoneFix | None:
    """Noisy device fix for a caller standing at the room centroid.

    z = floor height + 1.2 m + N(0, z_sigma). The horizontal fix is the centroid
    plus isotropic N(0, horizontal_sigma) noise. Each component is present only
    with probability ``p_z`` / ``p_horizontal``; returns None if both are missing.
    """
    rng = _rng(rng)
    z_m = x_m = y_m = h_sigma = None
    if rng.random() < p_z:
        z_true = float(building.floor_heights_m[room.floor]) + PHONE_HEIGHT_M
        z_m = round(z_true + float(rng.normal(0.0, z_sigma_m)), 2)
    if rng.random() < p_horizontal:
        x_m = round(room.centroid[0] + float(rng.normal(0.0, horizontal_sigma_m)), 2)
        y_m = round(room.centroid[1] + float(rng.normal(0.0, horizontal_sigma_m)), 2)
        h_sigma = horizontal_sigma_m
    if z_m is None and x_m is None:
        return None
    return PhoneFix(x_m=x_m, y_m=y_m, horizontal_sigma_m=h_sigma, z_m=z_m, z_sigma_m=z_sigma_m)


# ---------------------------------------------------------------------------
# Truthful clues
# ---------------------------------------------------------------------------


def _first_synonym(vocab: dict, section: str, key: str) -> str:
    entry = (vocab.get(section) or {}).get(key) or {}
    syns = entry.get("synonyms") or []
    return str(syns[0]) if syns else key.replace("_", " ")


def truthful_clues(
    building: Building,
    room: Room,
    rng: np.random.Generator | int | None = None,
    persona: str | Mapping[str, Any] | None = None,
    vocab: dict | None = None,
) -> list[Clue]:
    """Clues a caller in ``room`` would plausibly give, consistent with its attributes.

    Every true attribute (room type, floor, view, window, elevator / stairs
    nearby) is mentioned with probability ``clue_rate``; absences ("no window")
    with ``negative_rate``. With probability ``error_rate`` a mentioned floor is
    off by one or a view names the wrong side - the model's epsilon floor is
    what keeps those from being fatal. Hedged clues get a lower confidence.
    """
    rng = _rng(rng)
    vocab = vocab or load_vocab()
    p = resolve_persona(persona)
    conf = vocab.get("default_confidence") or {}
    room_types = vocab.get("room_types") or {}
    views = vocab.get("views") or {}

    def emit(ctype: str, value: str, span: str, polarity: bool = True) -> Clue:
        # Start from the vocabulary's default confidence for the clue type (configs/clue_vocab.yaml);
        # 0.7 only if the type is missing there.
        c = float(conf.get(ctype, 0.7))
        if rng.random() < p["hedge_rate"]:
            c *= float(p["hedge_factor"])
        return Clue(type=ctype, value=value, polarity=polarity, confidence=round(c, 3), span=span)  # type: ignore[arg-type]

    said = lambda rate: rng.random() < float(rate)  # noqa: E731
    wrong = lambda: rng.random() < float(p["error_rate"])  # noqa: E731
    clues: list[Clue] = []

    # Room type (never corrupted: callers know a bathroom from an office).
    if room.room_type in room_types and said(p["clue_rate"]):
        clues.append(emit("room_type", room.room_type, _first_synonym(vocab, "room_types", room.room_type)))

    # Floor, possibly miscounted by one (an off-by-one is the only floor error modelled).
    if said(float(p["clue_rate"]) * float(p["knows_floor"])):
        floor = room.floor
        if wrong():
            neighbours = [f for f in (floor - 1, floor + 1) if f in building.floors]
            if neighbours:
                floor = int(rng.choice(neighbours))
        label = next((r.floor_label for r in building.rooms if r.floor == floor), str(floor))
        clues.append(emit("floor", label, f"floor {label}"))

    # View out of the window, possibly naming the wrong side.
    if room.has_windows and room.window_facing in views and said(p["clue_rate"]):
        view = room.window_facing
        if wrong() and _VIEW_OPPOSITE.get(view) in views:
            view = _VIEW_OPPOSITE[view]
        clues.append(emit("view", view, _first_synonym(vocab, "views", view)))

    # Window: presence, or volunteered absence.
    if room.has_windows:
        if said(p["clue_rate"]):
            clues.append(emit("landmark", "window", "window"))
    elif said(p["negative_rate"]):
        clues.append(emit("negative", "landmarks:window", "no window", polarity=False))

    # Elevator / stairs proximity: presence, or volunteered absence. Tuned heuristic, unsourced: the 0.6
    # and 0.3 multipliers make callers mention a nearby core less often than the room itself, and volunteer
    # "no elevator / no stairs" rarely.
    for flag, key in ((room.near_elevator, "elevator"), (room.near_stairs, "stairs")):
        if flag:
            if said(float(p["clue_rate"]) * 0.6):
                clues.append(emit("landmark", key, _first_synonym(vocab, "landmarks", key)))
        elif said(float(p["negative_rate"]) * 0.3):
            clues.append(emit("negative", f"landmarks:{key}", f"no {key}", polarity=False))

    # Truncation keeps clues in the order built above, so a capped caller keeps room type and floor
    # first and drops elevator / stairs mentions first.
    max_clues = int(p.get("max_clues", len(clues)))
    return clues[:max_clues]


# ---------------------------------------------------------------------------
# One full synthetic caller
# ---------------------------------------------------------------------------


def sample_caller(
    building: Building,
    rng: np.random.Generator | int | None = None,
    hour: int = 14,
    persona: str | Mapping[str, Any] | None = None,
    priors_cfg: dict | None = None,
    p_z: float = 0.9,
    p_horizontal: float = 0.5,
    z_sigma_m: float = Z_SIGMA_M,
    horizontal_sigma_m: float = HORIZONTAL_SIGMA_M,
) -> dict[str, Any]:
    """Sample a ground-truth caller.

    Returns ``{"room_id", "room", "floor_label", "hour", "phone_fix", "clues", "persona"}``
    where ``phone_fix`` is a ``PhoneFix`` or None and ``clues`` is ``list[Clue]``.
    """
    rng = _rng(rng)
    room = sample_true_room(building, rng, hour=hour, priors_cfg=priors_cfg)
    fix = sample_phone_fix(
        building, room, rng, p_z=p_z, p_horizontal=p_horizontal,
        z_sigma_m=z_sigma_m, horizontal_sigma_m=horizontal_sigma_m,
    )
    clues = truthful_clues(building, room, rng, persona=persona)
    return {
        "room_id": room.id,
        "room": room,
        "floor_label": room.floor_label,
        "hour": hour,
        "phone_fix": fix,
        "clues": clues,
        "persona": resolve_persona(persona).get("name", "custom"),
    }

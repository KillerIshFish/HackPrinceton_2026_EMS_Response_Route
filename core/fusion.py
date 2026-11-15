"""Bayesian fusion of prior, phone fix and caller statements.

    p(r | E) ∝ pi(r) * L_phone(r) * Π_i F_i(r) * S(r)

Two kinds of statement:

* **Descriptive** (a room type, a view, a landmark, a sound, a smell, a route, any denial, a
  grounded statement with ``default_match > 0``):  ``F_i(r) = [eps + (1 - eps) * m_i(r)]^{c_i}``
  with eps = 0.1. c_i in [0, 1] is the confidence; it tempers the factor (c = 0 -> ignored).
  One such statement moves the odds between two rooms by at most 10 to 1.
* **Specific** (``names_rooms``): the caller said where they are by a room, unit or suite
  number, a room or tenant name, or a floor ("third floor", "top floor", "ground floor",
  "basement"). In data terms: a grounded statement that places the caller with
  ``default_match == 0``, or a positive ``floor`` vocabulary clue. Its ``confidence`` is the
  **stated share** rho: the share of the whole probability (rooms and the "not in model"
  bucket) the named rooms hold when the statement stands alone, whatever the prior says
  and however many floors the building has (Jeffrey's rule). rho comes from
  ``stated_share(certainty, specificity)``: 0.40 for a hesitant, vague statement up to 0.95
  for a plain, exact one (``configs/statements.yaml``, label judgement). The rest, 1 - rho,
  is split over the other rooms by the prior times the soft factor of the statement's match
  (so the floors next to a stated floor keep more than the far ones).

How specific statements combine (``S(r)``, ``_statement_factors``). They are applied one
after another, the one naming the fewest rooms first (a room before its suite before its
floor), so the result does not depend on the order they were said in:

* **Agreement reinforces.** A statement whose rooms already hold share c (prior share q)
  moves them to ``1 - (1 - rho) * (1 - e)`` with ``e = (c - q) / (1 - q)``, the part of the
  way from the prior to 1 that the statements before it already carried them; the rooms
  inside keep their proportions. Alone, e = 0 and the share is exactly rho. The same place
  said twice ends above either statement (0.95 twice: 0.9975), and a room named on a floor
  the caller also stated lifts both; never above ``REINFORCED_MAX`` (0.999).
* **A correction replaces.** Evidence whose ``status`` is not ``"active"`` (superseded by a
  correction, rejected by the dispatcher) is skipped entirely, so the reject button undoes
  any statement.
* **A conflict without a correction: the later statement wins.** Two specific statements
  with no room in common cannot both be right. The one from the later turn is used and the
  earlier one is set aside (within one turn: the one with the higher share, then the one
  naming fewer rooms). ``fuse_detail`` reports the pair so the console can show a notice;
  rejecting the later statement brings the earlier one back.
* **Caller against phone.** When a specific statement names no room inside the floors the
  phone's height fix allows (``PHONE_BAND``: within the fix's 80 % bound, +/- 3 m by
  default), the caller's statement leads. The posterior is a mix of two readings:
  ``(1 - k)`` x (the caller is right: the height fix is ignored) and ``k`` x (the phone is
  right: the statements it contradicts are ignored, and an earlier statement they had set
  aside counts again), with k =
  ``phone_share_when_caller_disagrees`` (0.25), lowered when the statement is weak so that
  the stated rooms still hold more than the phone's floors. The phone's share is what sends
  a second team to the phone's floor.

Safety: no factor is ever zero (eps = 0.1; rho is capped at ``RHO_MAX`` = 0.95 and the
reinforced share at 0.999), so one wrong or misheard statement can never zero a room.

* "Not in model" bucket: it has no geometry, so it is scored at the base rate -
  L_phone = prior-weighted mean of the room likelihoods, F_i = the factor of the
  prior-weighted mean match. A specific statement treats it as one more place the caller did
  not name.
* Statements may be vocabulary ``Clue`` objects or ``Evidence`` items (``core.evidence``).

Output is normalised: sum(room_probs) + p_not_in_model == 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence, Union

import yaml

from core.clue_likelihood import clue_match
from core.evidence import EvidenceMatch, basement_match, evidence_match, is_basement_clue
from core.phone_likelihood import phone_likelihood
from core.prior import compute_prior
from core.schema import Banner, Building, Clue, Evidence, PhoneFix

# Noise floor of a descriptive statement (judgement; kept by docs/BUILD_PLAN.md step C1): no factor
# drops below 0.1, so one wrong statement shifts the odds between two rooms by at most 1/eps = 10 to 1.
EPS = 0.1
STATEMENTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "statements.yaml"
# A room counts as named by a statement when its match value is at least this (1.0 less float slack).
_NAMED_AT = 1.0 - 1e-9
# Rationale prefix of the interpreter's overall conclusion (parser.agent record_conclusion).
CONCLUSION_TAG = "Conclusion"
# A floor is inside the phone's height fix when its likelihood is at least this share of the
# best floor's: exp(-0.5 * 1.28^2), the 80 % bound the fix is quoted with (+/- 3 m).
# 1.28 is the two-sided 80 % quantile of the standard normal (z = 1.2816), as in schema.Z_SIGMA_M.
PHONE_BAND = math.exp(-0.5 * 1.28 ** 2)
# With the phone's share, the stated rooms still hold at least this multiple of the mass on
# the phone's floors (when the statement alone is strong enough for that).
# Tuned heuristic: 10 % headroom so the stated place visibly leads rather than tying the phone's floor.
_LEAD_MARGIN = 1.1


# ---------------------------------------------------------------------------
# Settings (configs/statements.yaml, every number labelled judgement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementSettings:
    """Weights of a specific caller statement (see ``configs/statements.yaml``)."""

    # Fallbacks equal the values in configs/statements.yaml (all labelled judgement there), so a
    # missing or unreadable file gives the same weights.
    share_low: float = 0.40  # stated share of a hesitant, vague statement
    share_high: float = 0.95  # stated share of a plain, exact statement; cap on any one statement
    missing_rating: float = 0.5  # certainty / specificity used when a rating is missing
    reinforced_max: float = 0.999  # cap on the share agreeing statements reach together
    phone_share: float = 0.25  # share the phone's height fix keeps when the caller contradicts it
    labels: Mapping[str, str] = field(default_factory=dict)  # setting name -> label ("judgement")


def _entry(node: Any, default: float) -> tuple[float, str]:
    if isinstance(node, dict) and "value" in node:
        return float(node["value"]), str(node.get("label", ""))
    if isinstance(node, (int, float)):
        return float(node), ""
    return default, ""


@lru_cache(maxsize=4)
def _load_settings(path: str) -> StatementSettings:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError:
        data = {}
    d = StatementSettings()
    share = data.get("stated_share") or {}
    low, l_low = _entry(share.get("low"), d.share_low)
    high, l_high = _entry(share.get("high"), d.share_high)
    missing, l_missing = _entry(share.get("missing_rating"), d.missing_rating)
    reinforced, l_reinforced = _entry(data.get("reinforced_share_max"), d.reinforced_max)
    phone, l_phone = _entry(data.get("phone_share_when_caller_disagrees"), d.phone_share)
    # Safety clamps on edited configs: shares stay below 1 so a room a statement does not name never
    # reaches zero, and the phone keeps at most 0.9 so the caller's reading always keeps some mass.
    high = min(0.999, max(0.0, high))
    low = min(high, max(0.0, low))
    return StatementSettings(
        share_low=low, share_high=high, missing_rating=min(1.0, max(0.0, missing)),
        reinforced_max=min(0.999, max(high, reinforced)), phone_share=min(0.9, max(0.0, phone)),
        labels={"stated_share.low": l_low, "stated_share.high": l_high, "stated_share.missing_rating": l_missing,
                "reinforced_share_max": l_reinforced, "phone_share_when_caller_disagrees": l_phone},
    )


def statement_settings(path: str | Path | None = None) -> StatementSettings:
    """The statement weights from ``configs/statements.yaml`` (cached)."""
    return _load_settings(str(path or STATEMENTS_PATH))


# A single specific statement never counts for more than this (the top of the stated-share
# scale), so the rooms it does not name keep a factor above zero.
RHO_MAX = statement_settings().share_high
# The share agreeing statements can reach together.
REINFORCED_MAX = statement_settings().reinforced_max
# Kept for older callers: grounded evidence from the model reader used to be capped at 0.85.
# Its share now comes from the model's rating of the caller's words (``stated_share``), so
# the only cap left is ``RHO_MAX``.
GROUNDED_AGENT_MAX_CONFIDENCE = RHO_MAX


def _clamp01(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return min(1.0, max(0.0, x))


def stated_share(certainty: Any, specificity: Any, settings: StatementSettings | None = None) -> float:
    """Share of the probability a specific statement gives the place it names, standing alone.

    ``low + (high - low) * certainty * specificity`` with both ratings clamped to [0, 1]; a
    rating that is missing or not a number counts as ``missing_rating`` (0.5). 0.40 for
    "I think maybe the third floor?" up to 0.95 for "I'm in room 302".
    """
    s = settings or statement_settings()
    c = _clamp01(certainty, s.missing_rating)
    k = _clamp01(specificity, s.missing_rating)
    return round(s.share_low + (s.share_high - s.share_low) * c * k, 3)


# ---------------------------------------------------------------------------
# Statement terms
# ---------------------------------------------------------------------------


class Statement(NamedTuple):
    """One statement for ``fuse``. A plain ``(match, confidence)`` or ``(match, confidence,
    names)`` tuple is accepted too.

    * ``match``       room id -> match value in [0, 1]
    * ``confidence``  descriptive: the exponent c; specific (``names``): the stated share rho
    * ``names``       True: a specific statement naming the rooms whose match is 1.0
    * ``turn``        transcript turn it came from (None: its position in the list counts)
    * ``key``         evidence id, for notices
    * ``final``       True: the interpreter's overall conclusion (``is_conclusion``). It already
                      weighs every other statement, so it is applied last, to the whole
                      posterior: its rooms end at its share (raised to it, never lowered) and
                      everything else keeps its proportions.
    """

    match: Mapping[str, float]
    confidence: float
    names: bool = False
    turn: Optional[int] = None
    key: str = ""
    final: bool = False


StatementTerm = Union[Statement, tuple[Mapping[str, float], float], tuple[Mapping[str, float], float, bool]]


def _as_statement(term: StatementTerm) -> Statement:
    if isinstance(term, Statement):
        return term
    return Statement(term[0], float(term[1]), bool(term[2]) if len(term) > 2 else False)


def clue_factor(m: float, confidence: float, eps: float = EPS) -> float:
    """[eps + (1 - eps) * m]^c with m and c clamped to [0, 1]."""
    m = min(1.0, max(0.0, m))
    c = min(1.0, max(0.0, confidence))
    return (eps + (1.0 - eps) * m) ** c


def naming_floor(confidence: float, prior_share: float) -> float:
    """Factor of a place a specific statement does not name, relative to the places it names,
    when the statement stands alone and the rest is split by the prior.

    ``prior_share`` is the prior share q of the named rooms (of rooms and the "not in model"
    bucket together). With rho = min(confidence, RHO_MAX)::

        lambda = q * (1 - rho) / (rho * (1 - q))

    so the named rooms end at ``q / (q + (1 - q) * lambda) = rho`` exactly. 1.0 when the
    statement is ignored (confidence 0) or the named rooms already hold rho or more (a
    statement never lowers the rooms it names); always above zero because rho <= RHO_MAX.
    """
    rho = min(RHO_MAX, max(0.0, confidence))
    q = min(1.0, max(0.0, prior_share))
    if rho <= 0.0 or q >= rho:
        return 1.0
    return q * (1.0 - rho) / (rho * (1.0 - q))


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float], default: float = 1.0) -> float:
    total = sum(weights.values())
    if total <= 0.0:
        return default
    return sum(w * values.get(rid, default) for rid, w in weights.items()) / total


@dataclass(frozen=True)
class _Named:
    """A specific statement prepared for ``_statement_factors``."""

    index: int  # position in the statement list given to ``fuse``
    rooms: frozenset
    rho: float
    match: Mapping[str, float]
    time: int  # turn (or list position) for "the later one wins"


def _resolve_conflicts(named: Sequence[_Named]) -> tuple[list[_Named], list[tuple[int, int]]]:
    """``(kept, set_aside)``: of two specific statements with no room in common the later one
    is kept (same turn: higher share, then fewer rooms, then the later one). ``set_aside`` is
    ``(index set aside, index that replaced it)``; ``kept`` is in the order the statements
    are applied: the one naming the fewest rooms first (the room before the floor it is on)."""
    order = sorted(named, key=lambda n: (-n.time, -n.rho, len(n.rooms), -n.index))
    kept: list[_Named] = []
    aside: list[tuple[int, int]] = []
    for n in order:
        winner = next((k for k in kept if n.rooms.isdisjoint(k.rooms)), None)
        if winner is None:
            kept.append(n)
        else:
            aside.append((n.index, winner.index))
    kept.sort(key=lambda n: (len(n.rooms), n.index))
    return kept, sorted(aside)


def _statement_factors(
    shares: Mapping[str, float], nim_share: float, named: Sequence[_Named], eps: float, reinforced_max: float
) -> tuple[dict[str, float], float]:
    """Per-room factor S(r) (and the factor of the "not in model" bucket) of the specific
    statements, applied in the order given to the normalised prior ``shares`` + ``nim_share``.

    See the module docstring (agreement reinforces). Every factor is above zero.
    """
    cur = dict(shares)
    cur_nim = nim_share
    for n in named:
        c = sum(cur[r] for r in n.rooms)
        q = sum(shares[r] for r in n.rooms)
        if c <= 0.0 or q <= 0.0 or q >= 1.0 or n.rho <= 0.0:
            continue
        carried = max(0.0, (c - q) / (1.0 - q))
        target = min(reinforced_max, 1.0 - (1.0 - n.rho) * (1.0 - carried))
        if target <= c:
            continue  # the named rooms already hold that much: a statement never lowers them
        soft = {r: eps + (1.0 - eps) * min(1.0, max(0.0, n.match.get(r, 1.0))) for r in cur if r not in n.rooms}
        out_mass = sum(cur[r] for r in soft)
        out_weight = sum(cur[r] * g for r, g in soft.items())
        nim_soft = (out_weight / out_mass) if out_mass > 0.0 else 1.0
        weight = out_weight + cur_nim * nim_soft
        if weight <= 0.0:
            continue
        k_in, k_out = target / c, (1.0 - target) / weight
        for r in cur:
            cur[r] *= k_in if r in n.rooms else k_out * soft[r]
        cur_nim *= k_out * nim_soft
    factors = {r: (cur[r] / shares[r] if shares[r] > 0.0 else 1.0) for r in cur}
    return factors, (cur_nim / nim_share if nim_share > 0.0 else 1.0)


@dataclass
class Fused:
    """Result of ``fuse_detail``.

    * ``room_probs``, ``p_not_in_model``  the posterior (they sum to 1)
    * ``set_aside``       ``(i, j)``: specific statement ``i`` conflicts with the later
                          statement ``j`` and was left out (indices into the statement list)
    * ``phone_conflicts`` indices of specific statements naming no room on the floors the
                          phone's height fix allows
    * ``phone_share``     share of the posterior given to the phone's reading (0.0 when no
                          statement contradicts the height fix)
    * ``phone_band``      room ids on the floors the phone's height fix allows (empty when
                          there is no height fix)
    * ``notices``         console notices (``fuse_evidence`` fills them)
    """

    room_probs: dict[str, float]
    p_not_in_model: float
    set_aside: list[tuple[int, int]] = field(default_factory=list)
    phone_conflicts: list[int] = field(default_factory=list)
    phone_share: float = 0.0
    phone_band: frozenset = frozenset()
    notices: list[Banner] = field(default_factory=list)


def _normalised(scores: Mapping[str, float], nim: float, prior: Mapping[str, float], p_nim_prior: float
                ) -> tuple[dict[str, float], float]:
    total = sum(scores.values()) + nim
    if total <= 0.0:  # all evidence impossible: fall back to the prior
        scores, nim = {rid: float(p) for rid, p in prior.items()}, float(p_nim_prior)
        total = sum(scores.values()) + nim
        if total <= 0.0:
            return {rid: 0.0 for rid in scores}, 1.0
    return {rid: s / total for rid, s in scores.items()}, nim / total


def _apply_finals(probs: dict[str, float], nim: float, finals: Sequence[Statement], rho_max: float
                  ) -> tuple[dict[str, float], float]:
    """Apply the interpreter's overall conclusion(s) last (``Statement.final``), latest turn last.

    The named rooms end at the conclusion's share rho when they hold less (never lowered);
    every other room and the "not in model" bucket are scaled together, so the phone, the prior
    and the other statements still order them. No room reaches zero (rho <= ``rho_max``).
    """
    for t in sorted(finals, key=lambda f: (f.turn if f.turn is not None else -1)):
        rooms = {rid for rid in probs if t.match.get(rid, 0.0) >= _NAMED_AT}
        c = sum(probs[r] for r in rooms)
        rho = min(rho_max, max(0.0, t.confidence))
        if not rooms or c <= 0.0 or c >= rho or c >= 1.0:
            continue
        k_in, k_out = rho / c, (1.0 - rho) / (1.0 - c)
        probs = {rid: p * (k_in if rid in rooms else k_out) for rid, p in probs.items()}
        nim *= k_out
    return probs, nim


def fuse_detail(
    prior: Mapping[str, float],
    p_nim_prior: float,
    phone_lik: Mapping[str, float] | None,
    clue_matches: Sequence[StatementTerm],
    eps: float = EPS,
    *,
    phone_height: Mapping[str, float] | None = None,
    settings: StatementSettings | None = None,
) -> Fused:
    """``fuse`` with what it decided along the way (see ``Fused``)."""
    cfg = settings or statement_settings()
    rho_max = cfg.share_high
    all_terms = [_as_statement(t) for t in clue_matches]
    terms = [t for t in all_terms if not t.final]
    finals = [t for t in all_terms if t.final]
    pri = {rid: float(p) for rid, p in prior.items()}
    nim0 = float(p_nim_prior)
    prior_rooms = sum(pri.values())
    prior_total = prior_rooms + nim0

    # ---- descriptive statements: one soft factor each -------------------------
    soft = {rid: 1.0 for rid in pri}
    soft_nim = 1.0
    named: list[_Named] = []
    use_turns = all(t.turn is not None for t in terms if t.names)
    for i, t in enumerate(terms):
        if t.names and prior_rooms > 0.0:
            rooms = frozenset(rid for rid in pri if t.match.get(rid, 1.0) >= _NAMED_AT)
            share = sum(pri[rid] for rid in rooms) / prior_rooms
            if 0.0 < share < 1.0:
                rho = min(rho_max, max(0.0, t.confidence))
                named.append(_Named(i, rooms, rho, t.match, int(t.turn) if use_turns else i))
                continue
            # it names no room with prior mass, or every room: fall through to the soft factor
        soft_nim *= clue_factor(_weighted_mean(t.match, pri), t.confidence, eps)
        for rid in soft:
            soft[rid] *= clue_factor(t.match.get(rid, 1.0), t.confidence, eps)

    # ---- specific statements ----------------------------------------------------
    kept, set_aside = _resolve_conflicts(named)
    shares = {rid: (p / prior_total if prior_total > 0.0 else 0.0) for rid, p in pri.items()}
    nim_share = nim0 / prior_total if prior_total > 0.0 else 0.0

    def posterior(statements: Sequence[_Named], phone: Mapping[str, float] | None) -> tuple[dict[str, float], float]:
        stated, stated_nim = _statement_factors(shares, nim_share, statements, eps, cfg.reinforced_max)
        nim = nim0 * soft_nim * stated_nim
        scores = {rid: pri[rid] * soft[rid] * stated[rid] for rid in pri}
        if phone is not None:
            nim *= _weighted_mean(phone, pri)
            for rid in scores:
                scores[rid] *= phone.get(rid, 1.0)
        return _normalised(scores, nim, pri, nim0)

    # ---- caller against phone -----------------------------------------------------
    band: frozenset = frozenset()
    conflicts: list[_Named] = []
    if phone_lik is not None and phone_height is not None and pri:
        best = max(phone_height.get(rid, 1.0) for rid in pri)
        band = frozenset(rid for rid in pri if phone_height.get(rid, 1.0) >= PHONE_BAND * best)
        if len(band) < len(pri):
            conflicts = [n for n in kept if n.rooms.isdisjoint(band)]
        else:
            band = frozenset()  # the height fix does not tell the floors apart
    if not conflicts:
        probs, nim = posterior(kept, phone_lik)
        probs, nim = _apply_finals(probs, nim, finals, rho_max)
        return Fused(probs, nim, set_aside, [], 0.0, band)

    # Reading 1, the caller is right: the height fix is left out (any horizontal fix stays).
    # Dividing out the height part; 1e-300 only guards the division (phone likelihoods are >= 1e-12).
    level = {rid: phone_lik.get(rid, 1.0) / max(phone_height.get(rid, 1.0), 1e-300) for rid in pri}
    caller, caller_nim = posterior(kept, level)
    # Reading 2, the phone is right: every statement it contradicts is left out, so an
    # earlier statement that one of them had set aside counts again.
    agreeing, _ = _resolve_conflicts([n for n in named if not n.rooms.isdisjoint(band)])
    device, device_nim = posterior(agreeing, phone_lik)
    k = cfg.phone_share
    for n in conflicts:
        a, b = sum(caller[r] for r in n.rooms), sum(caller[r] for r in band)
        c, d = sum(device[r] for r in n.rooms), sum(device[r] for r in band)
        lead = a - _LEAD_MARGIN * b
        gap = _LEAD_MARGIN * d - c
        if lead > 0.0 and gap > 0.0:  # otherwise no share could make (or break) the lead
            k = min(k, lead / (lead + gap))
    probs = {rid: (1.0 - k) * caller[rid] + k * device[rid] for rid in pri}
    nim = (1.0 - k) * caller_nim + k * device_nim
    probs, nim = _apply_finals(probs, nim, finals, rho_max)
    return Fused(probs, nim, set_aside, sorted(n.index for n in conflicts), k, band)


def fuse(
    prior: Mapping[str, float],
    p_nim_prior: float,
    phone_lik: Mapping[str, float] | None,
    clue_matches: Sequence[StatementTerm],
    eps: float = EPS,
    *,
    phone_height: Mapping[str, float] | None = None,
    settings: StatementSettings | None = None,
) -> tuple[dict[str, float], float]:
    """Posterior (room_probs, p_not_in_model) from prior x phone likelihood x statements.

    ``clue_matches`` is a list of ``Statement`` terms, or ``(match vector m_i over rooms,
    confidence c_i)`` / ``(m_i, c_i, names)`` tuples. With ``names`` true the statement is a
    specific one naming the rooms whose match is 1.0, and its confidence is the stated
    share (module docstring); otherwise it is fused with ``clue_factor``. Rooms missing from
    a likelihood / match vector are treated as uninformative (1.0).

    ``phone_height`` is the height part of the phone likelihood on its own
    (``phone_height_likelihood``). With it, a specific statement that contradicts the height
    fix leads and the phone keeps a share; without it the phone likelihood is a plain factor.
    """
    out = fuse_detail(prior, p_nim_prior, phone_lik, clue_matches, eps, phone_height=phone_height, settings=settings)
    return out.room_probs, out.p_not_in_model


def phone_height_likelihood(building: Building, fix: PhoneFix | None) -> dict[str, float] | None:
    """The height (z) part of ``phone_likelihood`` on its own, or None without a height fix."""
    if fix is None or fix.z_m is None or not fix.z_sigma_m or fix.z_sigma_m <= 0:
        return None
    return phone_likelihood(building, PhoneFix(z_m=fix.z_m, z_sigma_m=fix.z_sigma_m))


def evidence_confidence(evidence: Evidence) -> float:
    """Confidence used in fusion: ``evidence.confidence`` (a specific statement's stated
    share is capped at ``RHO_MAX`` inside ``fuse``)."""
    return float(evidence.confidence)


def _names_a_floor(clue: Clue | None) -> bool:
    return clue is not None and clue.type == "floor" and bool(clue.polarity)


def names_rooms(evidence: Evidence, match: EvidenceMatch | None = None) -> bool:
    """True when ``evidence`` is a specific statement of where the caller is (module docstring).

    * grounded: it places the caller (not a denial) with ``default_match == 0``: the rooms
      its selector picks are where the caller said they are. A selector that matched no room
      names nothing. ``default_match > 0`` keeps the descriptive factor (a softer statement).
    * vocab: a positive ``floor`` clue ("third floor", "top floor", "ground floor",
      "basement"). A denial ("not on four") and every other clue type stay descriptive.
    """
    if evidence.kind == "vocab":
        return _names_a_floor(evidence.clue)
    if evidence.kind != "grounded" or evidence.negate or evidence.default_match > 0.0:
        return False
    return not (match is not None and match.unmatched)


def is_conclusion(evidence: Evidence) -> bool:
    """True for the interpreter's overall conclusion (``parser.agent`` ``record_conclusion``)."""
    return evidence.kind == "grounded" and (evidence.rationale or "").startswith(CONCLUSION_TAG + ":")


def statement_term(evidence: Evidence, match: EvidenceMatch) -> Statement:
    """The ``fuse`` term for one Evidence item, given its ``evidence_match`` result."""
    final = is_conclusion(evidence) and names_rooms(evidence, match)
    return Statement(match.match, evidence_confidence(evidence), names_rooms(evidence, match) and not final,
                     int(evidence.turn), evidence.id, final)


def statement_matches(
    building: Building,
    statements: Iterable[Union[Clue, Evidence]],
    vocab: dict | None = None,
) -> list[Statement]:
    """One ``fuse`` term per statement.

    ``Clue`` -> its ``clue_match`` and confidence (a positive floor clue is a specific
    statement). ``Evidence`` -> ``statement_term`` (match from ``evidence_match``, its
    confidence, and whether it is specific); items whose status is not ``"active"`` are
    left out.
    """
    out: list[Statement] = []
    for item in statements:
        if isinstance(item, Evidence):
            if item.status != "active":
                continue
            out.append(statement_term(item, evidence_match(building, item, vocab)))
        else:
            match = (basement_match(building, bool(item.polarity)).match if is_basement_clue(item)
                     else clue_match(building, item, vocab))
            out.append(Statement(match, item.confidence, _names_a_floor(item)))
    return out


def posterior_probs(
    building: Building,
    hour: int = 14,
    fix: PhoneFix | None = None,
    clues: Iterable[Union[Clue, Evidence]] = (),
    priors_cfg: dict | None = None,
    vocab: dict | None = None,
) -> tuple[dict[str, float], float]:
    """Full pipeline: prior -> phone likelihood -> statement matches -> fuse.

    ``clues`` may mix vocabulary ``Clue`` objects and ``Evidence`` items; inactive
    evidence is skipped (see ``statement_matches``). ``hour`` defaults to 14:00, inside the
    ``business`` band of configs/priors.yaml (the daytime office prior).
    """
    prior, p_nim = compute_prior(building, hour=hour, priors_cfg=priors_cfg)
    return fuse(prior, p_nim, phone_likelihood(building, fix), statement_matches(building, clues, vocab),
                phone_height=phone_height_likelihood(building, fix))


# ---------------------------------------------------------------------------
# Evidence in, posterior and console notices out (api.session)
# ---------------------------------------------------------------------------


def _floor_labels(building: Building, room_ids: Iterable[str]) -> list[str]:
    """Labels of the floors holding ``room_ids``, lowest floor first."""
    wanted = set(room_ids)
    floors: dict[int, str] = {}
    for r in building.rooms:
        if r.id in wanted:
            floors.setdefault(r.floor, r.floor_label)
    return [floors[f] for f in sorted(floors)]


def _floors_text(labels: Sequence[str]) -> str:
    if not labels:
        return "no floor"
    if len(labels) == 1:
        return f"floor {labels[0]}"
    return "floors " + ", ".join(labels[:-1]) + " and " + labels[-1]


def _place_text(building: Building, evidence: Evidence, match: EvidenceMatch) -> str:
    """What a specific statement names, in a few words: "floor 3", "Office 302 (floor 3)"."""
    rooms = [rid for rid, m in match.match.items() if m >= _NAMED_AT]
    labels = _floor_labels(building, rooms)
    if evidence.kind == "vocab":
        return _floors_text(labels)
    names = {r.id: r.name for r in building.rooms}
    shown = [names[rid] for rid in rooms[:2] if rid in names]
    more = len(rooms) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "") + f" ({_floors_text(labels)})"


def _quote(evidence: Evidence) -> str:
    span = " ".join((evidence.span or "").split())
    return f'"{span}"' if span else "no words recorded"


def statement_notices(building: Building, fix: PhoneFix | None, active: Sequence[Evidence],
                      matches: Sequence[EvidenceMatch], fused: Fused) -> list[Banner]:
    """Console notices for what ``fuse_detail`` decided about ``active`` (the evidence items
    behind its statement list, in the same order)."""
    out: list[Banner] = []
    for old_i, new_i in fused.set_aside:
        old, new = active[old_i], active[new_i]
        old_place, new_place = _place_text(building, old, matches[old_i]), _place_text(building, new, matches[new_i])
        if old.turn == new.turn:
            text = (f"Caller gave two different places in one turn: {new_place} ({_quote(new)}) and {old_place} "
                    f"({_quote(old)}). The first is used (said with more certainty, or the more specific of the "
                    f"two); rejecting it brings the other back.")
        else:
            text = (f"Caller gave two different places: {old_place} ({_quote(old)}, turn {old.turn}), then "
                    f"{new_place} ({_quote(new)}, turn {new.turn}). The later statement is used; rejecting it "
                    f"brings the earlier one back.")
        out.append(Banner(level="warning", text=text,
                          source=f"caller statements {old.id} and {new.id}, no correction heard"))
    if fused.phone_conflicts and fix is not None and fix.z_m is not None:
        band_floors = _floors_text(_floor_labels(building, fused.phone_band))
        pct = int(round(100 * fused.phone_share))
        for i in fused.phone_conflicts:
            ev = active[i]
            out.append(Banner(
                level="warning",
                text=(f"Caller says {_place_text(building, ev, matches[i])} ({_quote(ev)}); the phone's height fix "
                      f"says {band_floors} ({fix.z_m:.1f} m). The caller's statement leads; the phone's fix keeps "
                      f"about {pct}% of the probability."),
                source=f"caller statement {ev.id} against the phone's height fix",
            ))
    return out


def fuse_evidence(
    building: Building,
    prior: Mapping[str, float],
    p_nim_prior: float,
    fix: PhoneFix | None,
    evidence: Sequence[Evidence],
    match_of: Optional[Callable[[Evidence], EvidenceMatch]] = None,
    vocab: dict | None = None,
) -> Fused:
    """Posterior and console notices for a call's evidence (``api.session``).

    Same maths as ``posterior_probs``: items whose status is not ``"active"`` are skipped,
    the rest go through ``statement_term`` and ``fuse_detail`` with the phone fix.
    ``match_of`` supplies cached ``evidence_match`` results.
    """
    active = [e for e in evidence if e.status == "active"]
    matches = [match_of(e) if match_of is not None else evidence_match(building, e, vocab) for e in active]
    terms = [statement_term(e, m) for e, m in zip(active, matches)]
    fused = fuse_detail(prior, p_nim_prior, phone_likelihood(building, fix), terms,
                        phone_height=phone_height_likelihood(building, fix))
    fused.notices = statement_notices(building, fix, active, matches, fused)
    return fused

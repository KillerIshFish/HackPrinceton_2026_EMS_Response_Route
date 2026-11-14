"""Bayesian fusion of prior, phone fix and caller clues.

    p(r | E) ∝ pi(r) * L_phone(r) * Π_i F_i(r)

Two kinds of statement factor:

* **Descriptive** (every vocabulary clue, a denial, a grounded statement with
  ``default_match > 0``):  ``F_i(r) = [eps + (1 - eps) * m_i(r)]^{c_i}`` with eps = 0.1.
  c_i in [0, 1] is the confidence; it tempers the factor (c = 0 -> ignored).
* **Naming** (a grounded statement that names rooms outright: ``default_match == 0`` and not
  a denial, e.g. "I'm in 302", "suite 860", "Dr. Patel's office"):  ``F_i(r) = 1`` for the
  named rooms and ``lambda_i`` for every other room, with

      lambda_i = (1 - rho) * q / (rho + (1 - rho) * q),   rho = min(c_i, RHO_MAX)

  where q is the prior share of the named rooms. This is Jeffrey's rule: taken alone, the
  statement moves the named rooms to probability ``rho + (1 - rho) * q`` of the modelled
  rooms, whatever their prior was (a caller who names the vacant suite is in the vacant
  suite), and splits the rest by the prior. The fixed eps = 0.1 could not do that: it caps
  the odds one statement can move at 10:1, so in a 100-room building "I'm in 302" left
  room 302 near 10 %.

Safety, both kinds: no factor is ever zero (eps = 0.1; rho is capped at ``RHO_MAX`` = 0.95, so
``lambda_i > 0``), so one wrong or misheard statement can never zero a room and later
evidence can overturn it. Evidence whose ``status`` is not ``"active"`` (rejected by the
dispatcher, superseded by a correction) is skipped entirely, so the reject button undoes
any statement.

* "Not in model" bucket: it has no geometry, so it is scored at the base rate -
  L_phone = prior-weighted mean of the room likelihoods, F_i = prior-weighted mean of the
  room factors (for descriptive statements: the factor of the prior-weighted mean match).
* Statements may be vocabulary ``Clue`` objects or ``Evidence`` items (``core.evidence``).
  Grounded evidence from the agent has its confidence capped at
  ``GROUNDED_AGENT_MAX_CONFIDENCE`` (``evidence_confidence``).

Output is normalised: sum(room_probs) + p_not_in_model == 1.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Union

from core.clue_likelihood import clue_match
from core.evidence import EvidenceMatch, evidence_match
from core.phone_likelihood import phone_likelihood
from core.prior import compute_prior
from core.schema import Building, Clue, Evidence, PhoneFix

EPS = 0.1
# An agent's grounded reading of a statement never counts for more than this.
GROUNDED_AGENT_MAX_CONFIDENCE = 0.85
# A naming statement never counts for more than this, so the rooms it does not name keep a
# factor above zero.
RHO_MAX = 0.95
# A room counts as named by a statement when its match value is at least this.
_NAMED_AT = 1.0 - 1e-9

# One statement for ``fuse``: (match vector, confidence) or (match vector, confidence, names).
StatementTerm = Union[tuple[Mapping[str, float], float], tuple[Mapping[str, float], float, bool]]


def clue_factor(m: float, confidence: float, eps: float = EPS) -> float:
    """[eps + (1 - eps) * m]^c with m and c clamped to [0, 1]."""
    m = min(1.0, max(0.0, m))
    c = min(1.0, max(0.0, confidence))
    return (eps + (1.0 - eps) * m) ** c


def naming_floor(confidence: float, prior_share: float) -> float:
    """Factor of a room that a naming statement does not name (see the module docstring).

    ``prior_share`` is the prior share q of the named rooms among the modelled rooms. The
    result is 1.0 at confidence 0 (statement ignored), falls as confidence rises, and is
    always above zero because confidence is capped at ``RHO_MAX``.
    """
    rho = min(RHO_MAX, max(0.0, confidence))
    q = min(1.0, max(0.0, prior_share))
    if rho <= 0.0:
        return 1.0
    return (1.0 - rho) * q / (rho + (1.0 - rho) * q)


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float], default: float = 1.0) -> float:
    total = sum(weights.values())
    if total <= 0.0:
        return default
    return sum(w * values.get(rid, default) for rid, w in weights.items()) / total


def fuse(
    prior: Mapping[str, float],
    p_nim_prior: float,
    phone_lik: Mapping[str, float] | None,
    clue_matches: Sequence[StatementTerm],
    eps: float = EPS,
) -> tuple[dict[str, float], float]:
    """Posterior (room_probs, p_not_in_model) from prior x phone likelihood x statement factors.

    ``clue_matches`` is a list of ``(match vector m_i over rooms, confidence c_i)`` or
    ``(m_i, c_i, names)``. With ``names`` true the statement names the rooms whose match is
    1.0 and is fused with ``naming_floor``; otherwise with ``clue_factor``.
    Rooms missing from a likelihood / match vector are treated as uninformative (1.0).
    """
    scores = {rid: float(p) for rid, p in prior.items()}
    nim = float(p_nim_prior)
    prior_total = sum(float(p) for p in prior.values())

    if phone_lik is not None:
        nim *= _weighted_mean(phone_lik, prior)
        for rid in scores:
            scores[rid] *= phone_lik.get(rid, 1.0)

    for term in clue_matches:
        match, confidence = term[0], term[1]
        names = bool(term[2]) if len(term) > 2 else False
        if names and prior_total > 0.0:
            named = {rid for rid in scores if match.get(rid, 1.0) >= _NAMED_AT}
            share = sum(float(prior[rid]) for rid in named) / prior_total
            if 0.0 < share < 1.0:
                lam = naming_floor(confidence, share)
                nim *= share + (1.0 - share) * lam  # prior-weighted mean factor
                for rid in scores:
                    if rid not in named:
                        scores[rid] *= lam
                continue
            # it names no room with prior mass, or every room: fall through to the plain factor
        nim *= clue_factor(_weighted_mean(match, prior), confidence, eps)
        for rid in scores:
            scores[rid] *= clue_factor(match.get(rid, 1.0), confidence, eps)

    total = sum(scores.values()) + nim
    if total <= 0.0:  # all evidence impossible: fall back to the prior
        scores, nim = {rid: float(p) for rid, p in prior.items()}, float(p_nim_prior)
        total = sum(scores.values()) + nim
        if total <= 0.0:
            return {rid: 0.0 for rid in scores}, 1.0
    return {rid: s / total for rid, s in scores.items()}, nim / total


def evidence_confidence(evidence: Evidence) -> float:
    """Confidence used in fusion: ``evidence.confidence``, capped at 0.85 for grounded evidence
    recorded by the model interpreter (source "agent" or "llm")."""
    c = float(evidence.confidence)
    if evidence.kind == "grounded" and evidence.source in ("agent", "llm"):
        c = min(c, GROUNDED_AGENT_MAX_CONFIDENCE)
    return c


def names_rooms(evidence: Evidence, match: EvidenceMatch | None = None) -> bool:
    """True when ``evidence`` names rooms outright and is fused with ``naming_floor``.

    That is a grounded statement that places the caller (not a denial) with
    ``default_match == 0``: the rooms its selector picks are where the caller said they are,
    and nothing softer is claimed for the others. A selector that matched no room names
    nothing. ``default_match > 0`` keeps the descriptive factor (a softer statement).
    """
    if evidence.kind != "grounded" or evidence.negate or evidence.default_match > 0.0:
        return False
    return not (match is not None and match.unmatched)


def statement_term(evidence: Evidence, match: EvidenceMatch) -> StatementTerm:
    """The ``fuse`` term for one Evidence item, given its ``evidence_match`` result."""
    return match.match, evidence_confidence(evidence), names_rooms(evidence, match)


def statement_matches(
    building: Building,
    statements: Iterable[Union[Clue, Evidence]],
    vocab: dict | None = None,
) -> list[StatementTerm]:
    """One ``fuse`` term per statement.

    ``Clue`` -> ``(clue_match, confidence)``. ``Evidence`` -> ``statement_term`` (match from
    ``evidence_match``, ``evidence_confidence``, and whether it names rooms); items whose
    status is not ``"active"`` are left out.
    """
    out: list[StatementTerm] = []
    for item in statements:
        if isinstance(item, Evidence):
            if item.status != "active":
                continue
            out.append(statement_term(item, evidence_match(building, item, vocab)))
        else:
            out.append((clue_match(building, item, vocab), item.confidence))
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
    evidence is skipped (see ``statement_matches``).
    """
    prior, p_nim = compute_prior(building, hour=hour, priors_cfg=priors_cfg)
    phone_lik = phone_likelihood(building, fix)
    return fuse(prior, p_nim, phone_lik, statement_matches(building, clues, vocab))

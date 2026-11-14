"""Bayesian fusion of prior, phone fix and caller clues.

    p(r | E) ∝ pi(r) * L_phone(r) * Π_i [eps + (1 - eps) * m_i(r)]^{c_i}

* eps = 0.1 floors every clue factor so one wrong / misheard clue can never
  zero a room out.
* c_i in [0, 1] is the clue confidence; it tempers the factor (c = 0 -> ignored).
* "Not in model" bucket: it has no geometry, so it is scored at the base rate -
  L_phone = prior-weighted mean of the room likelihoods, m_i = prior-weighted
  mean of m_i(r) - with the same eps floor and tempering.

Output is normalised: sum(room_probs) + p_not_in_model == 1.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from core.clue_likelihood import clue_match
from core.phone_likelihood import phone_likelihood
from core.prior import compute_prior
from core.schema import Building, Clue, PhoneFix

EPS = 0.1


def clue_factor(m: float, confidence: float, eps: float = EPS) -> float:
    """[eps + (1 - eps) * m]^c with m and c clamped to [0, 1]."""
    m = min(1.0, max(0.0, m))
    c = min(1.0, max(0.0, confidence))
    return (eps + (1.0 - eps) * m) ** c


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float], default: float = 1.0) -> float:
    total = sum(weights.values())
    if total <= 0.0:
        return default
    return sum(w * values.get(rid, default) for rid, w in weights.items()) / total


def fuse(
    prior: Mapping[str, float],
    p_nim_prior: float,
    phone_lik: Mapping[str, float] | None,
    clue_matches: Sequence[tuple[Mapping[str, float], float]],
    eps: float = EPS,
) -> tuple[dict[str, float], float]:
    """Posterior (room_probs, p_not_in_model) from prior x phone likelihood x clue factors.

    ``clue_matches`` is a list of (match vector m_i over rooms, confidence c_i).
    Rooms missing from a likelihood / match vector are treated as uninformative (1.0).
    """
    scores = {rid: float(p) for rid, p in prior.items()}
    nim = float(p_nim_prior)

    if phone_lik is not None:
        nim *= _weighted_mean(phone_lik, prior)
        for rid in scores:
            scores[rid] *= phone_lik.get(rid, 1.0)

    for match, confidence in clue_matches:
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


def posterior_probs(
    building: Building,
    hour: int = 14,
    fix: PhoneFix | None = None,
    clues: Iterable[Clue] = (),
    priors_cfg: dict | None = None,
    vocab: dict | None = None,
) -> tuple[dict[str, float], float]:
    """Full pipeline: prior -> phone likelihood -> clue matches -> fuse."""
    prior, p_nim = compute_prior(building, hour=hour, priors_cfg=priors_cfg)
    phone_lik = phone_likelihood(building, fix)
    matches = [(clue_match(building, clue, vocab), clue.confidence) for clue in clues]
    return fuse(prior, p_nim, phone_lik, matches)

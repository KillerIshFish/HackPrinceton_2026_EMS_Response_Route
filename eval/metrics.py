"""Evaluation metrics - pure functions over a list of per-trial results.

A trial is one simulated caller scored under one evidence condition. Nothing
here imports the engine; build ``Trial`` objects however you like (see
``Trial.from_posterior`` for the usual path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping, Optional, Sequence


@dataclass
class Trial:
    true_room_id: str
    true_floor_label: str
    room_probs: Mapping[str, float]
    p_floor: Mapping[str, float]
    credible_set_90: Sequence[str]
    entropy_bits: float = 0.0
    p_not_in_model: float = 0.0
    # Planner outputs (None when the planner was not run for this trial).
    expected_time_s: Optional[float] = None  # planner's E[T] under its own posterior
    baseline_nearest_s: Optional[float] = None
    baseline_highest_prob_s: Optional[float] = None
    # Time at which the TRUE room's search completes under each ordering
    # (None = the true room was not in the plan / planner not run).
    realized_time_s: Optional[float] = None
    realized_nearest_s: Optional[float] = None
    realized_highest_prob_s: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_posterior(cls, true_room_id: str, true_floor_label: str, posterior: Any, **kw: Any) -> "Trial":
        """Build from a ``core.schema.Posterior`` (or any object with the same fields)."""
        return cls(
            true_room_id=true_room_id,
            true_floor_label=true_floor_label,
            room_probs=posterior.room_probs,
            p_floor=posterior.p_floor,
            credible_set_90=posterior.credible_set_90,
            entropy_bits=posterior.entropy_bits,
            p_not_in_model=posterior.p_not_in_model,
            **kw,
        )


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else float("nan")


def _present(trials: Sequence[Trial], attr: str) -> list[float]:
    return [float(v) for v in (getattr(t, attr) for t in trials) if v is not None]


# ---------------------------------------------------------------------------
# Posterior quality
# ---------------------------------------------------------------------------


def floor_accuracy(trials: Sequence[Trial]) -> float:
    """Fraction of trials whose most probable floor is the true floor."""
    hits = [
        1.0 if t.p_floor and max(t.p_floor, key=lambda k: t.p_floor[k]) == t.true_floor_label else 0.0
        for t in trials
    ]
    return _mean(hits)


def true_room_rank(trial: Trial) -> int:
    """1-based rank of the true room by posterior probability (ties broken by id)."""
    p_true = trial.room_probs.get(trial.true_room_id, 0.0)
    better = sum(
        1 for rid, p in trial.room_probs.items()
        if p > p_true or (p == p_true and rid < trial.true_room_id)
    )
    return better + 1


def top_k_accuracy(trials: Sequence[Trial], k: int = 3) -> float:
    """Fraction of trials with the true room among the k most probable rooms."""
    return _mean([1.0 if true_room_rank(t) <= k else 0.0 for t in trials])


def coverage_90(trials: Sequence[Trial]) -> float:
    """Fraction of trials whose 90% credible set contains the true room (target ~0.90)."""
    return _mean([1.0 if t.true_room_id in set(t.credible_set_90) else 0.0 for t in trials])


def mean_credible_set_size(trials: Sequence[Trial]) -> float:
    return _mean([float(len(t.credible_set_90)) for t in trials])


def mean_entropy_bits(trials: Sequence[Trial]) -> float:
    return _mean([float(t.entropy_bits) for t in trials])


def mean_true_room_prob(trials: Sequence[Trial]) -> float:
    return _mean([float(t.room_probs.get(t.true_room_id, 0.0)) for t in trials])


# ---------------------------------------------------------------------------
# Search-plan quality
# ---------------------------------------------------------------------------


def mean_expected_time(trials: Sequence[Trial], attr: str = "expected_time_s") -> float:
    """Mean of a planner time field over the trials where it is available.

    ``attr`` is one of expected_time_s / baseline_nearest_s / baseline_highest_prob_s
    (model-expected E[T]) or realized_time_s / realized_nearest_s /
    realized_highest_prob_s (time until the true room was actually searched).
    """
    return _mean(_present(trials, attr))


def found_rate(trials: Sequence[Trial], attr: str = "realized_time_s") -> float:
    """Fraction of trials where the true room appears in the plan at all."""
    return _mean([1.0 if getattr(t, attr) is not None else 0.0 for t in trials])


def summarize_trials(trials: Sequence[Trial]) -> dict[str, float]:
    """All headline metrics for one condition."""
    return {
        "n": float(len(trials)),
        "floor_acc": floor_accuracy(trials),
        "top1": top_k_accuracy(trials, 1),
        "top3": top_k_accuracy(trials, 3),
        "coverage_90": coverage_90(trials),
        "cs90_size": mean_credible_set_size(trials),
        "entropy_bits": mean_entropy_bits(trials),
        "p_true": mean_true_room_prob(trials),
        "et_plan_s": mean_expected_time(trials, "expected_time_s"),
        "et_nearest_s": mean_expected_time(trials, "baseline_nearest_s"),
        "et_highest_prob_s": mean_expected_time(trials, "baseline_highest_prob_s"),
        "realized_plan_s": mean_expected_time(trials, "realized_time_s"),
        "realized_nearest_s": mean_expected_time(trials, "realized_nearest_s"),
        "realized_highest_prob_s": mean_expected_time(trials, "realized_highest_prob_s"),
        "found_rate": found_rate(trials),
    }

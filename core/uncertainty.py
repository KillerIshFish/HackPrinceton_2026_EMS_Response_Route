"""Uncertainty summaries of a room posterior.

* entropy_bits:  H = -Σ p log2 p over rooms + the "not in model" bucket.
* floor_marginals:  P(floor) = Σ_{r on floor} p_r  (not renormalised: sums to 1 - p_nim).
* credible_set:  smallest set of rooms, highest probability first, covering
  >= 90% of the in-model mass.
* peak_count:  connected components (Building.edges adjacency) among rooms with
  p >= rel_threshold * max p - "how many separate places could the caller be".
"""

from __future__ import annotations

import math
from typing import Mapping

from core.schema import Building, Posterior


def entropy_bits(room_probs: Mapping[str, float], p_nim: float) -> float:
    """Shannon entropy in bits of the distribution {rooms..., not-in-model}."""
    probs = [p for p in room_probs.values() if p > 0.0]
    if p_nim > 0.0:
        probs.append(p_nim)
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for p in probs:
        q = p / total
        h -= q * math.log2(q)
    return max(h, 0.0)


def floor_marginals(building: Building, room_probs: Mapping[str, float]) -> dict[str, float]:
    """P(floor) keyed by floor_label, in floor order. Sums to 1 - p_not_in_model."""
    labels: dict[int, str] = {}
    for room in building.rooms:
        labels.setdefault(room.floor, room.floor_label)
    for floor in building.floors:
        labels.setdefault(floor, str(floor))
    marginals = {labels[f]: 0.0 for f in sorted(labels)}
    for room in building.rooms:
        marginals[labels[room.floor]] += room_probs.get(room.id, 0.0)
    return marginals


def credible_set(room_probs: Mapping[str, float], mass: float = 0.90) -> list[str]:
    """Smallest set of rooms (descending probability) covering >= ``mass`` of in-model mass."""
    total = sum(p for p in room_probs.values() if p > 0.0)
    if total <= 0.0:
        return []
    target = mass * total - 1e-12  # float slack, so rounding cannot add one more room past the 90 %
    ranked = sorted(room_probs.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen: list[str] = []
    cumulative = 0.0
    for rid, p in ranked:
        if p <= 0.0:
            break
        chosen.append(rid)
        cumulative += p
        if cumulative >= target:
            break
    return chosen


def peak_count(building: Building, room_probs: Mapping[str, float], rel_threshold: float = 0.5) -> int:
    """Number of connected components among rooms with p >= rel_threshold * max p.

    ``rel_threshold`` 0.5 is a display heuristic (initial commit): a room at least half as likely
    as the best one counts as a place the caller could be.
    """
    p_max = max(room_probs.values(), default=0.0)
    if p_max <= 0.0:
        return 0
    cutoff = rel_threshold * p_max
    hot = {rid for rid, p in room_probs.items() if p >= cutoff}

    parent = {rid: rid for rid in hot}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in building.edges:
        u, v = edge.get("u"), edge.get("v")
        if u in hot and v in hot:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    return len({find(rid) for rid in hot})


def summarize(building: Building, room_probs: Mapping[str, float], p_nim: float) -> Posterior:
    """Bundle a posterior and its uncertainty summaries into the frozen ``Posterior`` model."""
    probs = {rid: float(p) for rid, p in room_probs.items()}
    return Posterior(
        room_probs=probs,
        p_not_in_model=float(p_nim),
        p_floor=floor_marginals(building, probs),
        credible_set_90=credible_set(probs, 0.90),  # 90 % is fixed by the frozen contract field name
        entropy_bits=entropy_bits(probs, p_nim),
        peak_count=peak_count(building, probs),
    )

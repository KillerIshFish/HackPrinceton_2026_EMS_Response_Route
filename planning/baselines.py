"""Baseline search orders used to benchmark the planner's E[T].

Both baselines visit the *same candidate set* from the *same start* as the real
planner, with a single team, so the E[T] comparison is apples-to-apples.

Shared signature::

    order = baseline(candidates, probs, travel, start)

* ``candidates`` - room ids to visit
* ``probs``      - room id -> posterior probability
* ``travel``     - travel[a][b] = seconds from node a to node b
* ``start``      - node id the team starts from (entrance or a room)

This module deliberately imports nothing from ``planning.search_order`` (which
imports us) - it only produces orders; E[T] is computed by the caller.
"""

from __future__ import annotations

from typing import Mapping, Sequence

UNREACHABLE_S = 3600.0  # travel time assumed when the graph has no path


def _t(travel: Mapping[str, Mapping[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return travel.get(a, {}).get(b, UNREACHABLE_S)


def nearest_first_order(
    candidates: Sequence[str],
    probs: Mapping[str, float],
    travel: Mapping[str, Mapping[str, float]],
    start: str,
) -> list[str]:
    """Sweep: repeatedly walk to the nearest unvisited candidate (ignores probability)."""
    remaining = list(candidates)
    order: list[str] = []
    here = start
    while remaining:
        # Tie-break on higher probability, then id, so the order is deterministic.
        nxt = min(remaining, key=lambda r: (_t(travel, here, r), -probs.get(r, 0.0), r))
        order.append(nxt)
        remaining.remove(nxt)
        here = nxt
    return order


def highest_prob_first_order(
    candidates: Sequence[str],
    probs: Mapping[str, float],
    travel: Mapping[str, Mapping[str, float]] | None = None,
    start: str | None = None,
) -> list[str]:
    """Visit rooms in descending posterior probability (ignores travel time)."""
    return sorted(candidates, key=lambda r: (-probs.get(r, 0.0), r))

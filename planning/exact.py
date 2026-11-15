"""Exact single-team search order for small candidate sets (step A2).

Objective (linear, one team): minimise ``sum_k p_k * C_k`` where ``C_k`` is the time the
look at room ``k`` completes. Written leg by leg, every leg's duration is paid by all the
mass not yet searched when the leg starts::

    sum_k p_k C_k = sum over legs (T[prev][k] + s[k]) * (M - p(rooms searched before k))

so the cost of reaching state ``(S = rooms searched, j = room just searched)`` does not
depend on the order inside ``S``. That gives the Held-Karp style dynamic programme::

    f({k}, k)     = (T[start][k] + s[k]) * (M + lam)
    f(S + k, k)   = min_j  f(S, j) + (T[j][k] + s[k]) * (M - p(S) + lam)

Walking penalty ``lam`` (``planning.search_order``, "Walking penalty"): the objective
``sum_k p_k C_k + lam * T_total`` (``T_total`` = when the last look ends) is the same leg sum
with every leg also paying ``lam``, because every leg delays the end of the route. ``lam`` is a
constant added to the weight, so the state space, the exactness and the speed are unchanged.
``lam = 0`` is the original programme, bit for bit.

over ``2^n * n`` states, ``O(2^n n^2)`` work. It is vectorised with numpy one subset size
(layer) at a time. Measured (``python -m planning.exact``, median of random instances):
0.6 ms at 8 rooms, 3 ms at 12, 10 ms at 14, 68 ms at 16. ``EXACT_MAX_ROOMS`` stays at the
build plan's 12, far inside its 150 ms budget (16 would also fit).

The programme does not apply to a non-linear survival objective: elapsed time is not a
function of the state, so ``planning.search_order`` seeds that case with this linear
solution and improves it by local search.

Indices follow ``planning.search_order``: rooms ``0..n-1``, the start node is index ``n``;
``T`` is an ``(n+1) x (n+1)`` travel matrix in seconds, ``s`` look times, ``p`` weights.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

# Largest candidate set solved exactly: the build plan's 12 (docs/BUILD_PLAN.md, A2), measured well under its
# 150 ms budget (timings above). tests/test_search_maths.py pins both the value and the 150 ms bound.
EXACT_MAX_ROOMS = 12


def solve_exact(T: Sequence[Sequence[float]], s: Sequence[float], p: Sequence[float], start: int,
                lam: float = 0.0) -> tuple[list[int], float]:
    """Optimal order of rooms ``0..n-1`` from node ``start`` for ``sum p_k C_k + lam * T_total``.

    Returns ``(order, cost)`` with the un-normalised cost (penalty included). Ties go to the
    lower room index.
    Raises ``ValueError`` above 20 rooms (memory); callers use ``EXACT_MAX_ROOMS``.
    """
    n = len(p)
    if n == 0:
        return [], 0.0
    # Hard memory cap: f, parent and bits are (2^n, n) arrays; at 20 rooms f alone is 2^20 x 20 x 8 B = 168 MB.
    if n > 20:
        raise ValueError(f"exact solver limited to 20 rooms, got {n}")
    Tm = np.asarray(T, dtype=float)
    sv = np.asarray(s, dtype=float)
    pv = np.asarray(p, dtype=float)
    D = Tm[:n, :n] + sv[None, :]  # D[j, k] = leg j -> k plus the look at k
    D0 = Tm[start, :n] + sv  # start -> k plus the look at k
    total = float(pv.sum())

    full = 1 << n
    subsets = np.arange(full, dtype=np.int64)
    bits = (subsets[:, None] >> np.arange(n)) & 1  # (2^n, n)
    mass_in = bits @ pv  # p(S)
    remaining = total - mass_in + float(lam)  # mass not yet searched, plus the walking penalty
    popcount = bits.sum(axis=1)

    f = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int16)
    for k in range(n):
        f[1 << k, k] = D0[k] * (total + float(lam))

    for size in range(1, n):
        layer = subsets[popcount == size]
        f_layer = f[layer]  # (L, n), inf where j is not in S
        rem_layer = remaining[layer]
        for k in range(n):
            free = ((layer >> k) & 1) == 0
            if not free.any():
                continue
            vals = f_layer[free] + D[:, k][None, :] * rem_layer[free][:, None]
            j_best = np.argmin(vals, axis=1)
            best = vals[np.arange(len(j_best)), j_best]
            new = layer[free] | (1 << k)
            f[new, k] = best
            parent[new, k] = j_best

    last_set = full - 1
    j = int(np.argmin(f[last_set]))
    cost = float(f[last_set, j])
    order: list[int] = []
    S = last_set
    while j >= 0:
        order.append(j)
        prev = int(parent[S, j])
        S &= ~(1 << j)
        j = prev
    order.reverse()
    return order, cost


def time_exact(n: int, trials: int = 3, seed: int = 0) -> float:
    """Median wall time (ms) of ``solve_exact`` on random ``n``-room instances."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        # Synthetic floor: points in a 60 m square, straight-line walks at 1.2 m/s, looks of 10-40 s.
        # Run time depends only on n, so these just need to be plausible.
        pts = rng.random((n + 1, 2)) * 60.0
        T = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)) / 1.2
        s = rng.uniform(10, 40, n)
        p = rng.random(n)
        t0 = time.perf_counter()
        solve_exact(T, s, p, n)
        out.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(out))


if __name__ == "__main__":  # pragma: no cover - timing report
    for n in (8, 10, 12, 13, 14, 16):
        print(f"{n:2d} rooms: {time_exact(n):8.1f} ms")

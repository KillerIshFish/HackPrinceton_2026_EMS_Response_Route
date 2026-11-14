"""Search-order planner.

Objective (expected time to find the caller, conditioned on them being in one
of the candidate rooms)::

    E[T] = sum_r p_r * C_r / sum_r p_r

where ``C_r`` is the time at which the search of room ``r`` *completes*
(arrival + search time) under the team that visits it.

Pipeline
--------
1. Candidates  = most probable rooms covering ``coverage`` of the in-model mass.
2. Travel      = shortest paths on the building graph (1.2 m/s walking, +15 s
                 per floor on stairs - see ``core.building_graph``).
3. Greedy      = from the current position take the room maximising
                 ``p_r / (travel + search_r)`` (the classic density rule).
4. Local search = 2-opt segment reversals + or-opt single-room relocations,
                 accepting only moves that lower E[T].
5. Multi-team  = event-driven greedy (the team that frees up first picks next),
                 per-team local search, then cross-team relocations.
6. Koopman     = when a room is searched and declared clear with detection
                 probability 0.9: ``p_r <- p_r * (1 - 0.9)`` then renormalise.
"""

from __future__ import annotations

import time
from typing import Iterable, Mapping, Optional, Sequence

import networkx as nx

from core.building_graph import build_graph, travel_times_from
from core.schema import Building, Room, SearchPlan, Stop
from planning.approach import recommend_approach, staging_for_entrance
from planning.baselines import (
    UNREACHABLE_S,
    highest_prob_first_order,
    nearest_first_order,
)

SEARCH_S_PER_M2 = 0.5
MIN_SEARCH_S = 10.0
P_DETECT = 0.9

_OPT_BUDGET_S = 0.25  # wall-clock cap on local search per plan_search call
_LARGE_ROUTE = 200       # routes longer than this use windowed moves (see _improve_route)
_LARGE_ROUTE_SPAN = 30
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def search_time_s(room: Room) -> float:
    """Seconds to search a room: proportional to floor area, with a floor."""
    return max(MIN_SEARCH_S, SEARCH_S_PER_M2 * float(room.area_m2))


def koopman_update(
    room_probs: Mapping[str, float],
    p_nim: float,
    cleared_room_id: str,
    p_detect: float = P_DETECT,
) -> tuple[dict[str, float], float]:
    """Bayesian update after an unsuccessful search of ``cleared_room_id``.

    ``p_r <- p_r * (1 - p_detect)`` for the cleared room, then every room and the
    "not in model" bucket are renormalised so ``sum(rooms) + p_nim == 1``.
    """
    if cleared_room_id not in room_probs:
        raise KeyError(f"unknown room id: {cleared_room_id!r}")
    updated = dict(room_probs)
    updated[cleared_room_id] = updated[cleared_room_id] * (1.0 - p_detect)
    total = sum(updated.values()) + p_nim
    if total <= 0.0:
        return updated, p_nim
    return {rid: p / total for rid, p in updated.items()}, p_nim / total


def _t(travel: Mapping[str, Mapping[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return travel.get(a, {}).get(b, UNREACHABLE_S)


def completion_times(
    order: Sequence[str],
    travel: Mapping[str, Mapping[str, float]],
    search_times: Mapping[str, float],
    start: str,
) -> dict[str, float]:
    """room id -> time (s) at which its search completes when walking ``order``."""
    out: dict[str, float] = {}
    t = 0.0
    here = start
    for rid in order:
        t += _t(travel, here, rid) + search_times[rid]
        out[rid] = t
        here = rid
    return out


def expected_time(
    order: Sequence[str],
    probs: Mapping[str, float],
    travel: Mapping[str, Mapping[str, float]],
    search_times: Mapping[str, float],
    start: str,
) -> float:
    """E[T] = sum_r p_r C_r / sum_r p_r for a single team walking ``order``."""
    done = completion_times(order, travel, search_times, start)
    mass = sum(probs.get(rid, 0.0) for rid in order)
    if mass <= 0.0:
        return 0.0
    return sum(probs.get(rid, 0.0) * c for rid, c in done.items()) / mass


def select_candidates(
    building: Building,
    room_probs: Mapping[str, float],
    coverage: float = 0.99,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Most probable rooms (desc) covering ``coverage`` of the in-model mass.

    Rooms in ``exclude`` (e.g. already cleared) are never candidates; coverage is
    measured over the remaining rooms.
    """
    known = {r.id for r in building.rooms} - set(exclude)
    ranked = sorted(
        ((rid, p) for rid, p in room_probs.items() if rid in known and p > 0.0),
        key=lambda kv: (-kv[1], kv[0]),
    )
    total = sum(p for _, p in ranked)
    if total <= 0.0:
        return []
    out: list[str] = []
    acc = 0.0
    for rid, p in ranked:
        out.append(rid)
        acc += p
        if acc >= coverage * total - 1e-12:
            break
    return out


# Travel times never change for a building, and a call replans many times. Keep the rows
# already computed for the building being planned (one building at a time, matched by identity).
_travel_cache: dict = {"graph": None, "rows": {}}


_graph_cache: dict = {"building": None, "graph": None}


def _graph_for(building: Building) -> nx.Graph:
    """The building's room graph, rebuilt only when a different building object is planned."""
    if _graph_cache["building"] is not building:
        _graph_cache["building"], _graph_cache["graph"] = building, build_graph(building)
    return _graph_cache["graph"]


def pairwise_travel(G: nx.Graph, nodes: Sequence[str]) -> dict[str, dict[str, float]]:
    """travel[a][b] seconds for every a in ``nodes`` (b = any graph node)."""
    if _travel_cache["graph"] is not G:
        _travel_cache["graph"], _travel_cache["rows"] = G, {}
    rows = _travel_cache["rows"]
    out = {}
    for n in dict.fromkeys(nodes):
        if n in G:
            if n not in rows:
                rows[n] = travel_times_from(G, n)
            out[n] = rows[n]
    return out


# ---------------------------------------------------------------------------
# Index-based route optimisation (rooms 0..n-1, start = index n)
# ---------------------------------------------------------------------------


def _route_cost(order: Sequence[int], T, s, p, start: int) -> float:
    """Un-normalised sum p_r * C_r for one team."""
    t = 0.0
    tot = 0.0
    prev = start
    for k in order:
        t += T[prev][k] + s[k]
        tot += p[k] * t
        prev = k
    return tot


def _greedy_single(cands: Sequence[int], T, s, p, start: int) -> list[int]:
    remaining = list(cands)
    order: list[int] = []
    prev = start
    while remaining:
        best = max(remaining, key=lambda k: (p[k] / (T[prev][k] + s[k] + _EPS), -k))
        order.append(best)
        remaining.remove(best)
        prev = best
    return order


def _greedy_teams(cands: Sequence[int], T, s, p, start: int, n_teams: int) -> list[list[int]]:
    """Event-driven greedy: whichever team frees up first takes the best-ratio room."""
    routes: list[list[int]] = [[] for _ in range(n_teams)]
    clock = [0.0] * n_teams
    pos = [start] * n_teams
    remaining = list(cands)
    while remaining:
        team = min(range(n_teams), key=lambda i: (clock[i], i))
        here = pos[team]
        best = max(remaining, key=lambda k: (p[k] / (T[here][k] + s[k] + _EPS), -k))
        routes[team].append(best)
        remaining.remove(best)
        clock[team] += T[here][best] + s[best]
        pos[team] = best
    return routes


def _improve_route(order: Sequence[int], T, s, p, start: int, deadline: float) -> tuple[list[int], float]:
    """2-opt reversals + or-opt relocations; accepts only strict E[T] improvements.

    A move rewrites a window ``order[lo..hi]``. Everything before ``lo`` is
    untouched; everything after ``hi`` keeps its order, so its completion times
    all shift by one constant ``delta`` - the move is scored in O(window).
    """
    order = list(order)
    n = len(order)
    best = _route_cost(order, T, s, p, start)
    if n < 2:
        return order, best

    pre_t = [0.0] * (n + 1)  # clock after finishing order[:i]
    pre_c = [0.0] * (n + 1)  # cost of order[:i]
    suf_p = [0.0] * (n + 1)  # sum of p over order[i:]

    def refresh() -> None:
        t = 0.0
        c = 0.0
        prev = start
        for i, k in enumerate(order):
            t += T[prev][k] + s[k]
            c += p[k] * t
            pre_t[i + 1] = t
            pre_c[i + 1] = c
            prev = k
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc += p[order[i]]
            suf_p[i] = acc
        suf_p[n] = 0.0

    def window_cost(lo: int, hi: int, window: Sequence[int]) -> float:
        t = pre_t[lo]
        c = pre_c[lo]
        prev = order[lo - 1] if lo else start
        for k in window:
            t += T[prev][k] + s[k]
            c += p[k] * t
            prev = k
        if hi + 1 < n:
            nxt = order[hi + 1]
            # new clock on arrival at nxt minus old clock on arrival at nxt
            delta = (t + T[prev][nxt]) - (pre_t[hi + 1] + T[order[hi]][nxt])
            c += (pre_c[n] - pre_c[hi + 1]) + delta * suf_p[hi + 1]
        return c

    refresh()
    # A full sweep is O(n^3). Past _LARGE_ROUTE stops, moves are limited to a window of
    # _LARGE_ROUTE_SPAN positions and the time budget is checked inside the sweep, so a
    # building with hundreds of rooms still plans in a fraction of a second. Short routes
    # are searched exactly as before.
    large = n > _LARGE_ROUTE
    span = _LARGE_ROUTE_SPAN if large else n
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        # --- 2-opt: reverse order[i..j] ---------------------------------
        for i in range(n - 1):
            if large and time.perf_counter() >= deadline:
                break
            for j in range(i + 1, min(n, i + 1 + span)):
                window = order[i : j + 1][::-1]
                c = window_cost(i, j, window)
                if c < best - _EPS:
                    order[i : j + 1] = window
                    refresh()
                    best = pre_c[n]
                    improved = True
        # --- or-opt: move the room at i to position j ---------------------
        for i in range(n):
            if large and time.perf_counter() >= deadline:
                break
            for j in range(max(0, i - span), min(n, i + 1 + span)):
                if i == j:
                    continue
                if i < j:
                    lo, hi = i, j
                    window = order[i + 1 : j + 1] + [order[i]]
                else:
                    lo, hi = j, i
                    window = [order[i]] + order[j:i]
                c = window_cost(lo, hi, window)
                if c < best - _EPS:
                    order[lo : hi + 1] = window
                    refresh()
                    best = pre_c[n]
                    improved = True
    # Guard against accumulated float drift in the incremental bookkeeping.
    return order, _route_cost(order, T, s, p, start)


def _relocate_between_teams(
    routes: list[list[int]], T, s, p, start: int, deadline: float
) -> list[list[int]]:
    """Move single rooms between teams when that lowers the total cost."""
    costs = [_route_cost(r, T, s, p, start) for r in routes]
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for a in range(len(routes)):
            i = 0
            while i < len(routes[a]):
                room = routes[a][i]
                src_wo = routes[a][:i] + routes[a][i + 1 :]
                src_cost = _route_cost(src_wo, T, s, p, start)
                best_gain = _EPS
                best_move: Optional[tuple[int, int, float]] = None
                for b in range(len(routes)):
                    if b == a:
                        continue
                    for j in range(len(routes[b]) + 1):
                        cand = routes[b][:j] + [room] + routes[b][j:]
                        dst_cost = _route_cost(cand, T, s, p, start)
                        gain = (costs[a] + costs[b]) - (src_cost + dst_cost)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (b, j, dst_cost)
                if best_move is not None:
                    b, j, dst_cost = best_move
                    routes[a] = src_wo
                    routes[b] = routes[b][:j] + [room] + routes[b][j:]
                    costs[a], costs[b] = src_cost, dst_cost
                    improved = True
                else:
                    i += 1
                if time.perf_counter() >= deadline:
                    return routes
    return routes


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------


def plan_search(
    building: Building,
    room_probs: Mapping[str, float],
    p_nim: float,
    n_teams: int = 1,
    entrance_id: str | None = None,
    start_node: str | None = None,
    coverage: float = 0.99,
    exclude: Iterable[str] = (),
) -> SearchPlan:
    """Build the ordered search plan for the current posterior.

    * ``entrance_id``  - force an entrance; default is ``recommend_approach``'s pick.
    * ``start_node``   - where the team(s) are now (e.g. the room just cleared when
      replanning). Defaults to the entrance. All ETAs are relative to this node.
    * ``p_nim``        - "not in model" mass; E[T] is conditional on the caller
      being in a candidate room, so it only matters through ``room_probs``.
    * ``exclude``      - room ids to leave out of the plan (rooms already marked
      clear). Without this, a just-cleared room that still holds the largest
      residual probability can legitimately come back as stop #1 (Koopman
      re-search), which is rarely what a dispatcher wants to see.

    The two baselines are single-team routes over the same candidates from the
    same start. For ``n_teams == 1`` the returned E[T] is guaranteed to be
    <= both baselines.
    """
    n_teams = max(1, int(n_teams))
    G = _graph_for(building)
    rooms = {r.id: r for r in building.rooms}

    # ---- entrance + staging ------------------------------------------------
    if entrance_id is None:
        approach = recommend_approach(building, room_probs, G)
        entrance_id = approach["entrance_id"]
        staging_latlng = tuple(approach["staging_latlng"])
    else:
        if not any(e.get("id") == entrance_id for e in building.entrances):
            raise KeyError(f"unknown entrance id: {entrance_id!r}")
        spot = staging_for_entrance(building, entrance_id)
        ll = (spot or {}).get("latlng") or (0.0, 0.0)
        staging_latlng = (float(ll[0]), float(ll[1]))

    start = start_node or entrance_id
    if not start or start not in G:
        if start_node:
            raise KeyError(f"unknown start node: {start_node!r}")
        start = building.rooms[0].id if building.rooms else ""

    cand_ids = select_candidates(building, room_probs, coverage, exclude)
    if not cand_ids or not start:
        return SearchPlan(
            teams=[[] for _ in range(n_teams)],
            expected_time_to_find_s=0.0,
            baseline_nearest_s=0.0,
            baseline_highest_prob_s=0.0,
            entrance_id=entrance_id or "",
            staging_latlng=staging_latlng,
        )

    # ---- travel + search times ---------------------------------------------
    travel = pairwise_travel(G, [start, *cand_ids])
    search_times = {rid: search_time_s(rooms[rid]) for rid in cand_ids}
    probs = {rid: float(room_probs[rid]) for rid in cand_ids}
    mass = sum(probs.values())

    n = len(cand_ids)
    idx = {rid: i for i, rid in enumerate(cand_ids)}
    START = n
    nodes = [*cand_ids, start]
    T = [[_t(travel, a, b) for b in nodes] for a in nodes]
    s = [search_times[rid] for rid in cand_ids]
    p = [probs[rid] for rid in cand_ids]

    # ---- baselines (single team, same candidates, same start) ---------------
    near_order = [idx[r] for r in nearest_first_order(cand_ids, probs, travel, start)]
    high_order = [idx[r] for r in highest_prob_first_order(cand_ids, probs, travel, start)]
    near_cost = _route_cost(near_order, T, s, p, START)
    high_cost = _route_cost(high_order, T, s, p, START)

    # ---- optimise -----------------------------------------------------------
    deadline = time.perf_counter() + _OPT_BUDGET_S
    if n_teams == 1:
        order, cost = _improve_route(_greedy_single(range(n), T, s, p, START), T, s, p, START, deadline)
        # Guarantee: never worse than either baseline. If one beats the polished
        # greedy route, polish that baseline instead and keep the best.
        for b_order, b_cost in ((near_order, near_cost), (high_order, high_cost)):
            if b_cost < cost - _EPS:
                o2, c2 = _improve_route(b_order, T, s, p, START, time.perf_counter() + 0.1)
                if c2 > b_cost:  # cannot happen (moves only improve); belt and braces
                    o2, c2 = list(b_order), b_cost
                if c2 < cost:
                    order, cost = o2, c2
        routes = [order]
    else:
        routes = _greedy_teams(range(n), T, s, p, START, n_teams)
        routes = [_improve_route(r, T, s, p, START, deadline)[0] for r in routes]
        routes = _relocate_between_teams(routes, T, s, p, START, deadline)
        routes = [_improve_route(r, T, s, p, START, deadline)[0] for r in routes]
        cost = sum(_route_cost(r, T, s, p, START) for r in routes)

    # ---- assemble ------------------------------------------------------------
    teams: list[list[Stop]] = []
    for route in routes:
        stops: list[Stop] = []
        t = 0.0
        cum = 0.0
        prev = START
        for k in route:
            arrive = t + T[prev][k]
            t = arrive + s[k]
            cum += p[k]
            room = rooms[cand_ids[k]]
            stops.append(
                Stop(
                    room_id=room.id,
                    room_name=room.name,
                    floor_label=room.floor_label,
                    eta_s=round(arrive, 1),
                    cumulative_prob=round(cum, 6),
                )
            )
            prev = k
        teams.append(stops)

    return SearchPlan(
        teams=teams,
        expected_time_to_find_s=round(cost / mass, 2),
        baseline_nearest_s=round(near_cost / mass, 2),
        baseline_highest_prob_s=round(high_cost / mass, 2),
        entrance_id=entrance_id or "",
        staging_latlng=staging_latlng,
    )

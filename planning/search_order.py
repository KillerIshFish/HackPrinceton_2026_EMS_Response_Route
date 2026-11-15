"""Search-order planner.

Detection model
---------------
A look at a room in search mode ``m`` (``rapid`` / ``thorough``) finds a person who is
there with probability ``q_m`` and takes ``search_time_s(room) * time_multiplier_m``
(settings in ``configs/situations.yaml``, section ``search``, all labelled unverified).
A room that has had looks ``l_1..l_k`` without success keeps the residual mass::

    residual_r = p_r * prod_i (1 - q_{l_i})

and stays in the problem: it can come back as a ``re_search`` stop when its residual
index beats the untouched rooms (Black 1965: index ``p_r (1-q)^m q / t_r``).

The probability that a planned stop is the one that finds the caller is::

    p_find = residual_r * q_look

Objective (expected time to find the caller, conditioned on the plan finding them)::

    E[T] = sum_k p_find_k * C_k / sum_k p_find_k

where ``C_k`` is the time at which stop ``k`` *completes* (arrival + look time) under the
team that walks it. Every stop in one plan uses the same look mode, so ``q_look`` is a
common factor: the optimiser works on ``residual_r`` (= ``p_find / q_look``), which gives
the same order and E[T]. With nothing searched, residual_r == p_r and the plan is the
same as the original ``sum p_r C_r`` planner, bit for bit.

Pipeline
--------
1. Candidates  = rooms with the most residual mass covering ``coverage`` of it; rooms
                 the officer closed (``exclude`` / ``gates``) are never candidates.
2. Travel      = shortest paths on the building graph built for the travel profile and
                 the officer's gates (``core.building_graph.build_graph``; no profile =
                 1.2 m/s walking, +15 s per floor on stairs).
3. Look times  = area-based (``search_time_s``); with smoke reported, wall length divided
                 by the profile's search-advance rate (see ``search_time_s``).
4. One team    = exact dynamic programme (``planning.exact``) when there are at most
                 ``EXACT_MAX_ROOMS`` candidates. Larger sets: greedy by
                 ``p_find / (travel + look time)``, also seeded with the best wing first
                 (cluster-density check, ``_cluster_seed``), each polished by 2-opt +
                 or-opt local search; the best route wins, and it is never worse than the
                 nearest-first or highest-probability-first baseline.
5. Survival    = with a survival function ``S`` the objective becomes
                 ``sum p_find * S(arrival + treatment_s)`` (maximised). The programme does
                 not apply (elapsed time is not a function of its state), so the linear
                 solution seeds a local search on the survival objective.
6. Teams       = team 1 is planned on the full belief and keeps its stops up to a horizon
                 (one-team time to cover every candidate / number of teams); each room it
                 searches is multiplied by (1 - q); team 2 is planned on what is left, and
                 so on; then single rooms move between routes when that lowers the cost.
                 A team is a pair (``crew_size // 2`` teams at most).
7. Koopman     = posterior after an unsuccessful look with detection probability q:
                 ``p_r <- p_r * (1 - q)`` then renormalise with the outside mass
                 (``koopman_update`` / ``residual_probs``).
8. Second team = with ``second_team_rooms`` (the floors the phone's height fix allows, when
                 a caller statement contradicts the fix: ``core.fusion``) and two or more
                 teams, team 2's route starts with the stops in those rooms unless a team
                 already starts there (``_start_second_team_in``). Team 1 follows the
                 caller's statement; the phone's floor is not left for later.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import networkx as nx

from core.building_graph import build_graph, travel_times_from
from core.schema import Building, Gates, Room, SearchMode, SearchPlan, Stop, TravelProfile
from planning.approach import recommend_approach, staging_for_entrance
from planning.exact import EXACT_MAX_ROOMS, solve_exact
from planning.baselines import (
    UNREACHABLE_S,
    highest_prob_first_order,
    nearest_first_order,
)
from situation.config import section, setting_value

# Fallbacks when configs/situations.yaml has no search section (same numbers as the file).
SEARCH_S_PER_M2 = 0.5
MIN_SEARCH_S = 10.0
P_DETECT = 0.9
_DEFAULT_MODES: dict[str, tuple[float, float]] = {"rapid": (0.9, 1.0), "thorough": (0.97, 2.0)}

_OPT_BUDGET_S = 0.25  # wall-clock cap on local search per plan_search call
_LARGE_ROUTE = 200       # routes longer than this use windowed moves (see _improve_route)
_LARGE_ROUTE_SPAN = 30
_SEED_BUDGET_S = 0.1     # extra local-search time for a second seed (cluster / baseline)
_EPS = 1e-9
WING_LINK_M = 25.0       # same-floor candidates this close (walking) form one wing
_SURVIVAL_GRID_S = 1.0   # survival functions are interpolated on this grid (memoised)
NIST_TN_1797_NOTE = (
    "Teams are pairs. In the one measured study (NIST TN 1797), a crew of five or six "
    "working as two pairs searched about 1.55 times as fast as a crew of three, not twice."
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeSetting:
    """One search mode: detection probability of a single look and its time multiplier."""

    detection_probability: float
    time_multiplier: float = 1.0
    label: str = "unverified"


def search_modes() -> dict[str, ModeSetting]:
    """Search modes from ``configs/situations.yaml`` (``search.modes``), with built-in fallbacks."""
    cfg = (section("search").get("modes") or {})
    out: dict[str, ModeSetting] = {}
    for mode, (q_default, mult_default) in _DEFAULT_MODES.items():
        entry = cfg.get(mode) or {}
        q_entry = entry.get("detection_probability")
        label = q_entry.get("label", "unverified") if isinstance(q_entry, dict) else "unverified"
        out[mode] = ModeSetting(
            detection_probability=float(setting_value(q_entry, q_default)),
            time_multiplier=float(setting_value(entry.get("time_multiplier"), mult_default)),
            label=str(label),
        )
    return out


def detection_probability(mode: str = "rapid", modes: Optional[Mapping[str, ModeSetting]] = None) -> float:
    """q for one look in ``mode``."""
    table = modes if modes is not None else search_modes()
    if mode not in table:
        raise KeyError(f"unknown search mode: {mode!r}")
    return table[mode].detection_probability


def wall_length_m(room: Room) -> float:
    """Length of the room's walls (m): the polygon perimeter when the polygon's area agrees
    with ``area_m2`` (within a factor of two), else the perimeter of a square of the same
    area, ``4 * sqrt(area)`` (the shortest wall any room of that area can have)."""
    area = max(0.0, float(room.area_m2))
    pts = list(room.polygon or [])
    if len(pts) >= 3 and area > 0.0:
        n = len(pts)
        shoelace = abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))) / 2.0
        if 0.5 * area <= shoelace <= 2.0 * area:
            return float(sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n)))
    return 4.0 * math.sqrt(area)


def search_time_s(room: Room, profile: Optional[TravelProfile] = None) -> float:
    """Seconds for one reference (rapid) look at a room.

    Clear air (no profile, or ``profile.search_advance_mps`` unset): proportional to floor
    area, with a floor (``search_s_per_m2``, ``min_search_s``; unverified settings).

    Smoke reported (``profile.search_advance_mps`` set): a wall-oriented search, the
    standard primary-search pattern in smoke, follows the walls once, so the look takes
    ``wall_length_m(room) / search_advance_mps`` (0.040 m/s is the measured hose-advance
    rate in live fire, ODPM 2/2005). The same minimum applies. This is an estimate of
    pattern length, not a timed room search.
    """
    cfg = section("search")
    floor_s = float(setting_value(cfg.get("min_search_s"), MIN_SEARCH_S))
    if profile is not None and profile.search_advance_mps:
        return max(floor_s, wall_length_m(room) / float(profile.search_advance_mps))
    per_m2 = float(setting_value(cfg.get("search_s_per_m2"), SEARCH_S_PER_M2))
    return max(floor_s, per_m2 * float(room.area_m2))


def miss_factor(looks: Iterable[str], modes: Optional[Mapping[str, ModeSetting]] = None) -> float:
    """prod (1 - q) over earlier looks (modes), i.e. the share of a room's mass still unfound."""
    table = modes if modes is not None else search_modes()
    f = 1.0
    for mode in looks:
        f *= 1.0 - detection_probability(mode, table)
    return f


def looks_by_room(searched: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """room id -> modes of its looks so far, oldest first (``searched`` may repeat rooms)."""
    out: dict[str, list[str]] = {}
    for room_id, mode in searched:
        out.setdefault(room_id, []).append(mode)
    return out


def p_find(
    p_r: float,
    earlier_looks: Iterable[str] = (),
    mode: str = "rapid",
    modes: Optional[Mapping[str, ModeSetting]] = None,
) -> float:
    """Probability that the next look (in ``mode``) at a room is the one that finds the caller:
    ``p_r * q_mode * prod over earlier looks (1 - q_earlier)``, on the scale of ``p_r``."""
    table = modes if modes is not None else search_modes()
    return p_r * miss_factor(earlier_looks, table) * detection_probability(mode, table)


def outside_probability(
    room_probs: Mapping[str, float],
    p_nim: float,
    searched: Iterable[tuple[str, str]] = (),
    modes: Optional[Mapping[str, ModeSetting]] = None,
) -> float:
    """``p0 / (p0 + sum_r p_r * prod (1 - q) over looks so far)``: the chance the caller is
    outside the model given every unsuccessful look. ``room_probs`` / ``p_nim`` are the
    beliefs *before* those looks."""
    table = modes if modes is not None else search_modes()
    looks = looks_by_room(searched)
    left = sum(float(p) * miss_factor(looks.get(rid, ()), table) for rid, p in room_probs.items())
    total = float(p_nim) + left
    return float(p_nim) / total if total > 0.0 else 1.0


def residual_probs(
    room_probs: Mapping[str, float],
    p_nim: float,
    searched: Iterable[tuple[str, str]] = (),
    modes: Optional[Mapping[str, ModeSetting]] = None,
) -> tuple[dict[str, float], float]:
    """Posterior after every unsuccessful look in ``searched`` (one Koopman update per look,
    with that look's detection probability). Same normalisation as ``koopman_update``."""
    table = modes if modes is not None else search_modes()
    probs, p0 = dict(room_probs), float(p_nim)
    for room_id, mode in searched:
        probs, p0 = koopman_update(probs, p0, room_id, detection_probability(mode, table))
    return probs, p0


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

    Rooms in ``exclude`` (officer gates) are never candidates; coverage is measured
    over the remaining rooms. ``plan_search`` passes residual masses here, so a room
    already searched competes on what is left of its probability.
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


# A call replans many times with the same building, profile and gates. Graphs are cached per
# (profile, gates) for the building being planned (one building at a time, matched by
# identity); each graph keeps the Dijkstra rows already computed in ``G.graph``.
_graph_cache: dict = {"building": None, "graphs": {}}
_GRAPH_CACHE_MAX = 16


def _profile_key(profile: Optional[TravelProfile]) -> Optional[tuple]:
    """The parts of a profile that change the graph (notes and search rate do not)."""
    if profile is None:
        return None
    return (
        profile.walk_mps, profile.stair_s_per_floor, profile.stair_s_per_floor_high,
        profile.high_floor_from, profile.elevator_allowed, profile.elevator_fixed_s,
        profile.elevator_s_per_floor,
    )


def _graph_for(
    building: Building,
    profile: Optional[TravelProfile] = None,
    closed_room_ids: Iterable[str] = (),
    closed_entrance_ids: Iterable[str] = (),
) -> nx.Graph:
    """The building's room graph for this travel profile and these officer gates (cached).

    With no profile and no gates this is ``build_graph(building)``, today's graph.
    """
    if _graph_cache["building"] is not building:
        _graph_cache["building"], _graph_cache["graphs"] = building, {}
    rooms_closed = frozenset(closed_room_ids)
    ents_closed = frozenset(closed_entrance_ids)
    key = (_profile_key(profile), rooms_closed, ents_closed)
    graphs = _graph_cache["graphs"]
    if key not in graphs:
        if len(graphs) >= _GRAPH_CACHE_MAX:
            graphs.pop(next(iter(graphs)))
        graphs[key] = build_graph(
            building, profile=profile, closed_room_ids=rooms_closed, closed_entrance_ids=ents_closed
        )
    return graphs[key]


def pairwise_travel(G: nx.Graph, nodes: Sequence[str]) -> dict[str, dict[str, float]]:
    """travel[a][b] seconds for every a in ``nodes`` (b = any graph node). Rows are cached on ``G``."""
    rows = G.graph.setdefault("_travel_rows", {})
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
# Cluster-density check, single-team solver, survival objective, teams
# ---------------------------------------------------------------------------


def _wings(cands: Iterable[int], floors: Sequence[int], T, link_s: float) -> list[list[int]]:
    """Connected groups of candidates on one floor: two rooms on the same floor are linked
    when the walk between them takes at most ``link_s``. Groups sorted by lowest index."""
    cands = list(cands)
    parent = {k: k for k in cands}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_floor: dict[int, list[int]] = {}
    for k in cands:
        by_floor.setdefault(floors[k], []).append(k)
    for ks in by_floor.values():
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                if min(T[a][b], T[b][a]) <= link_s:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for k in cands:
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=min)


def _cluster_seed(T, s, p, start: int, floors: Sequence[int], link_s: float) -> Optional[list[int]]:
    """Greedy route that clears the densest wing first.

    Each wing (``_wings``) is scored by its total weight / (time to reach it from ``start``
    and cover it in greedy order). The best wing is searched first, then greedy over the
    rest from its last room. ``None`` when there is only one wing.
    """
    groups = _wings(range(len(p)), floors, T, link_s)
    if len(groups) < 2:
        return None
    best_inner: Optional[list[int]] = None
    best_score = -1.0
    for g in groups:
        inner = _greedy_single(g, T, s, p, start)
        t = 0.0
        prev = start
        for k in inner:
            t += T[prev][k] + s[k]
            prev = k
        score = sum(p[k] for k in g) / (t + _EPS)
        if score > best_score + 1e-15:
            best_inner, best_score = inner, score
    assert best_inner is not None
    chosen = set(best_inner)
    rest = [k for k in range(len(p)) if k not in chosen]
    return best_inner + _greedy_single(rest, T, s, p, best_inner[-1])


def _solve_single(
    T, s, p, start: int, floors: Sequence[int], link_s: float, deadline: float, exact_ok: bool = True
) -> tuple[list[int], float, str]:
    """Best single-team order of rooms ``0..n-1`` for ``sum p C`` -> (order, cost, solver).

    Greedy + local search always runs. Up to ``EXACT_MAX_ROOMS`` rooms the dynamic programme
    then proves (or improves) it and the solver is ``"exact"``; the heuristic order is kept
    when it is already optimal, so ties do not reshuffle a plan. Larger sets also polish the
    densest-wing-first seed (``_cluster_seed``) and keep the better route.
    """
    n = len(p)
    greedy = _greedy_single(range(n), T, s, p, start)
    order, cost = _improve_route(greedy, T, s, p, start, deadline)
    if exact_ok and 0 < n <= EXACT_MAX_ROOMS:
        o3, _ = solve_exact(T, s, p, start)
        c3 = _route_cost(o3, T, s, p, start)
        if c3 < cost - 1e-9 * max(1.0, abs(cost)):
            order, cost = o3, c3
        return order, cost, "exact"
    seed = _cluster_seed(T, s, p, start, floors, link_s) if n > 1 else None
    if seed is not None and seed != greedy:
        o2, c2 = _improve_route(seed, T, s, p, start, time.perf_counter() + _SEED_BUDGET_S)
        if c2 < cost - _EPS:
            order, cost = o2, c2
    return order, cost, "heuristic"


def _survival_curve(survival: Callable[[float], float], treatment_s: float) -> Callable[[float], float]:
    """``t -> S(t + treatment_s)``, linearly interpolated on a 1 s grid with memoised nodes
    (local search evaluates the curve many thousands of times)."""
    cache: dict[int, float] = {}
    h = _SURVIVAL_GRID_S

    def node(i: int) -> float:
        v = cache.get(i)
        if v is None:
            v = cache[i] = float(survival(i * h))
        return v

    def curve(t: float) -> float:
        x = (max(0.0, t) + treatment_s) / h
        i = math.floor(x)
        frac = x - i
        a = node(i)
        return a if frac == 0.0 else a + (node(i + 1) - a) * frac

    return curve


def _survival_value(order: Sequence[int], T, s, p, start: int, curve: Callable[[float], float]) -> float:
    """sum_k p_k * S(arrival_k): weight found while the patient is still savable."""
    t = 0.0
    v = 0.0
    prev = start
    for k in order:
        arrive = t + T[prev][k]
        v += p[k] * curve(arrive)
        t = arrive + s[k]
        prev = k
    return v


def _improve_route_value(
    order: Sequence[int], value: Callable[[Sequence[int]], float], deadline: float
) -> tuple[list[int], float]:
    """2-opt + or-opt local search *maximising* ``value(order)`` (any objective; full
    re-evaluation per move). Accepts only strict improvements."""
    order = list(order)
    n = len(order)
    best = value(order)
    if n < 2:
        return order, best
    span = _LARGE_ROUTE_SPAN if n > 60 else n
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for i in range(n - 1):
            if time.perf_counter() >= deadline:
                break
            for j in range(i + 1, min(n, i + 1 + span)):
                cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                v = value(cand)
                if v > best + 1e-12:
                    order, best, improved = cand, v, True
        for i in range(n):
            if time.perf_counter() >= deadline:
                break
            for j in range(max(0, i - span), min(n, i + 1 + span)):
                if i == j:
                    continue
                cand = order[:i] + order[i + 1 :]
                cand.insert(j, order[i])
                v = value(cand)
                if v > best + 1e-12:
                    order, best, improved = cand, v, True
    return order, value(order)


@dataclass
class _Visits:
    """Index space for assembling a plan: visit ``k`` is a look at ``room_ids[k]``; ``look[k]``
    counts earlier looks *planned in this plan* at that room (0 = first planned look)."""

    room_ids: list[str]
    look: list[int]
    T: list[list[float]]
    s: list[float]
    p: list[float]
    start: int


def _drop_early_repeats(routes: list[list[int]], V: _Visits) -> tuple[list[list[int]], int]:
    """A planned repeat look at a room only counts after the earlier planned look there has
    finished (its weight assumes that look missed). Drop repeats that would start sooner;
    the room stays in the plan through its earlier look. Returns (routes, dropped)."""
    dropped = 0
    while True:
        arrive: dict[int, float] = {}
        finish: dict[int, float] = {}
        for r in routes:
            t, prev = 0.0, V.start
            for k in r:
                a = t + V.T[prev][k]
                t = a + V.s[k]
                arrive[k], finish[k] = a, t
                prev = k
        by_room: dict[str, dict[int, int]] = {}
        for r in routes:
            for k in r:
                by_room.setdefault(V.room_ids[k], {})[V.look[k]] = k
        bad = None
        for room_looks in by_room.values():
            for n_look, k in room_looks.items():
                if n_look == 0:
                    continue
                before = room_looks.get(n_look - 1)
                if before is None or arrive[k] < finish[before] - 1e-9:
                    bad = k
                    break
            if bad is not None:
                break
        if bad is None:
            return routes, dropped
        routes = [[k for k in r if k != bad] for r in routes]
        dropped += 1


def _plan_teams(
    n_teams: int,
    building: Building,
    G: nx.Graph,
    start: str,
    residual: Mapping[str, float],
    cand_ids: Sequence[str],
    closed: Sequence[str],
    coverage: float,
    s_of: Callable[[str], float],
    q_look: float,
    link_s: float,
    deadline: float,
) -> tuple[_Visits, list[list[int]], list[str]]:
    """Sequential team planning (module docstring, step 6). Returns the visit space, one
    route per team (visit indices) and notes.

    Team 1 is planned on the full belief and keeps the stops it finishes by the horizon
    (one-team time to cover every candidate / ``n_teams``); each room it will search is
    multiplied by (1 - q); the next team is planned on what is left (the last team keeps its
    whole route). A room may then appear twice, as a planned repeat look with the reduced
    weight; ``_drop_early_repeats`` keeps that honest. Finally single rooms move between
    routes when that lowers the total cost.
    """
    rooms = {r.id: r for r in building.rooms}
    belief = {rid: float(p) for rid, p in residual.items()}
    planned_looks: dict[str, int] = {}
    team_visits: list[list[tuple[str, int, float]]] = []
    horizon = math.inf
    s_cache: dict[str, float] = {}

    def look_s(rid: str) -> float:
        if rid not in s_cache:
            s_cache[rid] = s_of(rid)
        return s_cache[rid]

    for team in range(n_teams):
        ids = list(cand_ids) if team == 0 else select_candidates(building, belief, coverage, closed)
        if not ids:
            team_visits.append([])
            continue
        travel = pairwise_travel(G, [start, *ids])
        nodes = [*ids, start]
        T = [[_t(travel, a, b) for b in nodes] for a in nodes]
        s = [look_s(r) for r in ids]
        p = [belief[r] for r in ids]
        floors = [rooms[r].floor for r in ids]
        budget = max(0.0, deadline - time.perf_counter()) / (n_teams - team)
        order, _, _ = _solve_single(T, s, p, len(ids), floors, link_s, time.perf_counter() + budget)
        if team < n_teams - 1:
            done = list(completion_times([ids[k] for k in order], travel, dict(zip(ids, s)), start).values())
            if team == 0:
                horizon = done[-1] / n_teams
            keep = max(1, sum(1 for c in done if c <= horizon + 1e-9))
            order = order[:keep]
        visits = []
        for k in order:
            rid = ids[k]
            visits.append((rid, planned_looks.get(rid, 0), p[k]))
            planned_looks[rid] = planned_looks.get(rid, 0) + 1
            belief[rid] = belief[rid] * (1.0 - q_look)
        team_visits.append(visits)

    flat = [v for tv in team_visits for v in tv]
    ids = [v[0] for v in flat]
    travel = pairwise_travel(G, [start, *ids])
    nodes = [*ids, start]
    V = _Visits(
        room_ids=ids,
        look=[v[1] for v in flat],
        T=[[_t(travel, a, b) for b in nodes] for a in nodes],
        s=[look_s(r) for r in ids],
        p=[v[2] for v in flat],
        start=len(ids),
    )
    routes: list[list[int]] = []
    k = 0
    for tv in team_visits:
        routes.append(list(range(k, k + len(tv))))
        k += len(tv)
    routes = _relocate_between_teams(routes, V.T, V.s, V.p, V.start, deadline)
    routes = [_improve_route(r, V.T, V.s, V.p, V.start, deadline)[0] for r in routes]
    routes, dropped = _drop_early_repeats(routes, V)
    notes = [
        f"{n_teams} teams: team 1 planned on the full belief; each room it searches counts at (1 - q) "
        "for the next team, which plans on what is left; then single rooms were moved between routes."
    ]
    if dropped:
        notes.append(f"{dropped} planned repeat look(s) left out: they would start before the first look there ends.")
    return V, routes, notes


def _start_second_team_in(routes: list[list[int]], V: _Visits, wanted: set[str], deadline: float
                          ) -> tuple[list[list[int]], int]:
    """Make team 2 start in the ``wanted`` rooms (module docstring, step 8).

    Nothing changes when there is one team, when some team's first stop is already in a
    wanted room, or when no planned stop is in one. Otherwise team 2's stops in wanted rooms
    come first; when it has none, the other teams' stops there are handed to it (never a
    team's first stop). The block is put in its best order from the start; the rest of each
    route keeps its order. Returns (routes, number of stops put first).
    """
    if len(routes) < 2 or not wanted:
        return routes, 0
    if any(r and V.room_ids[r[0]] in wanted for r in routes):
        return routes, 0
    block = [k for k in routes[1] if V.room_ids[k] in wanted]
    routes = [list(r) for r in routes]
    if not block:
        for a, r in enumerate(routes):
            if a == 1:
                continue
            take = [k for k in r[1:] if V.room_ids[k] in wanted]
            block.extend(take)
            routes[a] = [k for k in r if k not in take]
    if not block:
        return routes, 0
    block, _ = _improve_route(block, V.T, V.s, V.p, V.start, deadline)
    taken = set(block)
    routes[1] = block + [k for k in routes[1] if k not in taken]
    routes, _ = _drop_early_repeats(routes, V)
    return routes, len(block)


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
    profile: Optional[TravelProfile] = None,
    survival: Optional[Callable[[float], float]] = None,
    searched: Sequence[tuple[str, SearchMode]] = (),
    gates: Optional[Gates] = None,
    look_mode: SearchMode = "rapid",
    modes: Optional[Mapping[str, ModeSetting]] = None,
    *,
    crew_size: Optional[int] = None,
    treatment_s: float = 0.0,
    second_team_rooms: Optional[Iterable[str]] = None,
) -> SearchPlan:
    """Build the ordered search plan for the current beliefs and the looks made so far.

    * ``room_probs`` / ``p_nim`` - beliefs *before* any search (evidence only): room id ->
      probability, and the "not in model" mass. Do not pass a posterior that already has
      Koopman updates for ``searched`` applied, or those looks are counted twice.
    * ``searched``  - every look so far as ``(room_id, mode)``, oldest first, repeats
      allowed. A searched room keeps ``p_r * prod (1 - q)`` and may come back as a stop
      with ``kind="re_search"`` when that residual beats untouched rooms.
    * ``exclude``   - room ids the officer closed. Never offered as stops; the graph still
      routes through them (legacy callers pass rooms here).
    * ``gates``     - officer gates. ``closed_room_ids`` are never offered *and* are left out
      of the building graph (no route passes through them; a ``start_node`` among them is
      kept so the crew can leave it). ``closed_entrance_ids`` are neither recommended nor
      routed.
    * ``profile``   - travel rates (``Policy.travel`` / ``situation.travel.profile_for``);
      None = today's constants. With ``search_advance_mps`` set (smoke), look times follow
      the wall-length rule in ``search_time_s``.
    * ``survival``  - ``S(t)``: probability the patient is still savable when the crew
      reaches them ``t`` seconds after *the plan starts* (pass
      ``situation.survival.survival_from(situation, elapsed_s)``). The objective becomes
      ``sum p_find * S(arrival + treatment_s)``, arrival = the stop's ``eta_s``.
    * ``treatment_s`` - added to the arrival time inside ``S``. Default 0: the cardiac curves
      in ``situation.survival`` already include patient-side-to-shock time.
    * ``n_teams``   - teams asked for. A team is a pair: with ``crew_size`` given, at most
      ``crew_size // 2`` teams are planned (never a lone searcher, never more teams than
      asked).
    * ``look_mode`` - mode of every planned look (default ``rapid``).
    * ``modes``     - override the search-mode settings (default: ``search_modes()``).
    * ``entrance_id``  - force an entrance (``ValueError`` if the officer closed it);
      default is ``recommend_approach``'s pick on the residual masses.
    * ``start_node``   - where the team(s) are now (e.g. the room just searched when
      replanning). Defaults to the entrance. All ETAs are relative to this node.
    * ``second_team_rooms`` - rooms on the floors the phone's height fix allows, given when a
      caller statement contradicts that fix (``core.fusion.Fused.phone_band``). With two or
      more teams, team 2's route then starts with its stops in those rooms, unless a team
      already starts there. With one team nothing changes.

    Report fields: ``Stop.kind`` / ``mode`` / ``p_find`` / ``travel_s`` / ``search_s``;
    ``SearchPlan.cumulative_success`` (sum of p_find over the plan, on the scale of
    ``room_probs``, not renormalised), ``p_out`` (outside-the-model probability given the
    looks so far), ``time_to_cover_all_s`` (finish of the last first look; 0 when none),
    ``solver`` (``"exact"`` when the dynamic programme proved the single-team order,
    otherwise ``"heuristic"``), ``objective`` (``"survival"`` when ``survival`` is given).

    The two baselines are single-team routes over the same candidates from the same start,
    scored with the expected-time objective. For ``n_teams == 1`` without a survival function
    the returned E[T] is guaranteed to be <= both baselines.
    """
    asked = max(1, int(n_teams))
    n_teams = asked
    notes_pre: list[str] = []
    if crew_size is not None:
        pairs = max(1, int(crew_size) // 2)
        if pairs < n_teams:
            notes_pre.append(f"Crew of {int(crew_size)} makes {pairs} pair(s): planned {pairs} team(s), not {asked}.")
        n_teams = min(n_teams, pairs)

    closed_ents = list(dict.fromkeys(gates.closed_entrance_ids if gates else ()))
    gate_rooms = list(dict.fromkeys(gates.closed_room_ids if gates else ()))
    G = _graph_for(building, profile, [r for r in gate_rooms if r != start_node], closed_ents)
    rooms = {r.id: r for r in building.rooms}
    table = dict(modes) if modes is not None else search_modes()
    q_look = detection_probability(look_mode, table)
    look_mult = table[look_mode].time_multiplier

    # ---- residual mass after the looks so far --------------------------------
    looks = looks_by_room(searched)
    residual = {
        rid: (float(p) * miss_factor(looks[rid], table) if rid in looks else float(p))
        for rid, p in room_probs.items()
    }
    p_out = outside_probability(room_probs, p_nim, searched, table)
    closed = list(dict.fromkeys([*exclude, *gate_rooms]))

    # ---- entrance + staging ------------------------------------------------
    if entrance_id is None:
        approach = recommend_approach(building, residual, G, profile=profile, gates=gates)
        entrance_id = approach["entrance_id"]
        staging_latlng = tuple(approach["staging_latlng"])
    else:
        if not any(e.get("id") == entrance_id for e in building.entrances):
            raise KeyError(f"unknown entrance id: {entrance_id!r}")
        if entrance_id in closed_ents:
            raise ValueError(f"entrance {entrance_id!r} is closed by the officer in charge")
        spot = staging_for_entrance(building, entrance_id)
        ll = (spot or {}).get("latlng") or (0.0, 0.0)
        staging_latlng = (float(ll[0]), float(ll[1]))

    start = start_node or entrance_id
    if not start or start not in G:
        if start_node:
            raise KeyError(f"unknown start node: {start_node!r}")
        start = building.rooms[0].id if building.rooms else ""

    objective = "survival" if survival is not None else "expected_time"
    notes = notes_pre + [
        "Detection probability per look is a setting with no published basis (unverified): "
        f"{look_mode} {q_look:.2f}."
    ]
    if profile is not None and profile.search_advance_mps:
        notes.append(
            f"Smoke reported: each look takes the room's wall length / {profile.search_advance_mps:g} m/s "
            "(search-advance rate, wall-oriented search), not the clear-air area-based time."
        )
    if n_teams > 1:
        notes.append(NIST_TN_1797_NOTE)
    cand_ids = select_candidates(building, residual, coverage, closed)
    if not cand_ids or not start:
        return SearchPlan(
            teams=[[] for _ in range(n_teams)],
            expected_time_to_find_s=0.0,
            baseline_nearest_s=0.0,
            baseline_highest_prob_s=0.0,
            entrance_id=entrance_id or "",
            staging_latlng=staging_latlng,
            objective=objective,
            p_out=round(p_out, 6),
            cumulative_success=0.0,
            time_to_cover_all_s=0.0,
            notes=notes,
        )

    # ---- travel + look times -------------------------------------------------
    # Optimiser weight w_r = residual_r = p_find_r / q_look. q_look is common to every stop,
    # so the order and E[T] are those of p_find (see the module docstring).
    def look_time(rid: str) -> float:
        return search_time_s(rooms[rid], profile) * look_mult

    travel = pairwise_travel(G, [start, *cand_ids])
    search_times = {rid: look_time(rid) for rid in cand_ids}
    probs = {rid: residual[rid] for rid in cand_ids}
    unreachable = [rid for rid in cand_ids if rid != start and rid not in travel.get(start, {})]
    if unreachable:
        notes.append(
            f"{len(unreachable)} room(s) have no route from the start on the current graph "
            f"(closures or missing connections); their times use a {UNREACHABLE_S:.0f} s placeholder."
        )

    n = len(cand_ids)
    idx = {rid: i for i, rid in enumerate(cand_ids)}
    START = n
    nodes = [*cand_ids, start]
    T = [[_t(travel, a, b) for b in nodes] for a in nodes]
    s = [search_times[rid] for rid in cand_ids]
    p = [probs[rid] for rid in cand_ids]
    floors = [rooms[rid].floor for rid in cand_ids]
    link_s = WING_LINK_M / (profile.walk_mps if profile is not None else 1.2)

    # ---- baselines (single team, same candidates, same start) ---------------
    near_order = [idx[r] for r in nearest_first_order(cand_ids, probs, travel, start)]
    high_order = [idx[r] for r in highest_prob_first_order(cand_ids, probs, travel, start)]
    near_cost = _route_cost(near_order, T, s, p, START)
    high_cost = _route_cost(high_order, T, s, p, START)
    base_mass = sum(p)

    # ---- optimise (linear objective) ----------------------------------------
    deadline = time.perf_counter() + _OPT_BUDGET_S
    if n_teams == 1:
        order, cost, solver = _solve_single(T, s, p, START, floors, link_s, deadline)
        # Guarantee: never worse than either baseline. If one beats the polished route
        # (impossible after the exact solver), polish that baseline and keep the best.
        for b_order, b_cost in ((near_order, near_cost), (high_order, high_cost)):
            if b_cost < cost - _EPS:
                o2, c2 = _improve_route(b_order, T, s, p, START, time.perf_counter() + _SEED_BUDGET_S)
                if c2 > b_cost:  # cannot happen (moves only improve); belt and braces
                    o2, c2 = list(b_order), b_cost
                if c2 < cost:
                    order, cost, solver = o2, c2, "heuristic"
        V = _Visits(list(cand_ids), [0] * n, T, s, p, START)
        routes = [order]
    else:
        solver = "heuristic"

        def s_of(rid: str) -> float:
            return search_times[rid] if rid in search_times else look_time(rid)

        # (a) Event-driven, the original multi-team planner (full time budget): whichever team
        # frees up first takes the best-ratio room.
        ev = _greedy_teams(range(n), T, s, p, START, n_teams)
        ev = [_improve_route(r, T, s, p, START, deadline)[0] for r in ev]
        ev = _relocate_between_teams(ev, T, s, p, START, deadline)
        ev = [_improve_route(r, T, s, p, START, deadline)[0] for r in ev]
        ev_et = sum(_route_cost(r, T, s, p, START) for r in ev) / base_mass
        # (b) Sequential (build plan A2), on its own budget: team 1 on the full belief, team 2 on
        # what is left. The plan with the lower expected time to find is kept (both scored on
        # sum p_find C / sum p_find).
        V, routes, team_notes = _plan_teams(
            n_teams, building, G, start, residual, cand_ids, closed, coverage, s_of, q_look, link_s,
            time.perf_counter() + _OPT_BUDGET_S,
        )
        seq_mass = sum(V.p[k] for r in routes for k in r)
        seq_et = sum(_route_cost(r, V.T, V.s, V.p, V.start) for r in routes) / seq_mass if seq_mass else math.inf
        if ev_et < seq_et - 1e-9:
            V, routes = _Visits(list(cand_ids), [0] * n, T, s, p, START), ev
            notes.append(
                f"{n_teams} teams: whichever team frees up first takes the next best room. Planning team 1 "
                "first and the next team on what it leaves (each room it searches at 1 - q) gave a longer "
                f"expected time to find ({seq_et:.0f} s against {ev_et:.0f} s)."
            )
        else:
            notes.extend(team_notes)

    # ---- caller against phone: team 2 starts on the phone's floor (step 8) -------
    second_first = set(second_team_rooms or ()) if n_teams > 1 else set()
    routes, put_first = _start_second_team_in(routes, V, second_first, time.perf_counter() + _SEED_BUDGET_S)
    if put_first:
        notes.append(
            f"Team 2 starts with {put_first} stop(s) on the floor the phone's height fix points to: a caller "
            "statement contradicts that fix, so team 1 follows the caller's statement and the phone's floor is "
            "not left for later. This order was set by that rule, not by expected time alone."
        )

    # ---- survival objective (non-linear): local search from the linear solution ----
    if survival is not None:
        curve = _survival_curve(survival, float(treatment_s))

        def value(o: Sequence[int]) -> float:
            return _survival_value(o, V.T, V.s, V.p, V.start, curve)

        before = sum(value(r) for r in routes)
        sdeadline = time.perf_counter() + _OPT_BUDGET_S
        routes = [_improve_route_value(r, value, sdeadline)[0] for r in routes]
        if n_teams > 1:
            routes, _ = _drop_early_repeats(routes, V)
        routes, _ = _start_second_team_in(routes, V, second_first, time.perf_counter() + _SEED_BUDGET_S)
        after = sum(value(r) for r in routes)
        found = sum(V.p[k] for r in routes for k in r)
        seed_desc = "the exact linear order" if solver == "exact" else "the linear (expected-time) plan"
        notes.append(
            "Objective: chance of reaching the caller while still savable, sum p_find x S(arrival"
            + (f" + {treatment_s:g} s" if treatment_s else "")
            + "). The exact solver does not apply to a non-linear survival curve (elapsed time is "
            f"not a function of its state), so this order starts from {seed_desc} and was improved "
            "by local search on the survival objective."
        )
        if found > 0:
            notes.append(
                f"Expected survival if the caller is found by this plan: {after / found:.1%} "
                f"(expected-time order: {before / found:.1%})."
            )
        solver = "heuristic"
    cost = sum(_route_cost(r, V.T, V.s, V.p, V.start) for r in routes)
    mass = sum(V.p[k] for r in routes for k in r) or base_mass

    # ---- assemble ------------------------------------------------------------
    teams: list[list[Stop]] = []
    success = 0.0
    cover_all = 0.0
    n_re = 0
    for route in routes:
        stops: list[Stop] = []
        t = 0.0
        cum = 0.0
        prev = V.start
        for k in route:
            leg = V.T[prev][k]
            arrive = t + leg
            t = arrive + V.s[k]
            cum += V.p[k]
            room = rooms[V.room_ids[k]]
            kind = "re_search" if (room.id in looks or V.look[k] > 0) else "first"
            pf = V.p[k] * q_look
            success += pf
            if kind == "first":
                cover_all = max(cover_all, t)
            else:
                n_re += 1
            stops.append(
                Stop(
                    room_id=room.id,
                    room_name=room.name,
                    floor_label=room.floor_label,
                    eta_s=round(arrive, 1),
                    cumulative_prob=round(cum, 6),
                    kind=kind,
                    mode=look_mode,
                    p_find=round(pf, 8),
                    travel_s=round(leg, 1),
                    search_s=round(V.s[k], 1),
                )
            )
            prev = k
        teams.append(stops)
    if n_re:
        notes.append(f"{n_re} repeat look(s) offered: the probability left in those rooms "
                     "still beats rooms not yet searched.")
    if solver == "exact":
        notes.append(f"Order proven optimal for the expected-time objective (exact solver, {n} rooms).")

    return SearchPlan(
        teams=teams,
        expected_time_to_find_s=round(cost / mass, 2),
        baseline_nearest_s=round(near_cost / base_mass, 2),
        baseline_highest_prob_s=round(high_cost / base_mass, 2),
        entrance_id=entrance_id or "",
        staging_latlng=staging_latlng,
        objective=objective,
        solver=solver,
        p_out=round(p_out, 6),
        cumulative_success=round(success, 6),
        time_to_cover_all_s=round(cover_all, 1),
        notes=notes,
    )

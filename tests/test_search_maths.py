"""Search maths: detection probability per mode, residual mass, re-search stops and the plan
report fields (A1); exact solver, cluster check, survival objective, teams, travel profile and
gates in the planner (A2)."""

from __future__ import annotations

import random

import pytest

from core.schema import Building, Gates, Room
from ingest.matterport import load_building
from planning.search_order import (
    ModeSetting,
    _greedy_single,
    _improve_route,
    _t,
    expected_time,
    koopman_update,
    outside_probability,
    p_find,
    pairwise_travel,
    plan_search,
    residual_probs,
    search_modes,
    search_time_s,
    select_candidates,
    _graph_for,
)
from situation.config import section


def _room(rid: str, x: float, area: float = 20.0) -> Room:
    return Room(
        id=rid, name=rid.upper(), floor=1, floor_label="1", polygon=[(x - 1, -1), (x + 1, -1), (x + 1, 1), (x - 1, 1)],
        centroid=(x, 0.0, 0.0), area_m2=area, room_type="office", has_windows=False, near_elevator=False,
        near_stairs=False, occupancy_factor=1.0,
    )


def _star(n_rooms: int = 2) -> Building:
    """A hall with ``n_rooms`` identical rooms hung off it at the same distance: equal costs."""
    rooms = [_room("hall", 0.0, area=10.0)] + [_room(f"r{i}", 0.0) for i in range(1, n_rooms + 1)]
    edges = [{"u": "hall", "v": f"r{i}", "kind": "door", "length_m": 6.0} for i in range(1, n_rooms + 1)]
    return Building(
        id="star", name="Star", use_type="office", floors=[1], floor_heights_m={1: 4.0}, rooms=rooms,
        edges=edges, entrances=[{"id": "e1", "room_id": "hall", "length_m": 3.0}], staging_spots=[],
    )


@pytest.fixture(scope="module")
def ptc():
    return load_building("ptc-100")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_search_modes_come_from_config_and_are_unverified() -> None:
    modes = search_modes()
    assert modes["rapid"].detection_probability == pytest.approx(0.9)
    assert modes["rapid"].time_multiplier == pytest.approx(1.0)
    assert modes["thorough"].detection_probability == pytest.approx(0.97)
    assert modes["thorough"].time_multiplier == pytest.approx(2.0)
    cfg = section("search")
    for mode in ("rapid", "thorough"):
        for key in ("detection_probability", "time_multiplier"):
            assert cfg["modes"][mode][key]["label"] == "unverified"
    for key in ("search_s_per_m2", "min_search_s"):
        assert cfg[key]["label"] == "unverified"


# ---------------------------------------------------------------------------
# Posterior and outside mass
# ---------------------------------------------------------------------------


def test_exact_koopman_update_includes_the_outside_mass() -> None:
    probs = {"r1": 0.50, "r2": 0.30, "r3": 0.15}
    after, p_out = koopman_update(probs, 0.05, "r1", 0.9)
    assert (after["r1"], after["r2"], after["r3"], p_out) == pytest.approx((0.091, 0.545, 0.273, 0.091), abs=5e-4)
    # The same update through the mode table (rapid q = 0.9) and the closed form.
    after2, p_out2 = residual_probs(probs, 0.05, [("r1", "rapid")])
    assert after2 == pytest.approx(after) and p_out2 == pytest.approx(p_out)
    assert outside_probability(probs, 0.05, [("r1", "rapid")]) == pytest.approx(p_out)


def test_outside_probability_after_one_full_pass(ptc) -> None:
    ids = [r.id for r in ptc.rooms]
    probs = {rid: 0.95 / len(ids) for rid in ids}
    searched = [(rid, "rapid") for rid in ids]
    assert outside_probability(probs, 0.05, searched) == pytest.approx(0.05 / 0.145, abs=1e-9)
    assert 0.05 / 0.145 == pytest.approx(0.345, abs=5e-4)
    plan = plan_search(ptc, probs, 0.05, searched=searched)
    assert plan.p_out == pytest.approx(0.345, abs=5e-4)
    # Searched rooms stay candidates: the whole plan is re-search looks.
    stops = plan.teams[0]
    assert stops and all(s.kind == "re_search" for s in stops)
    assert plan.time_to_cover_all_s == 0.0


def test_p_find_uses_every_earlier_look() -> None:
    assert p_find(0.5) == pytest.approx(0.45)
    assert p_find(0.5, ["rapid"]) == pytest.approx(0.5 * 0.1 * 0.9)
    assert p_find(0.5, ["rapid", "thorough"], mode="thorough") == pytest.approx(0.5 * 0.1 * 0.03 * 0.97)


# ---------------------------------------------------------------------------
# Re-search
# ---------------------------------------------------------------------------


def test_searched_room_with_more_left_comes_back_before_an_untouched_room() -> None:
    b = _star(2)
    probs = {"r1": 0.60, "r2": 0.02}
    plan = plan_search(b, probs, 0.38, entrance_id="e1", searched=[("r1", "rapid")])
    stops = plan.teams[0]
    assert [s.room_id for s in stops] == ["r1", "r2"]
    assert stops[0].kind == "re_search" and stops[1].kind == "first"
    assert stops[0].p_find == pytest.approx(0.60 * 0.1 * 0.9)
    assert stops[1].p_find == pytest.approx(0.02 * 0.9)
    # Two unsuccessful looks leave 0.006 < 0.02: the untouched room goes first.
    plan2 = plan_search(b, probs, 0.38, entrance_id="e1", searched=[("r1", "rapid"), ("r1", "rapid")])
    assert [s.room_id for s in plan2.teams[0]] == ["r2", "r1"]


def test_gates_and_exclude_close_rooms_searched_does_not() -> None:
    b = _star(3)
    probs = {"r1": 0.5, "r2": 0.3, "r3": 0.1}
    searched = [("r1", "rapid")]
    ids = [s.room_id for s in plan_search(b, probs, 0.1, entrance_id="e1", searched=searched).teams[0]]
    assert sorted(ids) == ["r1", "r2", "r3"]
    gated = plan_search(b, probs, 0.1, entrance_id="e1", searched=searched, gates=Gates(closed_room_ids=["r2"]))
    assert "r2" not in [s.room_id for s in gated.teams[0]]
    excl = plan_search(b, probs, 0.1, entrance_id="e1", exclude=["r3"])
    assert "r3" not in [s.room_id for s in excl.teams[0]]


# ---------------------------------------------------------------------------
# Report fields
# ---------------------------------------------------------------------------


def test_report_fields_and_look_times() -> None:
    b = _star(3)
    probs = {"r1": 0.5, "r2": 0.3, "r3": 0.1}
    plan = plan_search(b, probs, 0.1, entrance_id="e1")
    stops = plan.teams[0]
    # Three candidates: the exact solver proves the order.
    assert plan.solver == "exact" and plan.objective == "expected_time"
    assert plan.p_out == pytest.approx(0.1)
    assert plan.cumulative_success == pytest.approx(0.9 * 0.9)
    assert sum(s.p_find for s in stops) == pytest.approx(plan.cumulative_success)
    # Times: entrance->hall 3 m (2.5 s), hall->room 6 m (5 s), room->room 12 m (10 s); 20 m2 -> 10 s look.
    assert [s.travel_s for s in stops] == pytest.approx([7.5, 10.0, 10.0])
    assert all(s.search_s == pytest.approx(10.0) for s in stops)
    finish = stops[-1].eta_s + stops[-1].search_s
    assert plan.time_to_cover_all_s == pytest.approx(finish)
    assert any("unverified" in n for n in plan.notes)

    thorough = plan_search(b, probs, 0.1, entrance_id="e1", look_mode="thorough").teams[0]
    assert all(s.mode == "thorough" and s.search_s == pytest.approx(20.0) for s in thorough)
    assert sum(s.p_find for s in thorough) == pytest.approx(0.9 * 0.97)


def test_plan_never_worse_than_baselines_with_searched_rooms(ptc) -> None:
    rng = random.Random(3)
    ids = [r.id for r in ptc.rooms]
    for _ in range(4):
        w = [rng.random() ** 3 for _ in ids]
        tot = sum(w)
        probs = {rid: 0.9 * x / tot for rid, x in zip(ids, w)}
        top = sorted(ids, key=lambda r: -probs[r])[:4]
        searched = [(top[0], "rapid"), (top[1], "thorough"), (top[0], "rapid")]
        plan = plan_search(ptc, probs, 0.1, searched=searched)
        assert plan.expected_time_to_find_s <= plan.baseline_nearest_s + 1e-6
        assert plan.expected_time_to_find_s <= plan.baseline_highest_prob_s + 1e-6
        assert plan.p_out == pytest.approx(outside_probability(probs, 0.1, searched), abs=1e-6)


# ---------------------------------------------------------------------------
# First pass with a uniform q equals the original planner
# ---------------------------------------------------------------------------


def _old_order(building, probs, start, coverage=0.99):
    """The original planner's single-team route: greedy p/(t+s) on raw p + local search."""
    rooms = {r.id: r for r in building.rooms}
    cands = select_candidates(building, probs, coverage)
    travel = pairwise_travel(_graph_for(building), [start, *cands])
    nodes = [*cands, start]
    T = [[_t(travel, a, b) for b in nodes] for a in nodes]
    s = [search_time_s(rooms[r]) for r in cands]
    p = [probs[r] for r in cands]
    order, _ = _improve_route(_greedy_single(range(len(cands)), T, s, p, len(cands)), T, s, p, len(cands), 1e18)
    return [cands[k] for k in order], travel, {r: search_time_s(rooms[r]) for r in cands}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_pass_order_unchanged_with_uniform_q(ptc, seed, monkeypatch) -> None:
    """With the walking penalty off (scale 0) the planner is the original one. (With it on,
    lambda depends on q, so a different q can change the order.)"""
    import planning.search_order as so

    monkeypatch.setattr(so, "walking_penalty_scale", lambda: 0.0)
    rng = random.Random(seed)
    ids = [r.id for r in ptc.rooms]
    w = [rng.random() ** 4 for _ in ids]
    tot = sum(w)
    probs = {rid: 0.95 * x / tot for rid, x in zip(ids, w)}
    plan = plan_search(ptc, probs, 0.05)
    got = [s.room_id for s in plan.teams[0]]
    assert all(s.kind == "first" and s.mode == "rapid" for s in plan.teams[0])

    start = plan.entrance_id
    old, travel, st = _old_order(ptc, probs, start)
    old_et = expected_time(old, probs, travel, st, start)
    new_et = expected_time(got, probs, travel, st, start)
    # Same route, or (if the baseline guard picked another) one at least as good under the old objective.
    assert got == old or new_et <= old_et + 1e-9
    assert plan.expected_time_to_find_s == pytest.approx(new_et, abs=0.01)

    # Any uniform q (here q = 1: p_find = p, the original objective exactly) gives the same order.
    for q in (1.0, 0.5):
        modes = {"rapid": ModeSetting(q, 1.0), "thorough": ModeSetting(0.97, 2.0)}
        other = plan_search(ptc, probs, 0.05, modes=modes)
        assert [s.room_id for s in other.teams[0]] == got
        assert other.expected_time_to_find_s == pytest.approx(plan.expected_time_to_find_s)
    assert plan.cumulative_success == pytest.approx(0.9 * sum(probs[r] for r in got), abs=1e-6)


# ===========================================================================
# A2: exact solver, cluster check, survival objective, teams, profile and gates
# ===========================================================================

import itertools  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

from core.schema import TravelProfile  # noqa: E402
from planning.exact import EXACT_MAX_ROOMS, solve_exact, time_exact  # noqa: E402
from planning.search_order import (  # noqa: E402
    NIST_TN_1797_NOTE,
    _cluster_seed,
    _drop_early_repeats,
    _improve_route_value,
    _plan_teams,
    _route_cost,
    _solve_single,
    _survival_curve,
    _survival_value,
    _Visits,
    wall_length_m,
)


def _random_instance(rng: random.Random, n: int):
    pts = [(rng.uniform(0, 60), rng.uniform(0, 60)) for _ in range(n + 1)]
    T = [[math.dist(a, b) / 1.2 for b in pts] for a in pts]
    s = [rng.uniform(10, 40) for _ in range(n)]
    p = [rng.random() ** 2 for _ in range(n)]
    return T, s, p


def _decoys_and_wing():
    """The constructed case from the build plan (A2), in seconds: five decoy rooms 1 min from
    the start and 2 min apart (1% each), a far wing 10 min away with 20 rooms at negligible
    spacing (4.75% each). Rooms 0-4 decoys, 5-24 wing, start = 25; look times negligible."""
    n = 25
    T = [[0.0] * (n + 1) for _ in range(n + 1)]
    decoys, wing, start = range(5), range(5, 25), 25
    for d in decoys:
        T[start][d] = T[d][start] = 60.0
        for e in decoys:
            if e != d:
                T[d][e] = 120.0
        for w in wing:
            T[d][w] = T[w][d] = 660.0
    for w in wing:
        T[start][w] = T[w][start] = 600.0
    s = [0.0] * n
    p = [0.01] * 5 + [0.0475] * 20
    floors = [1] * n
    return T, s, p, start, floors


def test_exact_solver_matches_brute_force() -> None:
    rng = random.Random(11)
    for _ in range(3):
        T, s, p = _random_instance(rng, 6)
        order, cost = solve_exact(T, s, p, 6)
        brute = min(_route_cost(o, T, s, p, 6) for o in itertools.permutations(range(6)))
        assert sorted(order) == list(range(6))
        assert cost == pytest.approx(brute, rel=1e-9)
        assert _route_cost(order, T, s, p, 6) == pytest.approx(brute, rel=1e-9)


def test_exact_solver_is_fast_enough_at_its_limit() -> None:
    assert EXACT_MAX_ROOMS == 12
    assert time_exact(EXACT_MAX_ROOMS) < 150.0  # ms, median of three random instances


def test_decoy_case_goes_to_the_wing_first() -> None:
    T, s, p, start, floors = _decoys_and_wing()
    # The density rule clears every decoy first: 19.25 min.
    greedy = _greedy_single(range(25), T, s, p, start)
    assert greedy[:5] == [0, 1, 2, 3, 4]
    assert _route_cost(greedy, T, s, p, start) / sum(p) / 60.0 == pytest.approx(19.25)
    # The cluster check scores the wing above every decoy and seeds the route with it.
    seed = _cluster_seed(T, s, p, start, floors, link_s=25.0 / 1.2)
    assert set(seed[:20]) == set(range(5, 25))
    order, cost, solver = _solve_single(T, s, p, start, floors, 25.0 / 1.2, time.perf_counter() + 1.0)
    assert set(order[:20]) == set(range(5, 25))
    assert cost / sum(p) / 60.0 == pytest.approx(10.75)
    assert solver == "heuristic"  # 25 rooms: above the exact limit


@pytest.mark.parametrize("seed", range(5))
def test_exact_and_heuristic_agree_on_eight_rooms(seed) -> None:
    T, s, p = _random_instance(random.Random(100 + seed), 8)
    _, exact_cost = solve_exact(T, s, p, 8)
    _, heur_cost, solver = _solve_single(T, s, p, 8, [1] * 8, 20.0, time.perf_counter() + 1.0, exact_ok=False)
    assert solver == "heuristic"
    assert exact_cost <= heur_cost + 1e-9
    assert heur_cost <= 1.01 * exact_cost


def test_small_candidate_sets_report_the_exact_solver(ptc) -> None:
    ids = [r.id for r in ptc.rooms]
    rng = random.Random(5)
    picked = set(rng.sample(ids, 9))
    raw = {rid: (0.1 + rng.random() if rid in picked else 0.0) for rid in ids}
    tot = sum(raw.values())
    probs = {rid: 0.9 * v / tot for rid, v in raw.items()}
    plan = plan_search(ptc, probs, 0.1, coverage=1.0)
    assert plan.solver == "exact" and len(plan.teams[0]) == 9
    assert plan.expected_time_to_find_s <= min(plan.baseline_nearest_s, plan.baseline_highest_prob_s) + 1e-6
    assert any("exact solver" in n for n in plan.notes)


# ---------------------------------------------------------------------------
# Survival objective
# ---------------------------------------------------------------------------


def _random_probs(building, seed: int, power: float = 4.0, mass: float = 0.95) -> dict[str, float]:
    rng = random.Random(seed)
    ids = [r.id for r in building.rooms]
    w = [rng.random() ** power for _ in ids]
    tot = sum(w)
    return {rid: mass * x / tot for rid, x in zip(ids, w)}


def _s_linear(t: float) -> float:
    return 1.0 - t / 100_000.0


@pytest.mark.parametrize("seed", [0, 1])
def test_linear_survival_gives_the_linear_order(ptc, seed) -> None:
    probs = _random_probs(ptc, seed)
    linear = plan_search(ptc, probs, 0.05)
    surv = plan_search(ptc, probs, 0.05, survival=_s_linear)
    assert [st.room_id for st in surv.teams[0]] == [st.room_id for st in linear.teams[0]]
    assert surv.objective == "survival" and surv.solver == "heuristic"
    assert any("does not apply to a non-linear survival curve" in n for n in surv.notes)


def test_linear_survival_gives_the_exact_order_on_a_small_set() -> None:
    b = _star(4)
    small = {"r1": 0.4, "r2": 0.3, "r3": 0.2, "r4": 0.05}
    a = plan_search(b, small, 0.05, entrance_id="e1")
    c = plan_search(b, small, 0.05, entrance_id="e1", survival=_s_linear)
    assert a.solver == "exact"
    assert [x.room_id for x in c.teams[0]] == [x.room_id for x in a.teams[0]]
    assert any("starts from the exact linear order" in n for n in c.notes)


def test_steep_survival_never_lowers_the_survival_objective(ptc) -> None:
    probs = _random_probs(ptc, 4)

    def s_steep(t: float) -> float:
        return math.exp(-t / 120.0)

    def value(plan) -> float:
        return sum(st.p_find * s_steep(st.eta_s) for st in plan.teams[0])

    linear = plan_search(ptc, probs, 0.05)
    surv = plan_search(ptc, probs, 0.05, survival=s_steep)
    assert value(surv) >= value(linear) - 1e-4
    later = plan_search(ptc, probs, 0.05, survival=s_steep, treatment_s=60.0)
    assert any("+ 60 s" in n for n in later.notes)


def test_survival_cliff_changes_the_order() -> None:
    """The linear objective goes to the far, likely room first; a curve that is gone by then
    makes the near room worth more."""
    # rooms: 0 near (10 s, 5%), 1 far (100 s, 95%); 0 <-> 1 is 100 s; looks 10 s.
    T = [[0.0, 100.0, 10.0], [100.0, 0.0, 100.0], [10.0, 100.0, 0.0]]
    s, p, start = [10.0, 10.0], [0.05, 0.95], 2
    order, _, _ = _solve_single(T, s, p, start, [1, 1], 20.0, time.perf_counter() + 1.0)
    assert order == [1, 0]
    curve = _survival_curve(lambda t: 1.0 / (1.0 + math.exp((t - 60.0) / 3.0)), 0.0)
    better, v = _improve_route_value(
        order, lambda o: _survival_value(o, T, s, p, start, curve), time.perf_counter() + 1.0
    )
    assert better == [0, 1]
    assert v > _survival_value(order, T, s, p, start, curve)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def test_teams_are_pairs_and_never_more_than_asked(ptc) -> None:
    probs = _random_probs(ptc, 2)
    three = plan_search(ptc, probs, 0.05, n_teams=2, crew_size=3)
    assert len(three.teams) == 1
    assert any("Crew of 3 makes 1 pair" in n for n in three.notes)
    six = plan_search(ptc, probs, 0.05, n_teams=2, crew_size=6)
    assert len(six.teams) == 2 and NIST_TN_1797_NOTE in six.notes
    assert len(plan_search(ptc, probs, 0.05, n_teams=2, crew_size=10).teams) == 2
    assert "1.55" in NIST_TN_1797_NOTE and "NIST TN 1797" in NIST_TN_1797_NOTE


@pytest.mark.parametrize("n_teams", [2, 3])
def test_team_plan_covers_every_candidate_and_repeats_come_late(ptc, n_teams) -> None:
    probs = _random_probs(ptc, 6)
    plan = plan_search(ptc, probs, 0.05, n_teams=n_teams)
    assert len(plan.teams) == n_teams and all(plan.teams)
    first = [st.room_id for t in plan.teams for st in t if st.kind == "first"]
    assert len(first) == len(set(first))
    assert set(select_candidates(ptc, probs, 0.99)) <= set(first)
    finish = {st.room_id: st.eta_s + st.search_s for t in plan.teams for st in t if st.kind == "first"}
    for t in plan.teams:
        for st in t:
            if st.kind == "re_search":
                assert st.eta_s >= finish[st.room_id] - 0.2


def test_sequential_team_planning_discounts_team_one_rooms(ptc) -> None:
    probs = _random_probs(ptc, 8)
    G = _graph_for(ptc)
    start = plan_search(ptc, probs, 0.05).entrance_id
    rooms = {r.id: r for r in ptc.rooms}
    cands = select_candidates(ptc, probs, 0.99)
    V, routes, notes = _plan_teams(
        2, ptc, G, start, probs, cands, [], 0.99, lambda rid: search_time_s(rooms[rid]), 0.9, 20.0,
        time.perf_counter() + 1.0,
    )
    assert len(routes) == 2 and routes[0] and routes[1]
    for k in (k for r in routes for k in r):
        assert V.p[k] == pytest.approx(probs[V.room_ids[k]] * (0.1 ** V.look[k]))
    assert "on what is left" in notes[0]


def test_repeat_look_before_the_first_look_ends_is_dropped() -> None:
    # visit 0: room a, first look (team 1 arrives at 100 s, 10 s look); visit 1: room a again
    # (team 2) arriving at 50 s -> dropped; visit 2: room b, untouched. Start = index 3.
    T = [[0.0, 0.0, 5.0, 100.0], [0.0, 0.0, 5.0, 50.0], [5.0, 5.0, 0.0, 5.0], [100.0, 50.0, 5.0, 0.0]]
    V = _Visits(["a", "a", "b"], [0, 1, 0], T, [10.0, 10.0, 10.0], [0.5, 0.05, 0.3], 3)
    routes, dropped = _drop_early_repeats([[0], [1, 2]], V)
    assert dropped == 1 and routes == [[0], [2]]
    # Arriving after the first look ends, the repeat stays.
    T2 = [row[:] for row in T]
    T2[3][1] = T2[1][3] = 200.0
    V2 = _Visits(["a", "a", "b"], [0, 1, 0], T2, [10.0, 10.0, 10.0], [0.5, 0.05, 0.3], 3)
    assert _drop_early_repeats([[0], [1, 2]], V2) == ([[0], [1, 2]], 0)


# ---------------------------------------------------------------------------
# Travel profile, smoke look times, gates (B1 hooks in the planner)
# ---------------------------------------------------------------------------


def _detour() -> Building:
    """a - b - t in a line (10 m each), plus a long a - t corridor (40 m). Entrance e1 opens
    into b, e2 into a (1 m each)."""
    rooms = [_room("a", 0.0), _room("b", 10.0), _room("t", 20.0)]
    edges = [
        {"u": "a", "v": "b", "kind": "door", "length_m": 10.0},
        {"u": "b", "v": "t", "kind": "door", "length_m": 10.0},
        {"u": "a", "v": "t", "kind": "door", "length_m": 40.0},
    ]
    ents = [{"id": "e1", "room_id": "b", "length_m": 1.0}, {"id": "e2", "room_id": "a", "length_m": 1.0}]
    return Building(
        id="detour", name="Detour", use_type="office", floors=[1], floor_heights_m={1: 4.0}, rooms=rooms,
        edges=edges, entrances=ents, staging_spots=[],
    )


def test_default_profile_gives_todays_plan(ptc) -> None:
    probs = _random_probs(ptc, 9)
    a = plan_search(ptc, probs, 0.05)
    b = plan_search(ptc, probs, 0.05, profile=TravelProfile(), gates=Gates())
    assert [s.room_id for s in a.teams[0]] == [s.room_id for s in b.teams[0]]
    assert a.expected_time_to_find_s == b.expected_time_to_find_s and a.entrance_id == b.entrance_id


def test_gates_close_entrances_and_routes() -> None:
    b = _detour()
    probs = {"t": 0.9}
    open_plan = plan_search(b, probs, 0.1)
    assert open_plan.entrance_id == "e1"
    assert open_plan.teams[0][0].travel_s == pytest.approx(11.0 / 1.2, abs=0.05)
    no_e1 = plan_search(b, probs, 0.1, gates=Gates(closed_entrance_ids=["e1"]))
    assert no_e1.entrance_id == "e2"
    assert no_e1.teams[0][0].travel_s == pytest.approx(21.0 / 1.2, abs=0.05)
    # Room b closed: no route through it (and e1, which opens into b, goes with it).
    no_b = plan_search(b, probs, 0.1, gates=Gates(closed_room_ids=["b"]))
    assert no_b.entrance_id == "e2"
    assert no_b.teams[0][0].travel_s == pytest.approx(41.0 / 1.2, abs=0.05)
    # exclude keeps its old meaning: never a stop, still walkable.
    excl = plan_search(b, {"t": 0.5, "b": 0.4}, 0.1, exclude=["b"])
    assert [s.room_id for s in excl.teams[0]] == ["t"] and excl.entrance_id == "e1"
    with pytest.raises(ValueError):
        plan_search(b, probs, 0.1, entrance_id="e1", gates=Gates(closed_entrance_ids=["e1"]))


def test_graph_cache_is_keyed_on_profile_and_gates(ptc) -> None:
    g0 = _graph_for(ptc)
    assert _graph_for(ptc) is g0
    some_room = ptc.rooms[5].id
    g1 = _graph_for(ptc, None, [some_room])
    assert g1 is not g0 and some_room not in g1 and some_room in g0
    assert _graph_for(ptc, TravelProfile(stair_s_per_floor=29.0)) is not g0
    assert _graph_for(ptc, TravelProfile(notes=["x"])) is _graph_for(ptc, TravelProfile())


def test_closed_entrance_never_chosen_on_ptc(ptc) -> None:
    probs = _random_probs(ptc, 1)
    chosen = plan_search(ptc, probs, 0.05).entrance_id
    other = plan_search(ptc, probs, 0.05, gates=Gates(closed_entrance_ids=[chosen]))
    assert other.entrance_id and other.entrance_id != chosen


def test_smoke_look_time_uses_wall_length_and_search_advance() -> None:
    rect = Room(
        id="r", name="R", floor=1, floor_label="1", polygon=[(0, 0), (4, 0), (4, 5), (0, 5)],
        centroid=(2.0, 2.5, 0.0), area_m2=20.0, room_type="office", has_windows=False,
        near_elevator=False, near_stairs=False, occupancy_factor=1.0,
    )
    assert wall_length_m(rect) == pytest.approx(18.0)
    smoke = TravelProfile(walk_mps=0.5, search_advance_mps=0.040)
    assert search_time_s(rect, smoke) == pytest.approx(18.0 / 0.040)
    assert search_time_s(rect) == pytest.approx(10.0)  # clear air: 0.5 s per m2
    # A polygon that disagrees with the recorded area: square-room wall length 4 sqrt(A).
    odd = _room("x", 0.0, area=20.0)  # 2 m x 2 m polygon, 20 m2 recorded
    assert wall_length_m(odd) == pytest.approx(4.0 * math.sqrt(20.0))

    b = _star(2)
    plan = plan_search(b, {"r1": 0.6, "r2": 0.3}, 0.1, entrance_id="e1", profile=smoke)
    assert all(s.search_s == pytest.approx(4.0 * math.sqrt(20.0) / 0.040, abs=0.05) for s in plan.teams[0])
    assert any("Smoke reported" in n and "0.04 m/s" in n for n in plan.notes)
    # Moving at the smoke rate: entrance 3 m + hall->room 6 m at 0.5 m/s.
    assert plan.teams[0][0].travel_s == pytest.approx(9.0 / 0.5, abs=0.05)


# ===========================================================================
# Walking penalty: sum p C + lam * T_total (planning.search_order, "Walking penalty")
# ===========================================================================


def _corridor(pl, pr, gap=10.0, look=15.0):
    """Rooms every ``gap`` m left and right of an entrance at x = 0, on one corridor."""
    xs = [-(i + 1) * gap for i in range(len(pl))] + [(i + 1) * gap for i in range(len(pr))] + [0.0]
    T = [[abs(a - b) / 1.2 for b in xs] for a in xs]
    return xs, T, [look] * (len(xs) - 1), list(pl) + list(pr)


def _walk_s(order, T, start):
    t, prev = 0.0, start
    for k in order:
        t, prev = t + T[prev][k], k
    return t


def test_zero_penalty_is_the_original_planner() -> None:
    rng = random.Random(7)
    for _ in range(5):
        n = 9
        pts = [(rng.uniform(0, 60), rng.uniform(0, 30)) for _ in range(n + 1)]
        T = [[math.dist(a, b) / 1.2 for b in pts] for a in pts]
        s = [rng.uniform(10, 40) for _ in range(n)]
        p = [rng.random() for _ in range(n)]
        assert solve_exact(T, s, p, n, 0.0) == solve_exact(T, s, p, n)
        g = _greedy_single(range(n), T, s, p, n)
        assert _greedy_single(range(n), T, s, p, n, 0.0) == g
        assert _improve_route(g, T, s, p, n, 1e18, 0.0) == _improve_route(g, T, s, p, n, 1e18)
        assert _route_cost(g, T, s, p, n, 0.0) == _route_cost(g, T, s, p, n)


def test_penalty_sweeps_a_corridor_the_old_objective_crosses_twice() -> None:
    """Left side likely near the door, right side tailing off: E[T] alone finishes the left
    end and then walks back across the floor for the last two rooms on the right. With the
    penalty the right side is finished first and the floor is crossed once."""
    xs, T, s, p = _corridor([.30, .05, .04, .03, .02, .01], [.25, .20, .04, .03, .02, .01])
    n = len(p)
    lam = (1.0 - 0.9 * sum(p)) / 0.9            # the phantom stop: not found by the path, / q
    old, _ = solve_exact(T, s, p, n)
    new, _ = solve_exact(T, s, p, n, lam)
    old_x, new_x = [xs[k] for k in old], [xs[k] for k in new]
    assert old_x[-2:] == [50.0, 60.0] and -60.0 in old_x[:-2]   # the walk back across
    assert new_x == [-10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, -20.0, -30.0, -40.0, -50.0, -60.0]
    assert _walk_s(new, T, n) < 0.75 * _walk_s(old, T, n)
    et_old = _route_cost(old, T, s, p, n) / sum(p)
    et_new = _route_cost(new, T, s, p, n) / sum(p)
    assert et_old <= et_new <= 1.05 * et_old       # a small price in E[T]
    # the heuristic path (greedy + local search) finds the same sweep
    h, _ = _improve_route(_greedy_single(range(n), T, s, p, n, lam), T, s, p, n, 1e18, lam)
    assert _route_cost(h, T, s, p, n, lam) == pytest.approx(_route_cost(new, T, s, p, n, lam), rel=1e-9)


@pytest.mark.parametrize("lam", [0.05, 0.3, 2.0])
def test_exact_solver_matches_brute_force_with_the_penalty(lam) -> None:
    rng = random.Random(11)
    for _ in range(3):
        pts = [(rng.uniform(0, 50), rng.uniform(0, 50)) for _ in range(7)]
        T = [[math.dist(a, b) / 1.2 for b in pts] for a in pts]
        s = [rng.uniform(5, 30) for _ in range(6)]
        p = [rng.random() for _ in range(6)]
        order, cost = solve_exact(T, s, p, 6, lam)
        brute = min(_route_cost(o, T, s, p, 6, lam) for o in itertools.permutations(range(6)))
        assert cost == pytest.approx(brute, rel=1e-9)
        assert _route_cost(order, T, s, p, 6, lam) == pytest.approx(brute, rel=1e-9)


def test_survival_penalty_is_a_phantom_stop_at_the_end() -> None:
    from planning.search_order import _survival_value

    xs, T, s, p = _corridor([.3, .1], [.2, .1])
    curve = lambda t: 1.0 / (1.0 + math.exp(-0.5 + t / 120.0))  # noqa: E731 - any falling curve
    order = [0, 2, 1, 3]
    end = sum(T[a][b] for a, b in zip([4, *order], order)) + sum(s)
    plain = _survival_value(order, T, s, p, 4, curve)
    assert _survival_value(order, T, s, p, 4, curve, 0.0) == plain
    assert _survival_value(order, T, s, p, 4, curve, 0.4) == pytest.approx(plain + 0.4 * curve(end))


def test_plan_reports_the_penalty_and_keeps_e_t_comparable(ptc, monkeypatch) -> None:
    import planning.search_order as so

    from core.prior import compute_prior

    probs, p0 = compute_prior(ptc, hour=14)
    on = plan_search(ptc, probs, p0)
    monkeypatch.setattr(so, "walking_penalty_scale", lambda: 0.0)
    off = plan_search(ptc, probs, p0)
    assert any(n.startswith("Walking penalty applied: lambda = ") for n in on.notes)
    assert not any(n.startswith("Walking penalty") for n in off.notes)
    walk = lambda plan: sum(st.travel_s for st in plan.teams[0])  # noqa: E731
    assert walk(on) < walk(off) and on.time_to_cover_all_s < off.time_to_cover_all_s
    assert on.expected_time_to_find_s <= 1.10 * off.expected_time_to_find_s
    # the baselines are the same E[T] either way
    assert on.baseline_nearest_s == pytest.approx(off.baseline_nearest_s)
    assert on.baseline_highest_prob_s == pytest.approx(off.baseline_highest_prob_s)

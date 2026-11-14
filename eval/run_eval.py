"""Evaluation harness (v0, synthetic callers on the synthetic building).

    uv run python -m eval.run_eval --n 200 --seed 0

For each simulated caller (true room drawn from the prior, noisy phone fix,
persona-filtered truthful clues) the posterior and search plan are computed
under three evidence conditions:

    prior only  ->  + phone  ->  + phone + clues

and scored on floor accuracy, top-1 / top-3, 90% credible-set coverage and size,
entropy, and time-to-find: the planner's own E[T] next to the nearest-first and
highest-probability-first baselines, plus the *realized* time until the true
room's search completes under each ordering.

Because callers are sampled from the model's own prior with the model's own
noise, this checks calibration and plumbing - it is not evidence about real calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional, Sequence

from eval import metrics
from eval.metrics import Trial

CONDITIONS = ("prior only", "+ phone", "+ phone + clues")


def _import_engine() -> dict[str, Any]:
    """Import the modules owned by other packages; fail with one clear message."""
    try:
        from core.fusion import posterior_probs
        from core.uncertainty import summarize
        from ingest.matterport import generate_synthetic_office, load_building
        from sim.sample_caller import sample_caller
    except ImportError as exc:  # pragma: no cover - only while the repo is half-built
        raise SystemExit(
            f"eval.run_eval: a required module is missing or incomplete ({exc}).\n"
            "Needs core.fusion.posterior_probs, core.uncertainty.summarize, "
            "ingest.matterport.generate_synthetic_office and sim.sample_caller.\n"
            "Run from the repo root with `uv run python -m eval.run_eval`."
        ) from exc
    return {
        "posterior_probs": posterior_probs,
        "summarize": summarize,
        "generate_synthetic_office": generate_synthetic_office,
        "load_building": load_building,
        "sample_caller": sample_caller,
    }


def _import_planner() -> Optional[dict[str, Any]]:
    """Planner is optional: without it the posterior metrics still print."""
    try:
        from core.building_graph import build_graph
        from planning.baselines import highest_prob_first_order, nearest_first_order
        from planning.search_order import (
            completion_times, pairwise_travel, plan_search, search_time_s, select_candidates,
        )
    except ImportError as exc:
        print(f"[eval] planning/ not available ({exc}); skipping time-to-find columns.", file=sys.stderr)
        return None
    return {
        "build_graph": build_graph,
        "nearest_first_order": nearest_first_order,
        "highest_prob_first_order": highest_prob_first_order,
        "completion_times": completion_times,
        "pairwise_travel": pairwise_travel,
        "plan_search": plan_search,
        "search_time_s": search_time_s,
        "select_candidates": select_candidates,
    }


def _plan_fields(building, graph, rooms, room_probs, p_nim, true_room_id, n_teams, P) -> dict[str, Any]:
    """Planner E[T] + baselines, and realized completion time of the true room."""
    plan = P["plan_search"](building, room_probs, p_nim, n_teams=n_teams)
    out: dict[str, Any] = {
        "expected_time_s": plan.expected_time_to_find_s,
        "baseline_nearest_s": plan.baseline_nearest_s,
        "baseline_highest_prob_s": plan.baseline_highest_prob_s,
    }
    for team in plan.teams:
        for stop in team:
            if stop.room_id == true_room_id:
                out["realized_time_s"] = stop.eta_s + P["search_time_s"](rooms[true_room_id])

    # Re-walk the two baselines over the planner's candidate set from the same start.
    cands = P["select_candidates"](building, room_probs)
    start = plan.entrance_id
    if cands and start in graph and true_room_id in cands:
        travel = P["pairwise_travel"](graph, [start, *cands])
        search_times = {rid: P["search_time_s"](rooms[rid]) for rid in cands}
        probs = {rid: room_probs[rid] for rid in cands}
        near = P["nearest_first_order"](cands, probs, travel, start)
        high = P["highest_prob_first_order"](cands, probs, travel, start)
        out["realized_nearest_s"] = P["completion_times"](near, travel, search_times, start)[true_room_id]
        out["realized_highest_prob_s"] = P["completion_times"](high, travel, search_times, start)[true_room_id]
    return out


def run(
    n: int = 200,
    seed: int = 0,
    hour: int = 14,
    persona: Optional[str] = None,
    n_teams: int = 1,
    building_id: Optional[str] = None,
    with_plan: bool = True,
) -> dict[str, dict[str, float]]:
    """Run the evaluation and return ``{condition: metrics dict}``."""
    import numpy as np

    E = _import_engine()
    P = _import_planner() if with_plan else None

    building = E["load_building"](building_id) if building_id else E["generate_synthetic_office"]()
    rooms = {r.id: r for r in building.rooms}
    graph = P["build_graph"](building) if P else None
    rng = np.random.default_rng(seed)

    trials: dict[str, list[Trial]] = {c: [] for c in CONDITIONS}
    for _ in range(n):
        caller = E["sample_caller"](building, rng, hour=hour, persona=persona)
        evidence = {
            "prior only": (None, ()),
            "+ phone": (caller["phone_fix"], ()),
            "+ phone + clues": (caller["phone_fix"], caller["clues"]),
        }
        for cond, (fix, clues) in evidence.items():
            room_probs, p_nim = E["posterior_probs"](building, hour=hour, fix=fix, clues=clues)
            posterior = E["summarize"](building, room_probs, p_nim)
            plan_kw: dict[str, Any] = {}
            if P:
                plan_kw = _plan_fields(
                    building, graph, rooms, room_probs, p_nim, caller["room_id"], n_teams, P
                )
            trials[cond].append(
                Trial.from_posterior(caller["room_id"], caller["floor_label"], posterior, **plan_kw)
            )
    return {cond: metrics.summarize_trials(ts) for cond, ts in trials.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_POSTERIOR_COLS = [
    ("floor acc", "floor_acc", "{:.1%}"),
    ("top-1", "top1", "{:.1%}"),
    ("top-3", "top3", "{:.1%}"),
    ("90% cov", "coverage_90", "{:.1%}"),
    ("|CS90|", "cs90_size", "{:.1f}"),
    ("H bits", "entropy_bits", "{:.2f}"),
    ("p(true)", "p_true", "{:.3f}"),
]
_TIME_COLS = [
    ("E[T] plan", "et_plan_s", "{:.0f}s"),
    ("E[T] nearest", "et_nearest_s", "{:.0f}s"),
    ("E[T] high-p", "et_highest_prob_s", "{:.0f}s"),
    ("T plan", "realized_plan_s", "{:.0f}s"),
    ("T nearest", "realized_nearest_s", "{:.0f}s"),
    ("T high-p", "realized_highest_prob_s", "{:.0f}s"),
    ("in plan", "found_rate", "{:.1%}"),
]


def _fmt(value: float, spec: str) -> str:
    return "-" if value != value else spec.format(value)  # NaN -> "-"


def format_table(results: dict[str, dict[str, float]], cols: Sequence[tuple[str, str, str]]) -> str:
    header = ["condition", *[c[0] for c in cols]]
    rows = [[cond, *[_fmt(m[key], spec) for _, key, spec in cols]] for cond, m in results.items()]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(  # noqa: E731
        cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(r, widths))
    )
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *[line(r) for r in rows]])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of simulated callers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hour", type=int, default=14, help="hour of day for the prior (0-23)")
    ap.add_argument("--persona", default=None, help="persona name from sim/personas.yaml (default: file default)")
    ap.add_argument("--teams", type=int, default=1, help="number of search teams")
    ap.add_argument("--building", default=None, help="building id (default: synthetic office)")
    ap.add_argument("--no-plan", action="store_true", help="skip the search planner (posterior metrics only)")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON instead of tables")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    results = run(
        n=args.n, seed=args.seed, hour=args.hour, persona=args.persona,
        n_teams=args.teams, building_id=args.building, with_plan=not args.no_plan,
    )
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\nSimulated callers: n={args.n} seed={args.seed} hour={args.hour} "
          f"persona={args.persona or 'default'} teams={args.teams}  ({elapsed:.1f}s)\n")
    print("Posterior quality")
    print(format_table(results, _POSTERIOR_COLS))
    has_plan = any(m["et_plan_s"] == m["et_plan_s"] for m in results.values())
    if has_plan:
        print("\nTime to find (seconds from the entrance)")
        print(format_table(results, _TIME_COLS))
        print(
            "\nE[T] = planner's expectation under its own posterior, vs nearest-first and "
            "highest-probability-first\n       over the same candidate rooms.  "
            "T = realized time until the TRUE room's search completes\n       "
            "(mean over trials where the true room is in the 99%-coverage plan; "
            "'in plan' is that fraction)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

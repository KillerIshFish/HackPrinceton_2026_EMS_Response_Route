"""Run the benchmark trials in parallel and write one CSV row per trial.

    uv run python -m eval.benchmark.simulate --main 400 --robust 150 --workers 20

Writes eval/results/benchmark_trials.csv. No network.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from eval.benchmark.common import BUILDINGS, CALL_TYPES, VARIANTS, engine, trial

OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark_trials.csv"


def _init() -> None:
    engine()


def _job(args):
    bid, ck, variant, idx = args
    rows = []
    for i in idx:
        try:
            rows.append(trial(bid, ck, variant, i))
        except Exception as exc:  # keep going; failures are counted and reported
            rows.append({"building": bid, "call_type": ck, "variant": variant, "trial": i, "error": repr(exc)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=int, default=400, help="trials per building x call type, main variant")
    ap.add_argument("--robust", type=int, default=150, help="trials per building x call type, each robustness variant")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args(argv)

    jobs = []
    for variant in VARIANTS:
        n = args.main if variant == "main" else args.robust
        for bid in BUILDINGS:
            for ct in CALL_TYPES:
                for s in range(0, n, args.chunk):
                    jobs.append((bid, ct.key, variant, list(range(s, min(n, s + args.chunk)))))
    total = sum(len(j[3]) for j in jobs)
    print(f"{total} trials in {len(jobs)} jobs on {args.workers} workers", flush=True)

    t0 = time.time()
    rows, errors, done = [], 0, 0
    with Pool(args.workers, initializer=_init) as pool:
        for out in pool.imap_unordered(_job, jobs):
            rows += out
            done += len(out)
            errors += sum(1 for r in out if "error" in r)
            if len(rows) % 2000 < len(out):
                print(f"  {done}/{total}  {time.time() - t0:.0f}s  errors={errors}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["variant"], r["building"], r["call_type"], r["trial"])))
    print(f"wrote {OUT} ({len(rows)} rows, {errors} errors) in {time.time() - t0:.0f}s")
    for r in [r for r in rows if "error" in r][:5]:
        print("  error:", r, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Statistics, charts and summary for the benchmark.

    uv run --with matplotlib --with pandas python -m eval.benchmark.analyze

(neither matplotlib nor pandas is a project dependency in pyproject.toml, hence the --with flags)

Reads eval/results/benchmark_trials.csv (and speech_results.json, freeform_parser.json when
present); writes eval/results/summary.json and eval/results/charts/*.png.

Statistics: every comparison is paired (same caller, same entrance, same call-to-curb time).
95% CIs are percentile bootstraps over trials (2,000 resamples).
Outcomes: in-building time is capped at 60 minutes for outcome calculations only (a real
incident would escalate to more crews long before that; the cap applies to every arm).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eval.benchmark.common import (
    BUILDINGS, CALL_TYPE_BY_KEY, CALL_TYPES, CURB_MEDIAN_S, CURB_SIGMA, ENTRY_S, REPORTED_VARIANTS, VARIANTS,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHARTS = RESULTS / "charts"
CAP_S = 3600.0  # 60 min cap on in-building time for outcomes and capped means (judgement; see module docstring)
RNG = np.random.default_rng(2026)  # fixed seed so the bootstrap CIs in summary.json are reproducible
N_BOOT = 2000  # bootstrap resamples per CI

ARMS = ["C", "B", "A"]
ARM_LABEL = {"C": "App", "B": "Informed dispatcher", "A": "Address only"}
ARM_COLOR = {"C": "#2a78d6", "B": "#eb6834", "A": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Outcome models
# ---------------------------------------------------------------------------


def survival_fn(ct_key: str):
    from core.schema import Situation
    from situation.survival import survival_for

    ct = CALL_TYPE_BY_KEY[ct_key]
    f = survival_for(Situation(**ct.situation))
    return np.vectorize(f)


def outcome(df: pd.DataFrame, ct_key: str, arm: str, rate_idx: int = 1) -> np.ndarray:
    """Per-trial outcome in percentage points (higher is better) for one arm.

    curve           survival % at contact time on the call clock
    linear_arrival  fire: 100 - rate * minutes after arrival at the entrance (floored at 0)
    linear_call     stroke / bleeding: - rate * minutes from the call (points lost)

    ``rate_idx`` picks the low / mid / high rate from ``CallType.rate_pp_min``. The fire baseline
    of 100 is not neutral: with the floor at 0, any search longer than 100 / rate minutes
    (about 27 min at the mid rate) scores 0 in every arm, so arm differences on long searches vanish.
    The shared call-to-curb and entry times cancel in linear_call differences, not in curves.
    """
    ct = CALL_TYPE_BY_KEY[ct_key]
    t_find = np.minimum(df[f"t_{arm}"].to_numpy(float), CAP_S)
    t_call = df["t_curb"].to_numpy(float) + ENTRY_S + t_find
    if ct.outcome == "curve":
        return 100.0 * survival_fn(ct_key)(t_call)
    if ct.outcome == "linear_arrival":
        r = ct.rate_pp_min[rate_idx]
        return np.maximum(0.0, 100.0 - r * t_find / 60.0)
    if ct.outcome == "linear_call":
        r = ct.rate_pp_min[rate_idx]
        return -r * t_call / 60.0
    return np.full(len(df), np.nan)


def boot_ci(x: np.ndarray, stat=np.mean) -> tuple[float, float]:
    """95% percentile-bootstrap CI of ``stat`` over trials (N_BOOT resamples with replacement)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    s = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def time_block(d: pd.DataFrame) -> dict:
    out = {"n": int(len(d))}
    for a in ARMS:
        t = d[f"t_{a}"].to_numpy(float)
        out[f"median_{a}"] = float(np.median(t))
        out[f"mean_{a}"] = float(np.mean(np.minimum(t, CAP_S)))
        out[f"p90_{a}"] = float(np.percentile(t, 90))
    saved = (d["t_B"] - d["t_C"]).to_numpy(float)
    saved_cap = np.minimum(d["t_B"], CAP_S).to_numpy(float) - np.minimum(d["t_C"], CAP_S).to_numpy(float)
    out["saved_vs_B_median"] = float(np.median(saved))
    out["saved_vs_B_median_ci"] = boot_ci(saved, np.median)
    out["saved_vs_B_mean_capped"] = float(saved_cap.mean())
    out["saved_vs_B_mean_capped_ci"] = boot_ci(saved_cap)
    out["saved_vs_A_median"] = float(np.median(d["t_A"] - d["t_C"]))
    out["pct_reduction_vs_B_median_time"] = float(1 - np.median(d["t_C"]) / np.median(d["t_B"]))
    # 1e-6 s tolerance: arms that search the same rooms in the same order count as a tie, not a float win.
    # 300 s = the "reached within 5 min of entering" share in REPORT.md.
    out["frac_faster_than_B"] = float((d["t_C"] < d["t_B"] - 1e-6).mean())
    out["frac_tie_B"] = float((abs(d["t_C"] - d["t_B"]) <= 1e-6).mean())
    out["frac_slower_than_B"] = float((d["t_C"] > d["t_B"] + 1e-6).mean())
    out["frac_C_within_5min"] = float((d["t_C"] <= 300).mean())
    out["frac_B_within_5min"] = float((d["t_B"] <= 300).mean())
    out["frac_A_within_5min"] = float((d["t_A"] <= 300).mean())
    out["top1"] = float((d["rank_true"] == 1).mean())
    out["top3"] = float((d["rank_true"] <= 3).mean())
    out["in_plan"] = float(d["in_plan"].astype(str).eq("True").mean())
    return out


def outcome_block(d: pd.DataFrame, ct_key: str) -> dict:
    ct = CALL_TYPE_BY_KEY[ct_key]
    if ct.outcome == "time_only":
        return {"kind": "time_only"}
    res = {"kind": ct.outcome, "label": ct.outcome_label}
    for a in ARMS:
        o = outcome(d, ct_key, a)
        res[f"mean_{a}"] = float(o.mean())
        res[f"mean_{a}_ci"] = boot_ci(o)
    diff = outcome(d, ct_key, "C") - outcome(d, ct_key, "B")
    res["gain_vs_B"] = float(diff.mean())
    res["gain_vs_B_ci"] = boot_ci(diff)
    res["gain_vs_A"] = float((outcome(d, ct_key, "C") - outcome(d, ct_key, "A")).mean())
    if ct.rate_pp_min:
        res["gain_vs_B_rate_low"] = float((outcome(d, ct_key, "C", 0) - outcome(d, ct_key, "B", 0)).mean())
        res["gain_vs_B_rate_high"] = float((outcome(d, ct_key, "C", 2) - outcome(d, ct_key, "B", 2)).mean())
    return res


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _legend(ax, arms=ARMS, **kw):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="", color=ARM_COLOR[a], markersize=8, label=ARM_LABEL[a]) for a in arms]
    ax.legend(handles=h, loc=kw.pop("loc", "lower right"), **kw)


def fmt_min(s: float) -> str:
    m = s / 60.0
    if m < 1:
        return f"{s:.0f} s"
    if m < 120:
        return f"{m:.1f} min"
    return f"{m / 60:.1f} h"


def chart_time_by_type(S: dict):
    import matplotlib.pyplot as plt

    keys = [c.key for c in CALL_TYPES]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for yi, k in enumerate(keys):
        blk = S["main"]["by_type"][k]["time"]
        xs = [blk[f"median_{a}"] / 60 for a in ARMS]
        ax.plot([min(xs), max(xs)], [yi, yi], color=GRID, lw=2, zorder=1)
        for a in ARMS:
            ax.scatter(blk[f"median_{a}"] / 60, yi, s=70, color=ARM_COLOR[a], zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(blk["median_C"] / 60, yi + 0.28, fmt_min(blk["median_C"]), ha="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(keys)), [CALL_TYPE_BY_KEY[k].label for k in keys])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200], ["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Median minutes from building entrance to patient (log scale)")
    ax.set_title("Time to find the patient inside the building, by call type")
    _legend(ax, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "01_time_by_call_type.png")
    plt.close(fig)


def chart_time_saved(S: dict):
    import matplotlib.pyplot as plt

    keys = [c.key for c in CALL_TYPES if c.key != "fire_smoke"]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for yi, k in enumerate(keys):
        blk = S["main"]["by_type"][k]["time"]
        m = blk["saved_vs_B_median"] / 60
        lo, hi = (v / 60 for v in blk["saved_vs_B_median_ci"])
        ax.barh(yi, m, height=0.5, color=ARM_COLOR["C"], zorder=2)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=3, zorder=3)
        ax.text(max(hi, m) + 0.12, yi, f"{m:.1f} min   (faster in {blk['frac_faster_than_B']:.0%} of calls, slower in "
                f"{blk['frac_slower_than_B']:.0%})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(keys)), [CALL_TYPE_BY_KEY[k].label for k in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    f = S["main"]["by_type"]["fire_smoke"]["time"]
    ax.set_xlabel("Median minutes saved per call vs. informed dispatcher (paired, 95% CI)\n"
                  f"Fire / smoke (not shown, slow smoke search rates): median {f['saved_vs_B_median'] / 60:.0f} min saved, "
                  f"{f['median_B'] / 60:.0f} -> {f['median_C'] / 60:.0f} min")
    ax.set_title("Medical calls: time saved by the app against an informed dispatcher")
    ax.set_xlim(0, ax.get_xlim()[1] * 2.3)
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "02_time_saved_vs_dispatcher.png")
    plt.close(fig)


def chart_survival(S: dict):
    import matplotlib.pyplot as plt

    keys = ["cardiac_cpr", "cardiac_nocpr", "cardiac_unwitnessed"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.26
    for xi, k in enumerate(keys):
        o = S["main"]["by_type"][k]["outcome"]
        for j, a in enumerate(ARMS):
            m = o[f"mean_{a}"]
            lo, hi = o[f"mean_{a}_ci"]
            x = xi + (j - 1) * w
            ax.bar(x, m, width=w - 0.03, color=ARM_COLOR[a], zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=2, zorder=3)
            ax.text(x, hi + 0.4, f"{m:.1f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(keys)), [CALL_TYPE_BY_KEY[k].label.replace("Cardiac arrest, ", "Cardiac arrest\n") for k in keys])
    ax.set_ylabel("Predicted survival to discharge (%)")
    ax.set_title("Cardiac arrest: predicted survival by search method")
    _legend(ax, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(CHARTS / "03_cardiac_survival.png")
    plt.close(fig)


def chart_linear(S: dict):
    import matplotlib.pyplot as plt

    keys = ["cardiac_cpr", "cardiac_nocpr", "cardiac_unwitnessed", "fire_smoke", "stroke", "internal_bleeding"]
    units = {"curve": "survival points", "linear_arrival": "survival points", "linear_call": "outcome points"}
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for yi, k in enumerate(keys):
        o = S["main"]["by_type"][k]["outcome"]
        g = o["gain_vs_B"]
        lo, hi = o["gain_vs_B_ci"]
        ax.barh(yi, g, height=0.5, color=ARM_COLOR["C"], zorder=2)
        ax.errorbar(g, yi, xerr=[[g - lo], [hi - g]], color=INK, lw=1.2, capsize=3, zorder=3)
        extra = ""
        if "gain_vs_B_rate_low" in o:
            lo_r, hi_r = sorted((o['gain_vs_B_rate_low'], o['gain_vs_B_rate_high']))
            extra = f"  (range {lo_r:.2f} to {hi_r:.2f})"
        ax.text(max(hi, g) + 0.1, yi, f"+{g:.2f} {units[o['kind']]}{extra}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(keys)), [CALL_TYPE_BY_KEY[k].label for k in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]), right=ax.get_xlim()[1] * 1.9)
    ax.set_xlabel("Percentage points gained per call vs. informed dispatcher (95% CI)")
    ax.set_title("Outcome gained per call, by call type")
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "04_outcome_gain.png")
    plt.close(fig)


def chart_cdf(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = df[(df.variant == "main") & df.call_type.str.startswith("cardiac")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for a in ARMS:
        t = np.sort((d["t_curb"] + ENTRY_S + d[f"t_{a}"]).to_numpy(float) / 60)
        ax.plot(t, np.arange(1, len(t) + 1) / len(t), color=ARM_COLOR[a], lw=2, label=ARM_LABEL[a])
    ax.axvline(15, color=AXIS, lw=1, ls="--", zorder=0)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Minutes from the 911 call to the crew reaching the patient")
    ax.set_ylabel("Share of calls reached")
    ax.set_title("Cardiac arrest calls: how fast patients are reached (from the 911 call)")
    for a in ARMS:
        t = (d["t_curb"] + ENTRY_S + d[f"t_{a}"]).to_numpy(float) / 60
        ax.annotate(f"{(t <= 15).mean():.0%} by 15 min", xy=(15, (t <= 15).mean()), xytext=(14.6, (t <= 15).mean()),
                    fontsize=8, color=INK2, va="center", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    _legend(ax, loc="lower right")
    fig.savefig(CHARTS / "05_cardiac_time_to_patient_cdf.png")
    plt.close(fig)


def chart_robustness(S: dict):
    import matplotlib.pyplot as plt

    vs = list(REPORTED_VARIANTS)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for yi, v in enumerate(vs):
        blk = S[v]["pooled"]["time"]
        m = blk["saved_vs_B_mean_capped"] / 60
        lo, hi = (x / 60 for x in blk["saved_vs_B_mean_capped_ci"])
        ax.barh(yi, m, height=0.5, color=ARM_COLOR["C"] if v == "main" else "#86b6ef", zorder=2)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=3, zorder=3)
        ax.text(max(hi, m) + 0.2, yi, f"{m:.1f} min  (faster in {blk['frac_faster_than_B']:.0%}, "
                f"slower in {blk['frac_slower_than_B']:.0%})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(vs)), [REPORTED_VARIANTS[v] for v in vs])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]), right=ax.get_xlim()[1] * 1.6)
    ax.set_xlabel("Mean minutes saved per call vs. informed dispatcher (all call types, 95% CI)")
    ax.set_title("Stress tests: does the advantage hold when assumptions break?")
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "06_robustness.png")
    plt.close(fig)


def chart_sensitivity(S: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = {"cardiac_cpr": "#2a78d6", "cardiac_nocpr": "#eb6834", "cardiac_unwitnessed": "#1baf7a"}
    for k, rows in S["sensitivity"].items():
        x = [r["curb_min"] for r in rows]
        ax.plot(x, [r["gain_vs_B"] for r in rows], color=colors[k], lw=2, label=CALL_TYPE_BY_KEY[k].label)
        ax.fill_between(x, [r["ci"][0] for r in rows], [r["ci"][1] for r in rows], color=colors[k], alpha=0.12, lw=0)
    ax.axvline(CURB_MEDIAN_S / 60, color=AXIS, ls="--", lw=1)
    ax.text(CURB_MEDIAN_S / 60 + 0.15, ax.get_ylim()[1] * 0.95, "US urban median (7 min)", fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("Minutes from the 911 call to the ambulance at the curb")
    ax.set_ylabel("Survival points gained vs. dispatcher")
    ax.set_title("Where the app matters most: gain vs. how fast the ambulance arrives")
    ax.legend(loc="upper right")
    fig.savefig(CHARTS / "07_sensitivity_curb_time.png")
    plt.close(fig)


def chart_buildings(S: dict):
    import matplotlib.pyplot as plt

    bids = list(BUILDINGS)
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for yi, b in enumerate(bids):
        blk = S["main"]["by_building"][b]["time"]
        for a in ARMS:
            ax.scatter(blk[f"median_{a}"] / 60, yi, s=70, color=ARM_COLOR[a], zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(blk["median_C"] / 60, yi + 0.3, fmt_min(blk["median_C"]), ha="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(bids)), [BUILDINGS[b] for b in bids])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Median minutes from entrance to patient (all call types, log scale)")
    ax.set_title("By building")
    _legend(ax, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "08_by_building.png")
    plt.close(fig)


def chart_speech(S: dict):
    import matplotlib.pyplot as plt

    sp = S.get("speech")
    if not sp:
        return
    rows = [("Informed dispatcher (hears same call)", "B", sp["median_B"], None),
            ("App, structured clues", "C", sp["median_structured"], sp["top3_structured"]),
            ("App, rules parser on speech", "C", sp["median_rules"], sp["top3_rules"]),
            ("App, Claude Haiku on speech", "C", sp["median_llm"], sp["top3_llm"])]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    for yi, (lab, arm, med, top3) in enumerate(rows):
        ax.barh(yi, med / 60, height=0.5, color=ARM_COLOR[arm] if yi != 2 else "#86b6ef", zorder=2)
        txt = fmt_min(med) + (f"   true room in top 3: {top3:.0%}" if top3 is not None else "")
        ax.text(med / 60 + 0.1, yi, txt, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[2] for r in rows) / 60 * 1.9)
    ax.set_xlabel(f"Median minutes from entrance to patient (n = {sp['n']} generated calls)")
    ax.set_title("Held-out speech test: Claude Haiku writes the caller, the app reads it")
    ax.grid(axis="y", visible=False)
    fig.savefig(CHARTS / "09_speech_test.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    df = pd.read_csv(RESULTS / "benchmark_trials.csv")
    if "error" in df.columns:
        n_err = int(df["error"].notna().sum())
        df = df[df["error"].isna()]
    else:
        n_err = 0
    S: dict = {"n_trials": int(len(df)), "n_errors": n_err,
               "assumptions": {"curb_median_s": CURB_MEDIAN_S, "curb_sigma": CURB_SIGMA, "entry_s": ENTRY_S, "cap_s": CAP_S}}

    for v in VARIANTS:
        dv = df[df.variant == v]
        blk = {"pooled": {"time": time_block(dv)}, "by_type": {}, "by_building": {}}
        for ct in CALL_TYPES:
            d = dv[dv.call_type == ct.key]
            blk["by_type"][ct.key] = {"time": time_block(d), "outcome": outcome_block(d, ct.key)}
            if v == "main":
                blk["by_type"][ct.key]["by_building"] = {
                    b: {"time": time_block(d[d.building == b]), "outcome": outcome_block(d[d.building == b], ct.key)}
                    for b in BUILDINGS
                }
        for b in BUILDINGS:
            blk["by_building"][b] = {"time": time_block(dv[dv.building == b])}
        # floor accuracy of what callers said (main only, informative)
        sf = dv["stated_floor_correct"].dropna().astype(str)
        blk["stated_floor_rate"] = float(dv["stated_floor_correct"].notna().mean())
        blk["stated_floor_correct_rate"] = float((sf == "True").mean()) if len(sf) else math.nan
        S[v] = blk

    # Lives per 1,000 calls for cardiac (pooled over the three cardiac types, equal weight)
    card = df[(df.variant == "main") & df.call_type.str.startswith("cardiac")]
    gains = np.concatenate([outcome(card[card.call_type == k], k, "C") - outcome(card[card.call_type == k], k, "B")
                            for k in ("cardiac_cpr", "cardiac_nocpr", "cardiac_unwitnessed")])
    gains_a = np.concatenate([outcome(card[card.call_type == k], k, "C") - outcome(card[card.call_type == k], k, "A")
                              for k in ("cardiac_cpr", "cardiac_nocpr", "cardiac_unwitnessed")])
    # 1 percentage point of survival per call = 10 survivors per 1,000 calls, hence the factor 10.
    S["cardiac_pooled"] = {"gain_vs_B_pp": float(gains.mean()), "gain_vs_B_ci": boot_ci(gains),
                           "lives_per_1000_vs_B": float(10 * gains.mean()),
                           "lives_per_1000_vs_B_ci": [10 * x for x in boot_ci(gains)],
                           "gain_vs_A_pp": float(gains_a.mean()), "lives_per_1000_vs_A": float(10 * gains_a.mean())}

    # Distribution of minutes from the call to the patient (cardiac, main), for the report page.
    S["cdf_cardiac"] = {
        a: [float(x) for x in np.percentile((card["t_curb"] + ENTRY_S + card[f"t_{a}"]).to_numpy(float) / 60, np.arange(0, 101))]
        for a in ARMS
    }

    # Sensitivity: replace each trial's call-to-curb time with a fixed value (search times kept).
    # Approximation: the app's cardiac plan was optimised for the trial's own curb time, not the swept one.
    S["sensitivity"] = {}
    main = df[df.variant == "main"]
    for k in ("cardiac_cpr", "cardiac_nocpr", "cardiac_unwitnessed"):
        d = main[main.call_type == k].copy()
        rows = []
        for m in np.arange(3, 15.5, 0.5):  # 3 to 15 min in 30 s steps, around the 7 min urban median
            d["t_curb"] = m * 60
            g = outcome(d, k, "C") - outcome(d, k, "B")
            rows.append({"curb_min": float(m), "gain_vs_B": float(g.mean()), "ci": boot_ci(g)})
        S["sensitivity"][k] = rows

    # Speech test
    sp_path = RESULTS / "speech_results.json"
    if sp_path.exists():
        sp = pd.DataFrame(json.loads(sp_path.read_text(encoding="utf-8")))
        S["speech"] = {
            "n": int(len(sp)),
            "median_B": float(sp.structured_t_B.median()), "median_A": float(sp.structured_t_A.median()),
            "median_structured": float(sp.structured_t_C.median()),
            "median_rules": float(sp.rules_t_C.median()), "median_llm": float(sp.llm_t_C.median()),
            "top1_structured": float((sp.structured_rank_true == 1).mean()),
            "top1_rules": float((sp.rules_rank_true == 1).mean()), "top1_llm": float((sp.llm_rank_true == 1).mean()),
            "top3_structured": float((sp.structured_rank_true <= 3).mean()),
            "top3_rules": float((sp.rules_rank_true <= 3).mean()), "top3_llm": float((sp.llm_rank_true <= 3).mean()),
            "saved_llm_vs_B_median": float((sp.structured_t_B - sp.llm_t_C).median()),
            "saved_llm_vs_B_median_ci": boot_ci((sp.structured_t_B - sp.llm_t_C).to_numpy(float), np.median),
            "saved_structured_vs_B_median": float((sp.structured_t_B - sp.structured_t_C).median()),
            "saved_rules_vs_B_median": float((sp.structured_t_B - sp.rules_t_C).median()),
            "llm_faster_than_B": float((sp.llm_t_C < sp.structured_t_B - 1e-6).mean()),
            "llm_slower_than_B": float((sp.llm_t_C > sp.structured_t_B + 1e-6).mean()),
            "llm_fallback_rate": float(sp.llm_fallback.mean()),
        }

    ff = RESULTS / "freeform_parser.txt"
    if ff.exists():
        S["freeform_note"] = "see freeform_parser.txt"

    CHARTS.mkdir(parents=True, exist_ok=True)
    _style()
    chart_time_by_type(S)
    chart_time_saved(S)
    chart_survival(S)
    chart_linear(S)
    chart_cdf(df)
    chart_robustness(S)
    chart_sensitivity(S)
    chart_buildings(S)
    chart_speech(S)
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(f"trials {S['n_trials']} (errors {n_err}); wrote summary.json and charts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

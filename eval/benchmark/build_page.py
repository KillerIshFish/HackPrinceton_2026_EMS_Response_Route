"""Fill report_template.html with summary.json and write eval/results/report.html.

    uv run python -m eval.benchmark.build_page
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from eval.benchmark.common import BUILDINGS, CALL_TYPES, REPORTED_VARIANTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def freeform() -> dict:
    """Top-1 / top-3 of the hand-written free-form calls, from eval.run_agent_eval's table."""
    text = (RESULTS / "freeform_parser.txt").read_text(encoding="utf-8", errors="replace")
    block = text.split("All free-form calls", 1)[1]
    out = {}
    for mode in ("rules", "llm"):
        m = re.search(rf"^{mode}\s+\d+ \(\d+\+\d+\)\s+(\d+%)\s+(\d+%)", block, re.M)
        out[f"{mode}_top1"], out[f"{mode}_top3"] = m.group(1), m.group(2)
    return out


def main() -> int:
    S = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    S["call_types"] = [{"key": c.key, "label": c.label, "outcome_label": c.outcome_label} for c in CALL_TYPES]
    S["variants"] = REPORTED_VARIANTS
    S["buildings"] = {
        "ptc-100": "Synthetic office, 100 rooms",
        "washington-5285": "5285 W Washington, 38 rooms",
        "bundy-olympic": "Bundy Olympic, 148 rooms",
        "jrr-princeton": "JRR Princeton, 208 rooms",
    }
    assert set(S["buildings"]) == set(BUILDINGS)
    S["freeform"] = freeform()
    S["generated"] = dt.date.today().isoformat()
    tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*DATA*/null", json.dumps(S, separators=(",", ":")))
    out = RESULTS / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

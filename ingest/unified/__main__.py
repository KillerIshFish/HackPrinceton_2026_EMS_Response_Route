"""One command for the whole pipeline.

    uv run python -m ingest.unified "11900 W Olympic Blvd"      # address or building name
    uv run python -m ingest.unified 246948                       # property id
    uv run python -m ingest.unified mp:1FTnLNf1sBz               # one whole-building Matterport scan
    uv run python -m ingest.unified --lookup "olympic"           # list matching properties only
    uv run python -m ingest.unified --index                      # (re)build the address index

Options: ``--offline`` (no Matterport or OSM calls; cached scans are still used),
``--no-stamp`` (skip unit-type scans), ``--no-context`` (skip the street-bearing sidecar),
``--refresh`` (look the street up again instead of reusing the sidecar), ``--json`` (print the
run report as JSON).

The building then appears in the web app's building menu, and at
``http://localhost:5173/?building=<id>`` when the two dev servers are running.
"""

from __future__ import annotations

import json
import sys

from ingest.unified import index
from ingest.unified.pipeline import generate


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in args if a.startswith("--")}
    words = [a for a in args if not a.startswith("--")]
    if "--index" in flags:
        print(json.dumps(index.build_index()))
        return 0
    if not words:
        print(__doc__)
        return 2
    query = " ".join(words)
    if "--lookup" in flags:
        for hit in index.lookup(query, limit=12):
            print(f"{hit['property_id']:>10}  {hit['address']}, {hit['city']}  "
                  f"({hit['stories'] or '?'} storeys{', ' + str(hit['units']) + ' units' if hit['units'] else ''})  {hit['name']}")
        return 0
    try:
        report = generate(query, stamp="--no-stamp" not in flags, context="--no-context" not in flags,
                          offline="--offline" in flags, refresh_context="--refresh" in flags)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if "--json" in flags:
        print(json.dumps(report, indent=1))
        return 0
    print(f"{report['name']} - {report['address'] or 'no address'}")
    print(f"  model {report['building_id']}: tier {report['tier']}, archetype {report['archetype']}, "
          f"{report['levels']} levels, {report['spaces']} spaces, {report['walls']} walls, {report['links']} links")
    for s in report["stages"]:
        print(f"  {s['ms']:>8.1f} ms  {s['stage']:<18} {s['detail']}")
    print(f"  {report['total_ms']:>8.1f} ms  total")
    for note in report["notes"]:
        print(f"  note: {note}")
    print(f"  open: http://localhost:5173/?building={report['building_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

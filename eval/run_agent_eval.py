"""Interpreter evaluation: the rules against the grounded Claude Haiku reader, on free-form calls.

    uv run python -m eval.run_agent_eval --modes rules                 # all calls, no network
    uv run python -m eval.run_agent_eval --modes rules,llm,agent -v    # live; replies are cached on disk

Calls come from ``sim/freeform_calls.yaml``: a building, a true room and a few dispatcher /
caller turns that talk the way people do (room and unit numbers, tenant names, signs,
relative position, corrections, yes / no answers, speech-recognition noise) plus calls with
no usable location, where the right output is no location evidence. The scripted console
scenarios (``sim.generate_calls.scenario_definitions``) are run as well, phone fix included,
and reported in their own table: rank and probability of the true room at the end of the call.

Modes

* ``rules``  deterministic rules only (``parser.interpret`` mode "rules").
* ``llm``    the grounded Claude Haiku interpreter (``parser.agent``), the console's "Claude Haiku".
* ``agent``  the same interpreter under its older name; its results come from the same disk cache.
* ``vocab``  the older single-utterance vocabulary extractor (``parser.extract.extract_clues_llm``),
             kept only for comparison; it is no longer a console mode.

Each call is run turn by turn through ``parser.interpret.interpret_turn``, carrying the
earlier evidence and facts forward the way ``api.session.CallSession`` does (a correction
marks what it replaces superseded), and the active evidence is fused with
``core.fusion.posterior_probs``. One table per call set:

* top-1 / top-3   share of location calls whose true room ranks 1st / in the top 3 (ties count
                  against the true room)
* p(true)         mean posterior on the true room (location calls)
* no-info empty   share of no-information calls that produced no active location evidence
                  (facts such as "he's not breathing" are allowed)
* req/call        Anthropic requests it took to produce the results (and how many of them
                  this run sent; the rest came from the on-disk caches)
* tok/call        input (cached and uncached) and output tokens of the requests this run sent
* s/turn          wall-clock seconds per caller turn spent in model requests: mean, median and
                  90th percentile over the turns that needed the model

Spend. Replies are cached on disk (``data/processed/llm_agent_cache.json``,
``llm_clue_cache.json``), so a second run with the same prompt and settings sends nothing.
There is no request cap unless ``LLM_MAX_CALLS`` is set in the environment (ground rule 7:
the ten-request cap was lifted on 2026-11-14). ``--project-only`` prints how many caller
turns are not cached yet and exits. The request and token totals of the run are printed at
the end, with ``data/processed/llm_usage.json`` before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CALLS_PATH = REPO_ROOT / "sim" / "freeform_calls.yaml"
MODES = ("rules", "llm", "agent", "vocab")
MODEL_MODES = ("llm", "agent")
_USAGE_KEYS = ("calls", "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def load_calls(path: Path = CALLS_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    calls = list(data.get("calls") or [])
    ids = [c["id"] for c in calls]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate call ids in {path}: {dupes}")
    return calls


def scenario_calls() -> list[dict]:
    """The scripted console scenarios as eval calls (system turns and phone fixes kept)."""
    from sim.generate_calls import scenario_definitions

    out = []
    for d in scenario_definitions():
        turns = [{"speaker": t["speaker"], "text": t["text"], "phone_fix": t.get("phone_fix")} for t in d["script"]]
        out.append({"id": f"scenario:{d['id']}", "building": d["building_id"], "true_room": d["true_room_id"],
                    "tags": ["scenario", d["incident_type"]], "scenario": True, "turns": turns})
    return out


_BUILDINGS: dict[str, Any] = {}


def get_building(building_id: str):
    if building_id not in _BUILDINGS:
        from ingest.matterport import generate_synthetic_office, load_building

        _BUILDINGS[building_id] = generate_synthetic_office() if building_id == "ptc-100" else load_building(building_id)
    return _BUILDINGS[building_id]


def check_calls(calls: Sequence[dict]) -> None:
    """Every call names a known building and a room that exists in it, and has a caller turn."""
    problems = []
    for c in calls:
        try:
            b = get_building(c["building"])
        except KeyError:
            problems.append(f"{c['id']}: unknown building {c['building']}")
            continue
        if c["true_room"] not in {r.id for r in b.rooms}:
            problems.append(f"{c['id']}: room {c['true_room']} is not in {b.id}")
        if not any(t["speaker"] == "caller" for t in c["turns"]):
            problems.append(f"{c['id']}: no caller turn")
    if problems:
        raise SystemExit("bad calls:\n  " + "\n  ".join(problems))


def make_transcript(call: dict) -> list:
    from core.schema import Question, TranscriptTurn

    turns = []
    for i, t in enumerate(call["turns"]):
        q = t.get("question")
        question = None
        if q:
            question = Question(id=f"{q['clue_type']}:{q['value']}", question=t["text"], kind="yes_no",
                                clue_type=q["clue_type"], value=str(q["value"]))
        turns.append(TranscriptTurn(turn=i, title=f"{t['speaker']} {i}", speaker=t["speaker"], text=t["text"],
                                    question=question, phone_fix=t.get("phone_fix")))
    return turns


def caller_indices(transcript: Sequence) -> list[int]:
    return [i for i, t in enumerate(transcript) if t.speaker == "caller"]


# ---------------------------------------------------------------------------
# Running one call
# ---------------------------------------------------------------------------


def _usage() -> dict:
    import parser.extract as ex

    u = ex.llm_usage()
    return {k: int(u.get(k, 0) or 0) for k in _USAGE_KEYS}


def _carry(prior: list, res: Any) -> list:
    sup = set(res.supersedes)
    kept = [e.model_copy(update={"status": "superseded"}) if e.id in sup and e.status == "active" else e for e in prior]
    return kept + list(res.evidence)


@dataclass
class CallResult:
    call_id: str
    mode: str
    no_info: bool
    scenario: bool = False
    status: str = "ok"  # ok | fallback
    rank: Optional[int] = None
    p_true: Optional[float] = None
    n_rooms: int = 0
    active_location: int = 0
    facts: int = 0
    suggested_type: Optional[str] = None
    requests_sent: int = 0  # requests this run sent for the call
    requests_to_produce: int = 0  # requests it took (this run or the run that filled the cache)
    turn_secs: list = field(default_factory=list)  # seconds in model requests per caller turn that needed the model
    turn_requests: list = field(default_factory=list)  # requests per caller turn that needed the model
    tokens: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # short descriptions of the final active evidence
    trajectory: list = field(default_factory=list)  # (turn index, rank, p) after each caller turn
    note: str = ""


def _describe(e: Any, building: Any) -> str:
    if e.kind == "vocab" and e.clue is not None:
        what = f"{e.clue.type}={e.clue.value}" + ("" if e.clue.polarity else "(no)")
    else:
        from core.evidence import selector_hits

        sel = e.selector.model_dump(exclude_none=True) if e.selector is not None else {}
        hits = selector_hits(building, e.selector) if e.selector is not None else []
        names = {r.id: r.name for r in building.rooms}
        where = ", ".join(names[h] for h in hits[:3]) + (f" +{len(hits) - 3}" if len(hits) > 3 else "") if hits else "UNMATCHED"
        what = ("NOT " if e.negate else "") + ",".join(f"{k}={v}" for k, v in sel.items()) + f" -> {where}"
    return f"{what}@{e.confidence:.2f}[{e.source}]"


def _rank(probs: dict, room_id: str) -> int:
    p = probs.get(room_id, 0.0)
    above = sum(1 for rid, q in probs.items() if rid != room_id and q >= p - 1e-12)
    return above + 1


def _interpret(building, transcript, i, mode, prior, facts, incident_type, vocab):
    """One caller turn in ``mode``: (Interpretation, requests to produce, seconds in the model)."""
    from parser.interpret import _rules_path, interpret_turn

    if mode == "vocab":
        import parser.extract as ex

        before = _usage()["calls"]
        started = time.perf_counter()
        out = _rules_path(building, transcript, i, use_llm=True, prior_evidence=prior, vocab=vocab)
        seconds = time.perf_counter() - started
        sent = _usage()["calls"] - before
        produced = 1 if out.source == "llm" and (transcript[i].text or "").strip() else 0
        return out, produced, (seconds if sent else None)
    if mode in MODEL_MODES:
        import parser.agent as ag

        before = _usage()["calls"]
        ag.LAST_RUN.clear()
        out = interpret_turn(building, transcript, i, mode, prior_evidence=prior, vocab=vocab,
                             prior_facts=facts, incident_type=incident_type)
        sent = _usage()["calls"] - before
        if sent:
            reqs = ag.LAST_RUN.get("requests") or []
            return out, len(reqs), sum(r.get("seconds", 0.0) for r in reqs)
        entry = _agent_cache_entry(building, transcript, i, prior, facts, incident_type, vocab)
        if entry is not None:
            return out, int(entry.get("requests", 0)), float(entry.get("seconds", 0.0)) or None
        return out, 0, None
    return interpret_turn(building, transcript, i, mode, prior_evidence=prior, vocab=vocab), 0, None


def _agent_cache_entry(building, transcript, index, prior, facts, incident_type, vocab) -> Optional[dict]:
    """The disk-cache entry ``parser.agent.interpret`` would read for this turn, if there is one."""
    import parser.agent as ag
    import parser.extract as ex
    from parser.building_tools import BuildingIndex
    from parser.dialog import read_dialog

    idx = BuildingIndex(building, None, vocab=vocab)
    reading = read_dialog(building, transcript, index, vocab, meta=idx.meta_rooms)
    system = ag.build_system_prompt(idx, vocab)
    user = ag.build_user_message(building, transcript, index, prior, reading,
                                 prior_facts=facts, incident_type=incident_type)
    return ex._read_json(ag.AGENT_CACHE_PATH, {}).get(ag._cache_key(system, user))


def run_call(call: dict, mode: str, vocab: dict) -> CallResult:
    from core.evidence import evidence_match
    from core.fusion import posterior_probs
    from parser.rule_facts import read_facts

    building = get_building(call["building"])
    transcript = make_transcript(call)
    res = CallResult(call_id=call["id"], mode=mode, no_info=bool(call.get("no_info")),
                     scenario=bool(call.get("scenario")))
    usage0 = _usage()
    prior: list = []
    facts: list = []
    incident_type: Optional[str] = None
    fix = None

    def posterior(upto: int):
        active = [e for e in prior if e.status == "active"]
        return posterior_probs(building, fix=fix, clues=active, vocab=vocab)[0], active

    for i, turn in enumerate(transcript):
        if turn.phone_fix is not None:
            fix = turn.phone_fix
        if turn.speaker != "caller":
            continue
        tr = transcript[: i + 1]
        out, produced, seconds = _interpret(building, tr, i, mode, prior, facts, incident_type, vocab)
        res.sources.append(out.source)
        res.requests_to_produce += produced
        if produced:
            res.turn_requests.append(produced)
            if seconds is not None:
                res.turn_secs.append(round(seconds, 3))
        model_read = mode in MODEL_MODES and out.source == mode
        if mode != "rules" and not (model_read or (mode == "vocab" and out.source == "llm")):
            res.status = "fallback"
            res.note = out.source
        turn_facts, suggested = list(out.facts), out.suggested_type
        if not model_read:  # as in api.session: the rules read caller reports when the model did not
            more, rule_type, _ = read_facts(building, tr, i, start_k=len(turn_facts))
            turn_facts += more
            suggested = suggested or rule_type
        facts = facts + turn_facts
        incident_type = incident_type or suggested
        prior = _carry(prior, out)
        probs, _ = posterior(i)
        res.trajectory.append((i, _rank(probs, call["true_room"]), round(probs.get(call["true_room"], 0.0), 4)))

    probs, active = posterior(len(transcript) - 1)
    res.rank = _rank(probs, call["true_room"])
    res.p_true = probs.get(call["true_room"], 0.0)
    res.n_rooms = len(probs)
    # a statement that could not be placed ("suite 999" in a building without one) is shown
    # to the dispatcher as unmatched and moves nothing: it is not location evidence
    res.active_location = sum(1 for e in active if not evidence_match(building, e, vocab).unmatched)
    res.facts = len(facts)
    res.suggested_type = incident_type
    res.evidence = [_describe(e, building) for e in active]
    usage1 = _usage()
    res.requests_sent = usage1["calls"] - usage0["calls"]
    res.tokens = {k: usage1[k] - usage0[k] for k in _USAGE_KEYS if k != "calls"}
    return res


def project(calls: Sequence[dict], modes: Sequence[str], vocab: dict) -> dict:
    """Caller turns that are not in the disk caches yet (each needs at least one request).

    For the grounded interpreter only the first turn of each call can be checked without a
    request: later turns depend on what the earlier ones recorded.
    """
    import hashlib

    import parser.extract as ex

    out = {"vocab_utterances": 0, "model_calls_uncached": 0, "model_turns": 0}
    if "vocab" in modes:
        cache = ex._read_json(ex.LLM_CACHE_PATH, {})
        prompt = ex.build_system_prompt(vocab)
        texts = {t.text for c in calls for t in make_transcript(c) if t.speaker == "caller" and (t.text or "").strip()}
        out["vocab_utterances"] = sum(
            1 for text in texts
            if hashlib.sha256(f"{ex.LLM_MODEL}\x00{prompt}\x00{text}".encode()).hexdigest() not in cache)
    if any(m in MODEL_MODES for m in modes):
        for c in calls:
            b = get_building(c["building"])
            tr = make_transcript(c)
            turns = caller_indices(tr)
            out["model_turns"] += len(turns)
            if _agent_cache_entry(b, tr[: turns[0] + 1], turns[0], [], [], None, vocab) is None:
                out["model_calls_uncached"] += 1
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt(x: Optional[float], pct: bool = True) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.3f}"


def _secs(values: Sequence[float]) -> Optional[dict]:
    if not values:
        return None
    vs = sorted(values)
    return {"mean": statistics.mean(vs), "median": statistics.median(vs),
            "p90": vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))], "max": vs[-1], "n": len(vs)}


def summarize(results: Sequence[CallResult], mode: str) -> dict:
    rs = [r for r in results if r.mode == mode and r.status == "ok"]
    loc = [r for r in rs if not r.no_info]
    ni = [r for r in rs if r.no_info]
    n = len(rs)
    tok_in = sum(r.tokens.get("input_tokens", 0) + r.tokens.get("cache_read_input_tokens", 0)
                 + r.tokens.get("cache_creation_input_tokens", 0) for r in rs)
    tok_out = sum(r.tokens.get("output_tokens", 0) for r in rs)
    turn_requests = [x for r in rs for x in r.turn_requests]
    return {
        "mode": mode,
        "n": n,
        "n_location": len(loc),
        "n_no_info": len(ni),
        "top1": (sum(r.rank == 1 for r in loc) / len(loc)) if loc else None,
        "top3": (sum(r.rank <= 3 for r in loc) / len(loc)) if loc else None,
        "p_true": (sum(r.p_true for r in loc) / len(loc)) if loc else None,
        "no_info_empty": (sum(r.active_location == 0 for r in ni) / len(ni)) if ni else None,
        "req_sent_per_call": (sum(r.requests_sent for r in rs) / n) if n else None,
        "req_produce_per_call": (sum(r.requests_to_produce for r in rs) / n) if n else None,
        "req_per_turn": (sum(turn_requests) / len(turn_requests)) if turn_requests else None,
        "one_request_turns": (sum(1 for x in turn_requests if x == 1) / len(turn_requests)) if turn_requests else None,
        "tok_in_per_call": (tok_in / n) if n else None,
        "tok_out_per_call": (tok_out / n) if n else None,
        "secs_per_turn": _secs([s for r in rs for s in r.turn_secs]),
        "fallback": sum(1 for r in results if r.mode == mode and r.status == "fallback"),
    }


def print_table(results: Sequence[CallResult], modes: Sequence[str], title: str) -> list[dict]:
    print(title)
    head = (f"{'mode':<6} {'calls (loc+none)':>16} {'top-1':>6} {'top-3':>6} {'p(true)':>8} {'no-info empty':>13} "
            f"{'req/call (sent)':>16} {'req/turn':>8} {'1-req turns':>11} {'tok/call in/out':>16} "
            f"{'s/turn mean/med/p90':>20} {'fallback':>8}")
    print(head)
    print("-" * len(head))
    rows = []
    for m in modes:
        s = summarize(results, m)
        rows.append(s)
        calls = f"{s['n']} ({s['n_location']}+{s['n_no_info']})"
        req = "-" if s["req_produce_per_call"] is None else f"{s['req_produce_per_call']:.2f} ({s['req_sent_per_call']:.2f})"
        rpt = "-" if s["req_per_turn"] is None else f"{s['req_per_turn']:.2f}"
        one = _fmt(s["one_request_turns"])
        tok = "-" if s["tok_in_per_call"] is None else f"{s['tok_in_per_call']:.0f}/{s['tok_out_per_call']:.0f}"
        st = s["secs_per_turn"]
        secs = "-" if st is None else f"{st['mean']:.2f}/{st['median']:.2f}/{st['p90']:.2f}"
        print(f"{m:<6} {calls:>16} {_fmt(s['top1']):>6} {_fmt(s['top3']):>6} {_fmt(s['p_true'], False):>8} "
              f"{_fmt(s['no_info_empty']):>13} {req:>16} {rpt:>8} {one:>11} {tok:>16} {secs:>20} {s['fallback']:>8}")
    return rows


def print_calls(results: Sequence[CallResult], calls: Sequence[dict], modes: Sequence[str], verbose: bool) -> None:
    by = {(r.call_id, r.mode): r for r in results}
    for c in calls:
        tag = "no-info" if c.get("no_info") else c["true_room"]
        print(f"{c['id']} [{c['building']}] true={tag}")
        for m in modes:
            r = by.get((c["id"], m))
            if r is None:
                continue
            line = (f"    {m:<6} rank {r.rank}/{r.n_rooms}  p={r.p_true:.3f}  evidence {r.active_location}  facts {r.facts}"
                    f"  req {r.requests_to_produce} ({r.requests_sent} sent)")
            if r.turn_secs:
                line += "  s/turn " + ",".join(f"{s:.1f}" for s in r.turn_secs)
            if r.status == "fallback":
                line += f"  FALLBACK: {r.note}"
            print(line)
            if verbose and r.evidence:
                print("           " + "; ".join(r.evidence))


def print_scenarios(results: Sequence[CallResult], calls: Sequence[dict], modes: Sequence[str]) -> None:
    by = {(r.call_id, r.mode): r for r in results}
    print("Scripted scenarios: rank / probability of the true room after each caller turn (last = end of call)")
    for c in calls:
        print(f"{c['id']} true={c['true_room']}")
        for m in modes:
            r = by.get((c["id"], m))
            if r is None:
                continue
            steps = "  ".join(f"[{i}] {rank}/{p:.3f}" for i, rank, p in r.trajectory)
            print(f"    {m:<6} {steps}   end: rank {r.rank}/{r.n_rooms} p={r.p_true:.3f}"
                  f"  suggested {r.suggested_type}  req {r.requests_to_produce}"
                  + (f"  FALLBACK: {r.note}" if r.status == "fallback" else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default="rules,llm", help="comma list of rules,llm,agent,vocab")
    ap.add_argument("--limit", type=int, default=None, help="only the first N calls (after --subset / --ids)")
    ap.add_argument("--subset", choices=("all", "live", "original", "added"), default="all",
                    help="'live': calls marked live: true; 'original': the first 32 calls (added: false); "
                         "'added': the calls added on 2026-11-14")
    ap.add_argument("--ids", default=None, help="comma list of call ids, run in this order")
    ap.add_argument("--calls", default=str(CALLS_PATH), help="calls YAML")
    ap.add_argument("--no-scenarios", action="store_true", help="leave the scripted console scenarios out")
    ap.add_argument("--verbose", "-v", action="store_true", help="print the active evidence per call and mode")
    ap.add_argument("--json", default=None, help="write per-call results and the tables to this file")
    ap.add_argument("--project-only", action="store_true", help="print how many turns are not cached and exit")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown mode(s) {bad}; choose from {MODES}")

    calls = load_calls(Path(args.calls))
    if args.subset == "live":
        calls = [c for c in calls if c.get("live")]
    elif args.subset == "original":
        calls = [c for c in calls if not c.get("added")]
    elif args.subset == "added":
        calls = [c for c in calls if c.get("added")]
    if args.ids:
        want = [i.strip() for i in args.ids.split(",") if i.strip()]
        by_id = {c["id"]: c for c in calls + scenario_calls()}
        missing = [i for i in want if i not in by_id]
        if missing:
            ap.error(f"unknown call id(s) {missing}")
        calls = [by_id[i] for i in want]
    if args.limit is not None:
        calls = calls[: args.limit]
    scenarios = [] if (args.no_scenarios or args.ids or args.limit is not None) else scenario_calls()
    scenarios += [c for c in calls if c.get("scenario")]
    calls = [c for c in calls if not c.get("scenario")]
    check_calls(calls + scenarios)

    from parser.extract import load_vocab

    vocab = load_vocab()
    live = [m for m in modes if m != "rules"]
    usage_before: dict = {}
    if live:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:  # pragma: no cover - python-dotenv is a dependency
            pass
        import parser.agent as ag
        import parser.extract as ex

        usage_before = dict(ex.llm_usage())
        proj = project(calls + scenarios, modes, vocab)
        cap = ex.request_budget()
        print(f"Model settings: {ag.model_settings()}; model {ex.LLM_MODEL}; "
              f"request cap: {'none' if cap is None else cap} (LLM_MAX_CALLS).")
        print(f"Not cached yet: {proj['model_calls_uncached']} of {len(calls) + len(scenarios)} calls for the grounded "
              f"interpreter ({proj['model_turns']} caller turns in all, usually one request each); "
              f"{proj['vocab_utterances']} utterances for the vocabulary extractor.")
        if args.project_only:
            return 0

    results: list[CallResult] = []
    for c in calls + scenarios:
        for m in modes:
            results.append(run_call(c, m, vocab))

    print()
    print_calls(results, calls, modes, args.verbose)
    print()
    tables: dict[str, list[dict]] = {}
    call_ids = {c["id"] for c in calls}
    original = {c["id"] for c in calls if not c.get("added")}
    added = call_ids - original
    if original and added:
        tables["original"] = print_table([r for r in results if r.call_id in original], modes,
                                         f"Original calls ({len(original)})")
        print()
        tables["added"] = print_table([r for r in results if r.call_id in added], modes,
                                      f"Calls added 2026-11-14 ({len(added)})")
        print()
    tables["all"] = print_table([r for r in results if r.call_id in call_ids], modes, f"All free-form calls ({len(call_ids)})")
    print("Accuracy columns cover location calls only; 'no-info empty' covers the no-information calls. Calls with a "
          "fallback are left out of every column except their own. s/turn is time spent in model requests.")
    if scenarios:
        print()
        print_scenarios(results, scenarios, modes)
        if args.verbose:
            print()
            print_calls(results, scenarios, modes, True)
    if live:
        import parser.extract as ex

        after = dict(ex.llm_usage())
        diff = {k: int(after.get(k, 0) or 0) - int(usage_before.get(k, 0) or 0) for k in _USAGE_KEYS}
        print()
        print(f"This run sent {diff['calls']} Anthropic request(s). Tokens: input {diff['input_tokens']}, "
              f"cache read {diff['cache_read_input_tokens']}, cache write {diff['cache_creation_input_tokens']}, "
              f"output {diff['output_tokens']}.")
        print(f"data/processed/llm_usage.json calls: {usage_before.get('calls', 0)} -> {after.get('calls', 0)}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "tables": tables,
            "calls": [r.__dict__ for r in results],
        }, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

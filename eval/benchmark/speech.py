"""Held-out speech test: Claude Haiku writes fresh caller speech, the app reads it end to end.

    uv run python -m eval.benchmark.speech --per-cell 8 --threads 8

For each building x call type, simulated callers are drawn exactly as in the main benchmark
(true room, phone fix, the structured clues the caller would give). Claude Haiku then turns
those clues into a short, messy 911 dialogue in the caller persona's style - it is told to
convey ONLY those facts, so the information content equals the structured test. The app's
interpreter (rules, and the grounded Claude reader the console uses) reads the dialogue
turn by turn, the parsed evidence is fused, and the search plan is timed as in the benchmark.
The reader's model is ``parser.agent``'s (``LLM_MODEL``, default Claude Sonnet 5.5 since 2026-11-15);
the run in eval/results/REPORT.md (14 November) used Claude Haiku as the reader.

This measures how much of the app's benefit survives real-sounding speech: same callers,
three readings (structured clues / rules parser / Haiku parser).

Generated dialogues are cached in eval/results/speech_calls.json; the interpreter's replies
are cached by parser.agent as usual, so a rerun sends nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eval.benchmark.common import (
    BUILDINGS, CALL_TYPES, CALL_TYPE_BY_KEY, curb_time, engine, run_arms, sample_world,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
GEN_PATH = RESULTS / "speech_calls.json"
OUT_PATH = RESULTS / "speech_results.json"
GEN_MODEL = "claude-haiku-5-5"  # writes the dialogues; a different job from the interpreter under test
_gen_lock = threading.Lock()
_plan_lock = threading.Lock()  # the planner's graph cache holds one building at a time

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["dispatcher", "caller"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["turns"],
    "additionalProperties": False,
}

GEN_SYSTEM = """You write realistic transcripts of 911 calls for testing a dispatch system.

You get the emergency, a caller persona, and a list of LOCATION FACTS that the caller knows. Write the
part of the call after the address has been confirmed, where the dispatcher works out where in the
building the caller is.

Rules:
- Start with a dispatcher turn. Use 3 to 7 turns in all, alternating dispatcher and caller.
- The caller must get across every location fact, in their own everyday words, in the persona's style:
  fillers, emotion, repetition, half sentences, talking about the emergency. Spread the facts over
  several caller turns; do not read them out as a list.
- Never change a fact. If a fact is marked "not sure", the caller hedges it ("I think", "maybe").
- Do not add any other location detail: no room numbers, names, signs, compass directions, left or
  right, landmarks or views that are not listed. If there are no location facts, the caller cannot
  say where they are inside the building.
- The dispatcher asks short, neutral questions and does not guess the location.
"""


def _human(s: str) -> str:
    return s.replace("_", " ")


def fact_lines(b, clues, vocab) -> list[str]:
    conf = vocab.get("default_confidence") or {}
    views = vocab.get("views") or {}
    lines = []
    for c in clues:
        # A clue below its vocabulary default confidence was hedged by the persona (sample_caller.truthful_clues);
        # 0.7 is the same fallback default that function uses, 1e-6 absorbs its rounding to 3 decimals.
        unsure = float(c.confidence) < float(conf.get(c.type, 0.7)) - 1e-6
        if c.type == "room_type":
            s = f"You are in a {_human(c.value)}" + (" (an office suite)" if c.value == "suite" else "") + "."
        elif c.type == "floor":
            s = f"You are on floor {c.value}." if str(c.value).isdigit() else f"You are on level {c.value}."
        elif c.type == "view":
            syn = ((views.get(c.value) or {}).get("synonyms") or [_human(c.value)])[0]
            s = f"Out of the window you can see {syn}."
        elif c.type == "landmark":
            s = {"window": "The room has a window.", "stairs": "You are right by the stairs.",
                 "elevator": "You are right by the elevators."}.get(c.value, f"You are near the {_human(c.value)}.")
        elif c.type == "negative":
            what = str(c.value).split(":")[-1]
            s = {"window": "There is no window in the room.", "stairs": "There are no stairs near you.",
                 "elevator": "There is no elevator near you."}.get(what, f"There is no {_human(what)} near you.")
        else:
            continue
        lines.append(s + (" (not sure)" if unsure else ""))
    return lines


def build_calls(per_cell: int) -> list[dict]:
    from parser.extract import load_vocab
    from sim.sample_caller import load_personas

    vocab = load_vocab()
    personas = load_personas()["personas"]
    calls = []
    for b_idx, bid in enumerate(BUILDINGS):
        b = engine()["buildings"][bid]["b"]
        for c_idx, ct in enumerate(CALL_TYPES):
            for i in range(per_cell):
                # Leading 100 (arbitrary tag) gives speech callers their own streams, apart from the
                # benchmark's [variant 0-5, ...] seeds, so these callers are not copies of benchmark rows.
                rng = np.random.default_rng([100, b_idx, c_idx, i])
                w = sample_world(bid, ct, "main", rng)
                p = personas.get(w["persona"], {})
                calls.append({
                    "id": f"{bid}|{ct.key}|{i}", "building": bid, "call_type": ct.key, "i": i,
                    "true_room": w["room"].id, "persona": w["persona"],
                    "persona_text": f"{p.get('description', '')} Style: {p.get('style', '')}",
                    "facts": fact_lines(b, w["clues"], vocab),
                    "t_curb": curb_time([100, b_idx, c_idx, i]),
                    "_world": w,
                })
    return calls


def generate(call: dict, client) -> list[dict]:
    ct = CALL_TYPE_BY_KEY[call["call_type"]]
    facts = "\n".join(f"- {f}" for f in call["facts"]) or "- (none: the caller does not know where they are inside the building)"
    user = (f"Emergency: {ct.complaint}.\nCaller persona: {call['persona_text']}\n\nLOCATION FACTS:\n{facts}\n\n"
            "Write the transcript.")
    # max_tokens is generous headroom for a 3-7 turn dialogue; the system prompt is cached across calls.
    resp = client.messages.create(
        model=GEN_MODEL, max_tokens=4000,
        system=[{"type": "text", "text": GEN_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": GEN_SCHEMA}},
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("refusal")
    text = next(b.text for b in resp.content if b.type == "text")
    turns = json.loads(text)["turns"]
    if not any(t["speaker"] == "caller" for t in turns):
        raise RuntimeError("no caller turn")
    return turns


def interpret(call: dict, turns: list[dict], mode: str) -> tuple[list, list[str]]:
    """Active location evidence after reading the dialogue in ``mode``, and per-turn sources."""
    from eval.run_agent_eval import _carry, get_building, make_transcript
    from parser.extract import load_vocab
    from parser.interpret import interpret_turn

    building = get_building(call["building"])
    transcript = make_transcript({"turns": turns})
    vocab = load_vocab()
    prior: list = []
    sources = []
    itype = CALL_TYPE_BY_KEY[call["call_type"]].situation["incident_type"]
    for i, turn in enumerate(transcript):
        if turn.speaker != "caller":
            continue
        out = interpret_turn(building, transcript[: i + 1], i, mode, prior_evidence=prior, vocab=vocab,
                             prior_facts=[], incident_type=itype)
        sources.append(out.source)
        prior = _carry(prior, out)
    return [e for e in prior if e.status == "active"], sources


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # Calls per building x call type. REPORT.md's 280-call run used --per-cell 10 (28 cells x 10).
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)  # concurrent API requests / evaluations
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    import anthropic

    from eval.run_agent_eval import get_building

    # parser.agent keeps its own building cache; make it share ours so evidence ids line up.
    from eval import run_agent_eval as rae
    for bid, E in engine()["buildings"].items():
        rae._BUILDINGS[bid] = E["b"]
    assert get_building("ptc-100") is engine()["buildings"]["ptc-100"]["b"]

    # Retries and a 120 s timeout so one slow or rate-limited request does not sink a batch run.
    client = anthropic.Anthropic(api_key=os.getenv("LLM_API_KEY"), max_retries=4, timeout=120)
    calls = build_calls(args.per_cell)
    cache = json.loads(GEN_PATH.read_text(encoding="utf-8")) if GEN_PATH.exists() else {}
    todo = [c for c in calls if c["id"] not in cache]
    print(f"{len(calls)} calls; generating {len(todo)} dialogues with {GEN_MODEL}", flush=True)
    t0 = time.time()

    def gen_one(c):
        try:
            turns = generate(c, client)
        except Exception as exc:  # noqa: BLE001
            print("  generation failed:", c["id"], repr(exc)[:200], flush=True)
            return
        with _gen_lock:
            cache[c["id"]] = {"facts": c["facts"], "persona": c["persona"], "turns": turns}
            GEN_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(gen_one, todo))
    print(f"  generated in {time.time() - t0:.0f}s", flush=True)

    results = []
    lock = threading.Lock()

    def eval_one(c):
        if c["id"] not in cache:
            return
        turns = cache[c["id"]]["turns"]
        w = c["_world"]
        ct = CALL_TYPE_BY_KEY[c["call_type"]]
        row = {k: c[k] for k in ("id", "building", "call_type", "i", "true_room", "persona", "t_curb")}
        row["n_facts"] = len(c["facts"])
        with _plan_lock:
            structured = run_arms(c["building"], ct, w["room"], w["fix"], w["clues"], c["t_curb"])
        row.update({f"structured_{k}": structured[k] for k in ("t_A", "t_B", "t_C", "rank_true", "p_true")})
        for mode in ("rules", "llm"):
            evidence, sources = interpret(c, turns, mode)
            with _plan_lock:
                r = run_arms(c["building"], ct, w["room"], w["fix"], w["clues"], c["t_curb"], evidence=evidence)
            row.update({f"{mode}_{k}": r[k] for k in ("t_C", "rank_true", "p_true")})
            row[f"{mode}_n_evidence"] = len(evidence)
            # True when any turn in llm mode was read by the rules instead (model error or budget).
            row[f"{mode}_fallback"] = mode == "llm" and any(s != "llm" for s in sources)
        with lock:
            results.append(row)
            if len(results) % 20 == 0:
                print(f"  evaluated {len(results)}  {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(eval_one, calls))
    OUT_PATH.write_text(json.dumps(sorted(results, key=lambda r: r["id"]), indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(results)} calls) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Reader model comparison (2026-11-15)

## Update: overall conclusions, fewer preset rules (Sonnet 5.5)

The reader now states its overall judgment (`record_conclusion`: rooms plus a probability it
chooses, resting on words from any turn; applied last in `core.fusion`), the pinpointing-clue rule
is gone, and the prompt asks for calibrated rather than cautious answers. 7 fuzzy multi-turn calls
were added (vague fragments that place the caller only together, e.g. "up high" / "far from where I
came in" / "the name starts with a B"). 78 calls in all; `freeform_parser.txt` is the low-effort run.

| Reader | All 78: top-1 | top-3 | p(true) | Layout 14: top-1 / top-3 / p | Fuzzy 7: top-1 / top-3 / p | s/turn mean / p90 | Run cost |
|---|---|---|---|---|---|---|---|
| Rules | 54% | 73% | 0.537 | 8% / 31% / 0.10 | 0% / 14% / 0.02 | - | $0 |
| **Sonnet 5.5, effort low (default)** | **93%** | **99%** | **0.810** | 85% / 100% / 0.685 | 86% / 86% / 0.718 | **3.2 / 5.0** | $1.89 |
| Sonnet 5.5, effort medium | 93% | 100% | 0.811 | 85% / 100% / 0.713 | 86% / 100% / 0.698 | 3.5 / 6.4 | $1.70 |

Low and medium effort tie within one call; low stays the default for speed (`LLM_EFFORT=medium`
to switch). The jrr "up high / far / starts with B" call ends at Bendheim Center for Finance
Level 3 (p 0.65, rank 1). Remaining misses: the side-door call (the model read the stairs the
caller came in by as where they are; medium effort recovers it to rank 2), and an unlabelled
suite next to a top-floor tenant that the model read as a weak top-floor hint (p 0.4) in a call
labelled no-information.

## First comparison (fixed rules for clues, before conclusions)

`uv run python -m eval.run_agent_eval --modes rules,llm` with `LLM_MODEL` set per row, effort `low`,
adaptive thinking, on the 71 free-form calls (32 original, 25 added 2026-11-14, 14 layout-reasoning
calls added 2026-11-15). Raw model replies are in `data/processed/llm_agent_cache.json`; the scores
below come from the final validation code (pinpointing clues, `PINPOINT_OTHERS_FIT = 0.1`).

| Reader | All 71: top-1 | top-3 | p(true) | Layout 14: top-1 | top-3 | p(true) | s/turn mean / p90 | Cost of the run |
|---|---|---|---|---|---|---|---|---|
| Rules (no model) | 60% | 79% | 0.594 | 8% | 31% | 0.100 | - | $0 |
| Claude Opus 5.5 | 90% | 95% | 0.764 | 77% | 85% | 0.468 | 3.7 / 6.2 | $2.24 |
| **Claude Sonnet 5.5 (default)** | **90%** | **97%** | **0.784** | **85%** | **100%** | **0.651** | **2.9 / 4.7** | **$1.21** |

On the older 57 calls Opus is slightly ahead (original 32: top-1 93% vs 89%; added 25: p(true) 0.86 vs
0.82); on the layout calls and overall Sonnet is ahead, and it is faster and half the price. The margins
are one to two calls, so a re-run could reorder them. The old fixed-vocabulary Claude Haiku 5.5 reader
scored 94% / 100% / 0.879 on the 57 older calls at 1.7 s per turn, and was not run on the layout calls.

No fallback to the rules happened in either run. Remaining weak spots: a room number heard through
speech recognition ("three oh seven") or a name the span does not carry word for word is capped at a
share of 0.7 by the room-name check; ambiguous descriptions (two corner offices by the stairs) leave
the true room ranked 2nd to 5th.

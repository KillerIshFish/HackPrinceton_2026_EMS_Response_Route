# Reader model comparison (2026-11-15)

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

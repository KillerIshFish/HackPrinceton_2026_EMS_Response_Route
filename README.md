# 911 Caller Location Predictor

When a 911 caller is somewhere inside a large building, the phone's location fix says
"this address, roughly this height" and the caller says things like *"I'm locked in a
bathroom"* or *"I can see the parking lot"*. This project fuses those signals into a
probability for every room, shows the dispatcher where the caller most likely is, and
gives responders a search order that minimises the expected time to find them.

**Status: v0.** Everything runs locally on a synthetic 4-floor office building with a
scripted demo call and a rule-based clue parser. No external API is called and no real
dataset is read yet (see [What is stubbed](#what-is-stubbed)).

## Quick start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
# 1. Backend (from the repo root)
uv sync
uv run uvicorn api.server:app --reload --port 8000

# 2. Frontend (second terminal)
cd web
npm install
npm run dev          # http://localhost:5173
```

Open http://localhost:5173 and press **Next Call Turn** to walk through the demo:

| Turn | Evidence | What you should see |
|---|---|---|
| 0 | Call connected, address only | Prior spread across all four floors |
| 1 | Phone barometric z-axis fix (12.6 m) | Probability narrows to floors 3 and 4 |
| 2 | "I'm locked in a bathroom" | Two peaks on floor 4 (the two restrooms) |
| 3 | "I can see the parking lot out a small window" | Mass concentrates on the parking-lot-facing restroom; the search plan updates and E[T] drops below both baselines |

Every stop in the search plan has a **Mark Cleared** button. It tells the backend the
room was searched and found empty, which down-weights that room and replans.

Other commands:

```bash
uv run pytest                                    # end-to-end demo acceptance tests
uv run python -m eval.run_eval --n 200 --seed 0  # simulated-caller metrics table
uv run python -m ingest.inventory                # which raw data folders are populated
```

Without uv: `python -m venv .venv`, activate it, `pip install -e .`, then drop the
`uv run` prefix.

## How it works

```
call transcript ──► parser/extract ──► clues ─┐
phone fix (x, y, z) ──────────────────────────┼─► core/fusion ──► Posterior ──► planning ──► SearchPlan
building model (ingest/) ──► core/prior ──────┘                       │
                                                             core/uncertainty
```

**Prior** (`core/prior.py`): `π(r) ∝ Area^0.7 · w(room type, building use, hour) · occupancy`,
with 5% of the mass reserved for "caller is somewhere not in our model".

**Phone likelihood** (`core/phone_likelihood.py`): a horizontal Gaussian on distance to
the room, and a vertical Gaussian comparing barometric height to `floor height + 1.2 m`
with `σ = 3 m / 1.28 ≈ 2.34 m` (the ±3 m z-axis bound read as an 80% interval). A missing
horizontal or vertical reading contributes a flat likelihood of 1.

**Clue likelihood** (`core/clue_likelihood.py`): each clue becomes a match score
`m(r) ∈ [0, 1]` per room, driven by `configs/clue_vocab.yaml`.

**Fusion** (`core/fusion.py`):

```
p(r | E) ∝ π(r) · L_phone(r) · Π_i [ε + (1 − ε) · m_i(r)]^c_i        ε = 0.1
```

The `ε` floor means no single clue can zero a room out (callers are often wrong), and
the exponent `c_i` tempers each clue by the parser's confidence in it. The
"not in model" bucket is scored at the prior-weighted average of the rooms, so it keeps
a roughly constant share unless the evidence fits no room well.

**Uncertainty** (`core/uncertainty.py`): entropy in bits, per-floor probability, the
smallest set of rooms covering 90%, and the number of separate high-probability
clusters ("peaks").

**Search plan** (`planning/`): travel times are shortest paths on the room graph
(1.2 m/s walking, +15 s per floor of stairs); room search time grows with area. The
planner orders rooms greedily by `p_r / (travel + search time)`, refines with 2-opt to
minimise `E[T] = Σ p_r · C_r`, can split rooms across teams, and reports E[T] next to
nearest-first and highest-probability-first baselines. **Mark Cleared** applies the
Koopman update `p_r ← p_r · (1 − 0.9)` (a search finds a present person 90% of the
time), renormalises, and replans from the room the crew is standing in.

## Repo layout and ownership

Each folder is meant to be owned by one person so work does not collide.
`core/schema.py` is the frozen contract between all of them.

| Path | Purpose |
|---|---|
| `core/schema.py` | **Frozen contract**: `Building`, `Room`, `Clue`, `Posterior`, `Stop`, `SearchPlan` (+ `PhoneFix`) |
| `core/` | Prior, likelihoods, fusion, uncertainty, building graph |
| `parser/` | Transcript → clues. v0 is keyword/regex; the LLM prompt is ready in `parser/prompts/` |
| `planning/` | Search order, baselines, entrance and staging recommendation |
| `ingest/` | Building data. v0: synthetic generator in `matterport.py`; the rest are stubs |
| `sim/` | Scripted demo call, simulated callers and personas |
| `eval/` | Floor accuracy, top-k, 90% coverage, E[T]; `run_eval.py` CLI |
| `api/server.py` | FastAPI app, holds the single in-memory demo call |
| `web/` | React + Vite + Tailwind dispatcher console |
| `configs/` | `priors.yaml`, `clue_vocab.yaml`, `demo_buildings.yaml` — tune these, not code |
| `data/` | Gitignored. Raw dataset drop folders |
| `tests/` | End-to-end test of the demo flow through the API |

## API

Base URL `http://127.0.0.1:8000` (interactive docs at `/docs`). The web dev server
proxies `/api/*` to it.

| Method | Path | Returns |
|---|---|---|
| GET | `/buildings`, `/buildings/{id}` | Building model(s) |
| GET | `/calls/demo/state` | `DemoState` |
| POST | `/calls/demo/step` | `DemoState` after revealing the next turn |
| POST | `/calls/demo/reset` | `DemoState` back at turn 0 |
| POST | `/calls/demo/teams?n=2` | `DemoState` with the plan split across `n` teams (1–3) |
| GET | `/posterior` | `Posterior` |
| GET | `/plan?teams=1` | `SearchPlan` |
| POST | `/rooms/{room_id}/clear` | `DemoState` after the Koopman update and replan |

`DemoState` bundles the transcript with extracted clues, the posterior, the plan, the
approach recommendation, a suggested next question, and a per-turn history.

## Configuration, keys and data

Copy `.env.example` to `.env` (gitignored). v0 reads none of these values.

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `parser/extract.py` | `openai`, `anthropic` or `gemini` |
| `LLM_API_KEY` | `parser/extract.py` | Key for that provider |
| `MATTERPORT_DATA_DIR` | `ingest/matterport.py` | Default `./data/raw/matterport` |
| `VEXCEL_DATA_DIR` | `ingest/vexcel.py` | Default `./data/raw/vexcel` |
| `COSTAR_DATA_DIR` | `ingest/costar.py` | Default `./data/raw/costar` |

Drop raw files into the matching `data/raw/<source>/` folder; derived artefacts go in
`data/processed/`. Nothing under `data/` is committed.

## What is stubbed

| Area | v0 behaviour | Real version needs |
|---|---|---|
| Clue extraction (`parser/extract.py`) | Keyword/regex over `clue_vocab.yaml` | LLM key; prompt is in `parser/prompts/extract_clues.md` |
| Matterport ingest | Synthetic "100 Princeton Tech Center" | Scan exports in `data/raw/matterport/` |
| Vexcel aerial layer | Schematic site plan in the UI | Aerial tiles; window-ray intersection for view clues |
| CoStar occupancy | Hand-set `occupancy_factor` | Tenant/occupancy export |
| Geo-registration | Simple local ↔ lat/lng offset from a config origin | Per-building alignment |
| Call audio | Scripted text turns | Live transcript feed |

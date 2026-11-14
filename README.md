# 911 Caller Location Predictor

When a 911 caller is somewhere inside a large building, the phone's location fix says
"this address, roughly this height" and the caller says things like *"I'm locked in a
bathroom"* or *"I can see the parking lot"*. This project fuses those signals into a
probability for every room, shows the dispatcher where the caller most likely is, and
gives responders a search order that minimises the expected time to find them.

**Status.** Two buildings run through the same pipeline:

| Building | What it is | Data |
|---|---|---|
| **Bundy Olympic** (`bundy-olympic`), 11900 W Olympic Blvd, Los Angeles | A real 8-storey office building | CoStar suites and tenants, eight Matterport suite scans, Vexcel aerial imagery |
| **100 Princeton Tech Center** (`ptc-100`) | A synthetic 4-floor office | Generated from `configs/demo_buildings.yaml`; needs no keys or data |

The real building appears once it has been built (see [Real building](#real-building)).
Without it, the app starts on the synthetic one and needs nothing but this repo.

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

Open http://localhost:5173 and press **Next Call Turn** to walk through the call. The
building menu in the header switches between buildings.

Real building (caller is in a private office of Suite 860 on floor 8):

| Turn | Evidence | What you should see |
|---|---|---|
| 0 | Call connected, address only | Prior spread over the tenant suites on six office floors |
| 1 | Phone barometric z-axis fix (26.4 m) | Floors 8 and 7 hold 94% |
| 2 | "I'm in an office on the top floor, I locked the door" | Floor 8 suites lead |
| 3 | "I can see the parking structure out the window" | Parking-side suites on floor 8 lead; the scanned office is stop 2 of the plan |

Synthetic building:

| Turn | Evidence | What you should see |
|---|---|---|
| 0 | Call connected, address only | Prior spread across all four floors |
| 1 | Phone barometric z-axis fix (12.6 m) | Probability narrows to floors 3 and 4 |
| 2 | "I'm locked in a bathroom" | Two peaks on floor 4 (the two restrooms) |
| 3 | "I can see the parking lot out a small window" | Mass concentrates on the parking-lot-facing restroom; E[T] drops below both baselines |

Things to try in the console:

- **Caller said**: type anything the caller says; it is parsed and the posterior updates.
- **Parser: Rules / Claude Haiku**: keyword rules work offline; Claude Haiku understands
  free-form speech ("some kind of janitor's cupboard, nothing to look out of").
- **Mark Cleared** on a stop: the room was searched and found empty, so it is
  down-weighted and the plan is rebuilt from where the crew is standing.
- **3D scan** (real building): Matterport's viewer shows the scan of the most probable
  scanned room and moves there as the posterior changes. The **3D** button on a stop, or a
  click on a scanned room in the floor plan, pins a room instead.
- **Approach** (real building): Vexcel aerial imagery with the roof outline, entrances,
  staging spots and the recommended approach.

Other commands:

```bash
uv run pytest                                    # 18 tests, no network
uv run python -m eval.run_eval --n 200 --seed 0  # simulated-caller metrics table (synthetic building)
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

**Clue extraction** (`parser/extract.py`): keyword rules over `configs/clue_vocab.yaml`,
or Claude Haiku 5.5 with the prompt in `parser/prompts/extract_clues.md`. The model's
reply is constrained to a JSON schema and then validated: a clue is dropped unless its
span is verbatim in the utterance and its value is in the vocabulary. Any failure falls
back to the rules.

**Clue likelihood** (`core/clue_likelihood.py`): each clue becomes a match score
`m(r) ∈ [0, 1]` per room, driven by `configs/clue_vocab.yaml`. A room that stands for a
whole unscanned tenant suite (`room_type: suite`) matches an interior room clue
partially: "I'm in a closet" does not rule out a suite whose interior was never scanned.

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

## Real building

```bash
uv run python -m ingest.real_building bundy-olympic            # build from cached data
uv run python -m ingest.real_building bundy-olympic --refresh  # re-query Matterport and CoStar
```

This writes `data/processed/bundy-olympic.building.json` (a `Building` that validates
against the frozen schema) and `bundy-olympic.meta.json` (where each room's data comes
from, the Matterport model and sweep for each scanned room, and the site annotation).
`data/` is gitignored, so each teammate builds it once; it needs the keys below and the
CoStar kit in `data/raw/costar/costar-hackathon-kit/`.

Where each fact comes from:

| Source | Module | Supplies |
|---|---|---|
| CoStar kit (local files) | `ingest/costar.py` | Storeys, floor plate area, slab-to-slab height, suites per floor with rentable area, which suites have a current tenant (`occupancy_factor` 1.0 occupied, 0.15 vacant, 0.5 unknown), Matterport model ids |
| Matterport Model API | `ingest/matterport_api.py` | Rooms inside each scanned suite with measured area, room tags (mapped to `room_type`), window openings, sweep positions |
| Vexcel API v2 | `ingest/vexcel.py`, `configs/real_buildings.yaml` | Ortho tiles; the roof outline, facade views, entrances and staging spots were annotated from them by hand |
| Inferred | `ingest/real_building.py` | Stairwells, elevators, restrooms, corridors, the two parking storeys, and where each suite sits along the corridor |

What that means for Bundy Olympic: 148 rooms, of which 28 come from scans, 21 are whole
suites known only from CoStar, and 99 are inferred. The header of the console shows this
split and the floor-plan tooltip names each room's source.

Limits you should know about:

- **A Matterport scan covers one suite, not the building.** Eight suites are scanned;
  four of the scans are older and have room areas and tags but no wall geometry.
- **No dataset says where a suite sits on its floor.** Suites are laid out in
  suite-number order around a schematic plate with the real floor area. Which suites
  face the street and which face the parking deck is therefore an assumption, and so is
  every `window_facing` value.
- **The scans' geo-coordinates are unusable** (the eight suites of one building are
  scattered over about 100 m), so scans are not placed by their own coordinates.
- **The building core is inferred.** Restrooms, stairs and elevators are placed where an
  office tower of this shape usually has them.
- **The site annotation is by hand** and approximate to a few metres. The roof outline
  measures 1,777 m² against CoStar's 1,790 m² typical floor.

## Repo layout and ownership

Each folder is meant to be owned by one person so work does not collide.
`core/schema.py` is the frozen contract between all of them.

| Path | Purpose |
|---|---|
| `core/schema.py` | **Frozen contract**: `Building`, `Room`, `Clue`, `Posterior`, `Stop`, `SearchPlan` (+ `PhoneFix`) |
| `core/` | Prior, likelihoods, fusion, uncertainty, building graph |
| `parser/` | Transcript → clues: keyword rules and the Claude Haiku extractor; next-question ranking |
| `planning/` | Search order, baselines, entrance and staging recommendation |
| `ingest/` | Building data: synthetic generator and registry (`matterport.py`), real-building assembler (`real_building.py`), CoStar, Matterport and Vexcel clients |
| `sim/` | Scripted demo calls, simulated callers and personas |
| `eval/` | Floor accuracy, top-k, 90% coverage, E[T]; `run_eval.py` CLI |
| `api/server.py` | FastAPI app, holds the single in-memory demo call |
| `web/` | React + Vite + Tailwind dispatcher console |
| `configs/` | `priors.yaml`, `clue_vocab.yaml`, `demo_buildings.yaml`, `real_buildings.yaml`: tune these, not code |
| `data/` | Gitignored. Raw dataset folders, API caches and processed buildings |
| `tests/` | End-to-end tests through the API, plus unit tests for the new pieces |

## API

Base URL `http://127.0.0.1:8000` (interactive docs at `/docs`). The web dev server
proxies `/api/*` to it.

| Method | Path | Returns |
|---|---|---|
| GET | `/config` | Matterport browser key, whether the LLM parser and Vexcel are configured |
| GET | `/buildings`, `/buildings/{id}` | Building model(s) |
| GET | `/buildings/{id}/meta` | Room provenance, Matterport models, site annotation and aerial tile window |
| GET | `/vexcel/tile/{layer}/{z}/{x}/{y}` | One aerial tile (JPEG), only inside a building's site window |
| GET | `/calls/demo/state` | `DemoState` |
| POST | `/calls/demo/step` | `DemoState` after revealing the next scripted turn |
| POST | `/calls/demo/utterance` | `DemoState` after adding `{"text": "..."}` said by the caller |
| POST | `/calls/demo/parser?mode=rules\|llm` | `DemoState` re-parsed with that parser |
| POST | `/calls/demo/building?id=...` | `DemoState` for a new call on that building |
| POST | `/calls/demo/reset` | `DemoState` back at turn 0 |
| POST | `/calls/demo/teams?n=2` | `DemoState` with the plan split across `n` teams (1–3) |
| GET | `/posterior` | `Posterior` |
| GET | `/plan?teams=1` | `SearchPlan` |
| POST | `/rooms/{room_id}/clear` | `DemoState` after the Koopman update and replan |

`DemoState` bundles the transcript with extracted clues, the posterior, the plan, the
approach recommendation, a suggested next question, a per-turn history, the parser mode
and LLM usage, and the list of buildings.

## Configuration, keys and data

Copy `.env.example` to `.env` (gitignored). Never commit real keys.

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_PROVIDER`, `LLM_API_KEY` | `parser/extract.py` | `anthropic` and its key; only used when the parser is set to Claude Haiku |
| `LLM_MAX_CALLS` | `parser/extract.py` | Request budget per server process (default 200); the rules take over beyond it |
| `MATTERPORT_API_KEY`, `MATTERPORT_API_TOKEN` | `ingest/matterport_api.py` | Model API token id and secret, used when building a real building |
| `MATTERPORT_SDK_KEY` | web viewer, via `/config` | Browser key that lets the page steer the embedded viewer; the viewer still loads without it |
| `VEXCEL_API_KEY` | `ingest/vexcel.py` | Aerial tiles; stays on the server |
| `DEMO_BUILDING_ID` | `api/server.py` | Building the demo starts on (default: the real building if built, else `ptc-100`) |
| `MATTERPORT_DATA_DIR`, `VEXCEL_DATA_DIR`, `COSTAR_DATA_DIR` | `ingest/` | Raw data and cache folders under `data/raw/` |

What calls the network, and how often:

- **Anthropic**: one small request per new caller utterance while the parser is set to
  Claude Haiku. Replies are cached in `data/processed/llm_clue_cache.json`, so replaying
  the demo costs nothing, and totals are kept in `data/processed/llm_usage.json`.
- **Matterport**: one request per scan when a real building is built, cached in
  `data/raw/matterport/`. The embedded viewer loads from Matterport in the browser.
- **Vexcel**: one request per aerial tile (49 for Bundy Olympic), cached in
  `data/raw/vexcel/tiles/`.

## What is still stubbed

| Area | Today | Next step |
|---|---|---|
| Suite positions on a floor | Suite-number order on a schematic plate | A floor plan per storey, or matching scan window views to the aerial |
| Facade views and site annotation | Annotated by hand in `configs/real_buildings.yaml` | `classify_surroundings` / `window_ray_intersections` in `ingest/vexcel.py` |
| Unscanned suites | One `suite` room each | More scans, or a typical-suite interior model |
| Adding a building | A new entry in `configs/real_buildings.yaml` | Automatic footprint and facade extraction |
| Call audio | Scripted turns and typed utterances | Live transcript feed |
| Eval harness | Synthetic building only | Simulated callers on the real building |

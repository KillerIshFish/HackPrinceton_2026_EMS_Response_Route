# ELC — Emergency Location Console

**Find the 911 caller inside the building.** ELC turns a phone's location fix and the
caller's own words into a probability for every room, then gives responders the search
order that reaches the caller soonest.

Built at HackPrinceton 2026 · [Website](website11/index.html) · [Simulation benchmark](website11/benchmark.html)

> [!IMPORTANT]
> **Most of the real-building data needs private access.** The Matterport scans, CoStar
> property records and Vexcel aerial imagery behind the real buildings are licensed. They
> are reached with private API keys or private data files, and none of that data is in
> this repository. A fresh clone with no keys still runs the full pipeline on a synthetic
> building and on one building traced from public floor plans, with offline keyword rules
> instead of Claude. Real scanned buildings, address search, the Matterport 3D scan viewer,
> aerial imagery and the Claude reader are missing until keys and data are supplied. See
> [What runs without keys](#what-runs-without-keys).

---

## The problem

A phone fix gives an address and a rough height. The caller says things like *"I'm locked
in a bathroom"* or *"the name on the door starts with a B"*, in fragments spread across
the call. The dispatcher has to turn that into a room, and the crew has to decide where
to look first in a building they have never seen.

## Impact (simulation)

From a benchmark of **32,200 simulated 911 calls** in four buildings, timing ELC's search
plan against an informed dispatcher who relays what the caller said. Callers, entrance and
walking speeds are the same for both; only the search order differs.

| Result | Value |
|---|---|
| Median time from building entrance to patient | **10.0 → 5.8 min** |
| Median time saved per call (paired) | **3.1 min** |
| Calls where ELC reached the caller sooner | **67%** (same on 13%, slower on 20%) |
| Cardiac arrest survivors | **+15 per 1,000 cardiac arrest calls** |

This is a simulation, not field evidence. Callers, phone fixes and room-search times are
modelled. The simulation hands ELC each caller's clues already structured, so it tests the
probability model and the planner, not Claude's reading of speech. When the caller's room
is drawn uniformly instead of from ELC's own occupancy prior, the median advantage falls
to zero. Full method and limitations: [`eval/results/REPORT.md`](eval/results/REPORT.md)
and the [benchmark page](website11/benchmark.html).

## How it works

```
 Building model ──► Agent reads the call ──► Probability per room ──► Search plan
 rooms, hallways,   Claude Sonnet 5.5 reads    fused with the phone's     stop order with
 stairs, windows    each turn against a        barometric height and a    arrival times, plus
                    layout map of the building prior from room size,      the most useful
                                               type and time of day       question to ask next
```

1. **The building** (`ingest/`, `core/prior.py`). Rooms, doors, stairs and windows come
   from Matterport scans, CoStar records and aerial imagery, from published floor plans
   traced by hand, or from a model generated from an address. Before any evidence, each
   room gets a prior from its area, room type, building use and time of day.
2. **The phone** (`core/phone_likelihood.py`). The horizontal fix and barometric height
   (the z-axis) are compared with each room and floor. In the synthetic building a 12.6 m
   reading narrows the call to floors 3 and 4.
3. **What the caller says** (`parser/agent.py`, `parser/layout.py`,
   `parser/prompts/reason_call.md`). Claude Sonnet 5.5 gets the whole conversation, with
   the newest turn marked, and a layout map: every hallway with its rooms in door order,
   and per room its size rank, corner or interior position, walk to the stairs, elevator
   and entrances, windows and what they face, neighbours and tenant. It reasons from the
   map to the rooms the words fit. There is no fixed word list. It usually answers in one
   request per turn, never more than four, at low effort.
4. **Code checks the agent** (`parser/agent.py`). The agent only proposes records through
   tools (`parser/building_tools.py`). A record is dropped if its quoted words are not
   exactly in what the caller just said, if a room or floor does not exist, if the agent is
   less than 40% sure, or if a fact claims something is verified or safe. A named place
   that does not quote the room's number or name is capped at 70%. If Claude does not
   answer in time, offline keyword rules read the turn.
5. **Fusion** (`core/fusion.py`). A Bayesian update:
   `p(r | E) ∝ prior(r) · L_phone(r) · Π clue factors · stated-place factors`. A named place
   takes 40% to 95% of the probability, depending on how plainly and exactly the caller said
   it. A descriptive clue moves the odds between two rooms by at most 10 to 1. No room is
   ever ruled out, so a wrong statement only pushes the right room later in the plan.
6. **Search plan** (`planning/search_order.py`). The planner picks the stop order that
   minimises the expected time to find the caller. That is not always the likeliest room
   first: a slightly less likely room by the entrance can be checked on the way in. It uses
   an exact dynamic programme for small candidate sets, and greedy plus 2-opt/or-opt local
   search for larger ones. It is never worse than the nearest-first or
   highest-probability-first baselines, which it reports next to its own E[T]. A room
   marked **Searched** gets a Koopman update `p ← p · (1 − q)`; the remaining stops keep
   their order.
7. **The emergency type changes the route** (`situation/`, `configs/situations.yaml`).
   For a confirmed cardiac arrest the planner maximises the chance of reaching the patient
   while they can still be saved (Valenzuela 1997 and De Maio 2003 survival curves) instead
   of minimising time. Stroke, internal bleeding and fire have a constant per-minute cost,
   so the fastest route stays the best one and the console shows the cost. With smoke
   reported, crews move at 0.5 m/s, search along the walls, and are never offered the
   elevator. For an active threat, route planning is off.
8. **Next question** (`parser/next_question.py`). The console suggests the question whose
   answer would split the remaining rooms most (highest expected information gain), such as
   *"What number is on the door nearest you?"*

## Key features

- **Dispatcher console** (React + Vite): floor-plan heatmap, per-floor probability bar,
  live transcript with every record tied to the caller's highlighted words, and a search
  plan with arrival times.
- **Claude Sonnet 5.5 layout reader** that handles descriptions like "the last door past
  the stairs" or "a big corner room", with offline keyword rules as the fallback. In the
  team's reader comparison (`eval/results/model_comparison.md`), Sonnet placed callers as
  reliably as Opus, about a second faster per turn, at about half the cost.
- **Dispatcher stays in control**: reject or restore any record, confirm the emergency type
  before it changes the plan, mark rooms **Searched** (rapid or thorough) or **Found here**,
  close rooms, undo.
- **Crew facts**, such as "smoke under the door" or "my coworker will meet you out front",
  are listed as reported by the caller, never as verified.
- **Emergency-aware routing**: survival-maximising plans for cardiac arrest, smoke search
  rates, and elevator routing for medical calls when a ride beats the stairs (75 s per ride
  plus 2 s per floor). Settings live in `configs/situations.yaml`, each with its source.
- **Multi-team plans**: the plan can be split across up to three teams.
- **Views**: Floor plan, 3D (any building, coloured by probability), Scan (Matterport
  viewer that follows the most probable scanned room), and Approach (aerial imagery with
  entrances and staging spots).
- **Scripted scenarios** from the header menu, including cardiac arrest in an office, smoke
  on the third floor, and an active threat with two callers. **Next call turn** steps
  through them.
- **Building generation at call time** (`ingest/unified/`): an address, property ID or
  Matterport model ID becomes a building model in seconds.
- **Simulation benchmark** (`eval/benchmark/`) and **565 tests**.

## Buildings

| Building | Source | Needs |
|---|---|---|
| **100 Princeton Tech Center** (`ptc-100`) | Synthetic 4-floor office, 100 rooms (`configs/demo_buildings.yaml`) | Nothing |
| **Julis Romo Rabinowitz Building** (`jrr-princeton`), Princeton | 208 rooms on five levels, traced from the university's published floor plans (`configs/plan_buildings.yaml`) | One build command, no keys |
| **5285 W Washington Blvd** (`washington-5285`), Los Angeles | 2-storey office, 38 rooms from a single whole-building Matterport scan | Matterport, Vexcel, CoStar data |
| **Bundy Olympic** (`bundy-olympic`), Los Angeles | 8-storey tower, 148 rooms: eight scanned suites, CoStar suites, inferred core | Matterport, Vexcel, CoStar data |
| **Any kit property** | Generated from an address or property ID (`ingest/unified/`) | CoStar kit |

## Repository structure

```
api/            FastAPI server (server.py) and the in-memory call session (session.py)
core/           schema.py (frozen data contract), prior, phone likelihood, fusion, uncertainty, building graph
parser/         Claude layout reader (agent.py, layout.py, building_tools.py, prompts/), keyword rules, next question
planning/       search order (search_order.py, exact.py), baselines, entrance and staging approach
situation/      emergency types: travel rates, survival curves, routing policy
ingest/         building data: synthetic generator, Matterport / CoStar / Vexcel clients
  mp3d/         one Matterport scan -> surveyed 3D model
  unified/      address, property ID, scan or traced plan -> building model at call time
configs/        priors, clue vocabulary, situations, statement weights, demo and traced buildings
sim/            scripted calls, simulated callers and personas
eval/           metrics, reader comparison, simulation benchmark (benchmark/) and results (results/)
web/            dispatcher console (React 19, Vite, Tailwind, three.js)
website11/      project website (index.html, benchmark.html), deployed with Vercel
tests/          pytest suite
docs/           build plan, OpenStreetMap site-pack notes
data/           gitignored: licensed raw data, API caches, built buildings
```

## Quickstart

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
# 1. Backend, from the repo root
uv sync
uv run uvicorn api.server:app --reload --port 8000     # API docs at http://127.0.0.1:8000/docs

# 2. Console, in a second terminal
cd web
npm install
npm run dev                                            # http://localhost:5173
```

Open http://localhost:5173 and press **Next call turn**. On the synthetic building the
scripted call goes:

| Turn | Evidence | Result |
|---|---|---|
| 0 | Call connected, address only | Prior spread across all four floors |
| 1 | Phone height 12.6 m | Floors 3 and 4 hold 99% |
| 2 | "I'm locked in a bathroom" | Two peaks on floor 4: the two restrooms |
| 3 | "I can see the parking lot out a small window" | Parking-lot restroom at 51% and stop 1; E[T] 251 s against 294 s and 817 s for the baselines |

Type anything into the **Caller said** box to add a free-form turn.

**Add the traced Princeton building** (no keys; about 30 s, most of it a public
OpenStreetMap query):

```bash
uv run python -m ingest.unified plan:jrr-princeton
```

It then appears in the building menu in the header.

**Other commands**

```bash
uv run pytest                                     # all data present: 565 pass, 18 skip
                                                  # no private data: 551 pass, 27 skip, 5 error
                                                  #   (tests/test_statement_weight.py needs bundy-olympic)
uv run python -m eval.run_eval --n 200 --seed 0   # simulated-caller metrics, synthetic building
```

Without uv: `python -m venv .venv`, activate it, `pip install -e .`, then drop the
`uv run` prefix.

## What runs without keys

| Capability | Without keys or data | Unlocked by |
|---|---|---|
| Synthetic and traced buildings, fusion, search plan, emergency routing, 3D view, scripted scenarios | ✅ | — |
| Reading free-form speech | Keyword rules | `LLM_API_KEY` (Anthropic): Claude Sonnet 5.5 reader |
| Real scanned buildings (5285 W Washington, Bundy Olympic) | ❌ | Matterport Model API keys, Vexcel key, the CoStar hackathon kit, and the gitignored `configs/real_buildings.yaml`; then `uv run python -m ingest.real_building` |
| Address search and generation | ❌ | CoStar hackathon kit in `data/raw/costar/costar-hackathon-kit/` |
| Scan tab (Matterport viewer) | ❌ | A built real building; `MATTERPORT_SDK_KEY` lets the console steer the viewer |
| Aerial imagery in Approach | ❌ | `VEXCEL_API_KEY` |
| Fast local OpenStreetMap lookups | Public Overpass server (slower) | `data/osm/site_pack.sqlite.gz`, shared privately ([docs/osm-site-map.md](docs/osm-site-map.md)) |
| Re-running the benchmark | ❌ (committed results are in `eval/results/`) | All four buildings built |

### Configuration

Copy `.env.example` to `.env` (gitignored). Never commit keys.

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | Anthropic key for the Claude reader. With it set, a new call starts in Claude mode |
| `LLM_MODEL` | Reader model (default `claude-sonnet-5-5`) |
| `PARSER_MODE` | `rules` or `llm` for new calls |
| `LLM_MAX_CALLS` | Optional cap on Claude requests per server process |
| `MATTERPORT_API_KEY`, `MATTERPORT_API_TOKEN` | Matterport Model API, used when building real buildings |
| `MATTERPORT_SDK_KEY` | Browser key for the embedded Matterport viewer |
| `VEXCEL_API_KEY` | Aerial tiles and building outlines; stays on the server |
| `DEMO_BUILDING_ID` | Building the console opens on |

Claude replies are cached in `data/processed/`, so replaying a call costs nothing.

## Reproducing the benchmark

Requires all four benchmark buildings to be built (see above). No network is needed.

```bash
uv run python -m eval.benchmark.simulate --main 400 --robust 150    # 32,200 calls, about 5 min on 20 cores
uv run --with matplotlib --with pandas python -m eval.benchmark.analyze
```

Outputs go to `eval/results/`.

## Limitations

- Every benchmark number comes from simulated callers. No real 911 calls were used.
- The occupancy prior's room-type and time-of-day weights have not been checked against
  real incident locations.
- Room-search times and the 90% detection rate are unverified planning settings, and are
  labelled so in `configs/situations.yaml` and the console.
- In Bundy Olympic, where each suite sits on its floor is assumed, and so are the stairs,
  elevators and restrooms.
- ELC is a decision aid for dispatchers. It ranks where to look; it does not replace a search.

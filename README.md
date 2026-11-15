# 911 Caller Location Predictor

When a 911 caller is somewhere inside a large building, the phone's location fix says
"this address, roughly this height" and the caller says things like *"I'm locked in a
bathroom"* or *"I can see the parking lot"*. This project fuses those signals into a
probability for every room, shows the dispatcher where the caller most likely is, and
gives responders a search order that minimises the expected time to find them.

**Status.** Four buildings run through the same pipeline:

| Building | What it is | Data |
|---|---|---|
| **5285 W Washington Blvd** (`washington-5285`), Los Angeles | A real 2-storey office with a roof deck. **One Matterport scan covers the whole building**, so the floor plan, rooms, doors, windows and stairs are all real | One Matterport scan (38 rooms), CoStar facts, Vexcel aerial imagery |
| **Bundy Olympic** (`bundy-olympic`), 11900 W Olympic Blvd, Los Angeles | A real 8-storey office tower. Only eight tenant suites are scanned, so most of the floor plan is schematic | CoStar suites and tenants, eight Matterport suite scans, Vexcel aerial imagery |
| **Julis Romo Rabinowitz Building and Louis A. Simpson International Building** (`jrr-princeton`), 20 Washington Road, Princeton | A real university building on five levels (B, A, 1, 2, 3). **Traced from the university's published floor plans**: 208 rooms and department areas in their real places, to about a metre | Floor plans and room matrix from the 2016 user guide, OpenStreetMap outline for scale and position, Vexcel aerial imagery; a public Matterport tour of the Economics Forum is shown but not measured. Build it with `uv run python -m ingest.unified plan:jrr-princeton` |
| **100 Princeton Tech Center** (`ptc-100`) | A synthetic 4-floor office | Generated from `configs/demo_buildings.yaml`; needs no keys or data |

The real buildings appear once they have been built (see [Real buildings](#real-buildings)).
Without them, the app starts on the synthetic one and needs nothing but this repo.

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

5285 W Washington Blvd (caller is in a rear office on the second floor):

| Turn | Evidence | What you should see |
|---|---|---|
| 0 | Call connected, address only | Prior over 38 real rooms, drawn on Matterport's own plan image |
| 1 | Phone barometric z-axis fix (4.4 m) | Leans to floor 2 but cannot rule out floor 1: the floors are only 3 m apart |
| 2 | "I'm on the second floor in an office, I locked the door" | Second-floor offices lead (floor 2 at 81%) |
| 3 | "I came up the stairs and turned right to get to this room" | A route clue. The rear offices are on the right of their stair and the front offices on the left of theirs, so the rear offices go from 41% to 76% and the front ones drop to 14% |
| 4 | "I can see houses and a backyard out the window" | The four rear offices with windows hold 93%; the caller's office is stop 1; E[T] 95 s against 101 s and 206 s |

Bundy Olympic (caller is in a private office of Suite 860 on floor 8):

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
- **3D scan** (real buildings): Matterport's viewer shows the scan of the most probable
  scanned room and moves there as the posterior changes. The **3D** button on a stop, or a
  click on a scanned room in the floor plan, pins a room instead. **Inside / Dollhouse /
  Floor plan** switch between Matterport's own view modes. The status line under the
  viewer reports what the viewer itself says it is showing. The dollhouse works for
  5285 W Washington Blvd; the owner of the eight Bundy Olympic suite scans switched it
  off in Matterport, so that button is disabled there.
- **Route clues**: "I turned right after coming up the stairs", "got off the elevator and
  went left", "came in the front door and went straight back".
- **Floor plan** (5285 W Washington Blvd): the heatmap is drawn over Matterport's plan
  image of the floor, and the room outlines are traced from the scan's walls.
- **Approach** (real buildings): Vexcel aerial imagery with the roof outline, entrances,
  staging spots and the recommended approach.

Other commands:

```bash
uv run pytest                                    # 137 tests, no network (some skip without data/)
uv run python -m ingest.unified "165 N Hoover St" # generate a building from an address, live
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

**Route clues** (`core/route_clues.py`): "I turned right after coming up the stairs"
names an anchor the caller passed (stairs up or down, an elevator, the entrance) and the
turn they made after it. For each room the module walks the shortest same-floor route
from where that anchor arrives (the top of each stair flight, with the direction of
travel there) through the door openings to the room, lists the turns of 35° or more, and
scores the room by how likely a caller there is to describe the route that way: the first
turn 80%, a later turn 10%, "straight" 10%. Rooms on a floor the anchor never reaches
score 0, so coming *up* the stairs also rules out the lowest floor. In a whole-building
scan the door positions and stair directions are measured; in the synthetic building and
the Bundy schematic they are derived from the room outlines, so the clue is only as good
as that layout. A left or right with nothing to measure it from ("the door is on my
left") is not used.

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

## Real buildings

```bash
uv run python -m ingest.real_building             # build every entry of configs/real_buildings.yaml
uv run python -m ingest.real_building --refresh   # re-query Matterport and CoStar first
```

For each building this writes `data/processed/<id>.building.json` (a `Building` that
validates against the frozen schema) and `<id>.meta.json` (where each room's data comes
from, the Matterport model and sweep for each scanned room, and the site annotation).
`data/` is gitignored, so each teammate builds once; it needs the keys below and the
CoStar kit in `data/raw/costar/costar-hackathon-kit/`.

There are two kinds of real building, chosen by `kind` in `configs/real_buildings.yaml`.

### One scan of the whole building (`kind: full_scan`)

`ingest/scan_building.py` turns a single Matterport scan into the building. Nothing is
schematic:

| Part of the model | Where it comes from |
|---|---|
| Rooms and their outlines | The scan's rooms and wall vertices. Building-local coordinates are the scan's own, so sweeps and Matterport's plan images line up with the rooms with no transform |
| Doors | An opening two rooms share is a door between them; its position is kept on the edge (`xy`) for route clues |
| Entrances | A doorway that only one ground-floor room has, on the outer wall |
| Stairs | Scan positions partway up between two floors are grouped into flights; each flight links the room at its foot to the room it arrives in, and records where it arrives and the direction of travel there |
| Windows and views | The scan's window openings; a window on the outer wall gets the view annotated for that side |
| Room types | The scan's labels ("Conference Room") and Matterport's room classes ("Office", "Bathroom") |
| Position on the map | Three building corners matched between the scan and Vexcel imagery (`ingest/georegister.py`) |

For 5285 W Washington Blvd that gives 38 rooms on three levels, 2 entrances, 3 stair
flights read from sweeps, and one link that had to be inferred (listed in the metadata).
The two wings of the second floor are not joined on that floor in the scan: each is
reached by its own stair.

**3D.** Matterport's dollhouse of this building is in the embedded viewer. The Model API
also lists downloadable geometry for the scan: a 50k-triangle OBJ mesh with 22 texture
images, and the MatterPak bundle is unlocked. Together with the room outlines, floor
levels, doors and windows already in the model, that is enough to render the building
ourselves (for example with rooms coloured by probability); nothing in the repo does so
yet. Ceiling heights are only given for 3 of the 39 scan rooms.
The map fit is 32.9° from east with 1.4 m residual; the scan's own compass says 29.2°.

**Finding such buildings.** Most Matterport scans in the CoStar kit cover one suite or
one floor. Of 142 scans on 92 small multi-storey commercial buildings, the two that
cover a whole office building with room geometry are this one and 911 Broxton Ave
(scan `rDU2DaDBbyz`: five levels, 97 rooms, but almost no room labels or windows). To
add a building, put a `kind: full_scan` entry in `configs/real_buildings.yaml`.

### A tower with a few scanned suites (Bundy Olympic)

Matterport has no model of this building, only eight separate scans of individual
suites, so neither its viewer nor our floor plan can show the whole building from
scans. Where each fact comes from:

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
- **CoStar lists floor plan images for seven suites**, which might show where suites sit
  on a floor, but its image server refuses those links (HTTP 403) from outside CoStar's
  network. If you can open them, save them under `data/raw/costar/` and the suite layout
  can be redone from them.

## Any building from an address (`ingest/unified/`)

```bash
uv run python -m ingest.unified "165 N Hoover St"   # address or building name
uv run python -m ingest.unified 4405016              # property id
uv run python -m ingest.unified mp:1FTnLNf1sBz       # one whole-building Matterport scan
uv run python -m ingest.unified plan:jrr-princeton   # a building traced from its floor plans
uv run python -m ingest.unified --lookup "olympic"   # list matching properties
```

The pipeline runs at call time and prints each stage with its time. In the web app the
search box in the header does the same thing: pick a result and the building is generated
and opened. `http://localhost:5173/?building=prop-4405016` opens one directly.

| Stage | What it does | Typical time |
|---|---|---|
| Address lookup | SQLite index over the 193 MB property file (`index.py`): byte offsets per property, built once in about 2 s. A lookup is one query; reading a record is one seek | 1 ms |
| Floor stack and roster | Storeys, storey height, and every unit or tenant on a floor (`roster.py`). Unit numbers are decoded by scheme: "304" is floor 3, "7156" floor 7, "01-0317" building 1 floor 3 | 1 ms |
| Unit-type scans | Scans listed for a unit type are checked with Track A's `preflight()` and, when buildable, fetched as a `Model3D` (`stamp.py`). Cached after the first call | 0.2-0.5 s per new scan |
| Archetype layout | Corridor bar, garden walk-up, central core, bays, or a bare floor stack (`layout.py`); scanned interiors stamped into every unit of their type, a generic bed/bath layout elsewhere | 2-100 ms |
| Site context | Where the model sits on the map (`context.py`): the real building outline from Vexcel's property data, the street from OSM, the aerial window. A sidecar file, never part of the model | about 1 s |
| Projection | `core.schema.Building` for the posterior and the search plan, and the 3D viewer payload (`project.py`) | 5 ms |

Everything meets in one schema, `BuildingModel` (`schema.py`): levels, spaces, walls with
openings, links, entrances. Each piece of geometry carries its `fidelity`:

| Fidelity | Meaning |
|---|---|
| `surveyed` | Measured by a Matterport scan of that very room (Track A `Model3D`) |
| `stamped` | Measured in a scan of the same unit type and fitted into this unit's slot |
| `cluster` | A scan with no wall data: room positions and areas only (Track A `ScanModel`) |
| `template` | Generated by an archetype rule; plausible, not measured |
| `roster` | Known to be on this floor; position unknown |

Three kinds of building come out of it:

- **Whole-building scan** (`mp:<model id>`, and 5285 W Washington Blvd in the app): a 1:1
  lift of the scan. Stairs are not in a scan and are not invented; floors are joined by the
  scan-position graph, or by a link marked `inferred`.
- **Generated from a property record**: floors, units and tenants are real; where a unit
  sits along its corridor is assumed (number order, alternating sides).
- **Hybrid**: a generated building whose units of a scanned type carry that scan's rooms.

All 3,339 properties in the kit build offline: 2,960 with a unit or tenant roster and 379
as a bare floor stack, median 13 ms, slowest 1.4 s. The same record always gives a
byte-identical model. The app counts a unit as one room however many rooms its interior
has, because the question is which unit the caller is in.

What is not real, and says so in the model's notes:

- Unit numbers encode the floor for about 45% of apartment properties. The rest are spread
  evenly over the floors in number order (`floor_source: spread`).
- Buildings of a multi-building property are placed on a grid; their real positions are
  not in the data.
- A stamped interior is stretched to its slot, so room proportions follow the slot.
- What a window looks at comes from OpenStreetMap, which has not mapped every car park;
  where it shows nothing in front of the glass, only a street-facing side is labelled.

**Placement on the map.** A property's coordinates are an address point, usually at the
kerb, so a model centred on them lands on the street or the neighbours. The sidecar
therefore looks up the real building outline (Vexcel `property/extract`: the structures
on the parcel, newest vintage of the imagery layer in use, about a third of a second) and:

- centres a generated building on that outline, turns its long side along the outline's
  long side, and fits its extent to the outline's box, so entrances land on the building.
  Which of the two long sides is the front comes from the street bearing (OSM), or, until
  that arrives, from which side the address point is on;
- for a property with several buildings, draws only the address point and the real
  outline there: the grid the buildings are generated on is invented, so none of it is
  put on the map and staging is given as the address point;
- leaves a scan where its own geolocation and yaw put it.

**Window views and the site map.** From each window a fan of rays is cast at the
OpenStreetMap features around the building; the first thing most rays meet within 60 m
sets `Room.window_facing` (`street`, `parking_lot`, `residential`, `adjacent_building`),
which is what view clues are matched against (`ingest/unified/surroundings.py`). The same
features, plus nearby named places, are drawn as a street map in the Approach panel
(Aerial / Map switch). Properties with several buildings are skipped, because their
positions on the ground are invented.

**OpenStreetMap from disk.** The map data around every property in the kit is committed
as `data/osm/site_pack.sqlite.gz` (36 MB) and is unpacked on first use, so lookups take a
few milliseconds and nothing needs downloading. `docs/osm-site-map.md` explains how to
bring this into another branch and how to rebuild the data.

For an address that is not near a kit property, the public Overpass server is
asked instead. That server can take a minute or refuse connections. The pipeline waits
one second for it once the outline has placed the building (four without an outline) and
carries on; when the answer arrives the sidecar is completed, the building is re-projected,
and the web app picks the change up. Requests run one at a time, and after a failure none
is sent for five minutes.

**3D viewer.** The "3D model" panel draws the `BuildingModel` of whichever building is
open, coloured by the posterior, with the next stop marked. Cutaway shows the selected
floor and those below it. Double-click a room to select it. The camera frames whatever
is on show, whatever the panel's shape and however long, flat or tall the building is;
"Reset view" frames it again after the camera has been moved.

**Aerial view.** Imagery and overlay are drawn in one coordinate system, so the outline
cannot drift off the picture. The view is fitted to the building, labels are placed so
they do not cover each other, and a site with many entrances names only the recommended
one on the image.

## A building traced from its floor plans (`plan:<id>`)

For a building with no property record and no Matterport scan the Model API can read, but
with published floor plans. The Julis Romo Rabinowitz / Simpson buildings at Princeton are
the example: the only scan of them is a public tour of one atrium that belongs to another
Matterport account, so its geometry cannot be fetched.

```bash
uv run python -m ingest.unified plan:jrr-princeton   # then http://localhost:5173/?building=jrr-princeton
```

`configs/plan_buildings.yaml` holds the tracing: one rectangle or polygon per room, in the
pixels of each level's plan image, with its name, type, department and the sides that have
windows. `ingest/unified/plan.py` turns it into a `BuildingModel` with `fidelity = traced`.

| What | Where it comes from |
|---|---|
| Rooms, their outlines, levels and names | Traced from the plans (pages 24 to 28 of the 2016 user guide); room data from its room matrix |
| Scale and position on the map | Nine building corners matched to the OpenStreetMap outline: 0.149 m per plan pixel, 0.4 m residual |
| Doors | By rule: a room opens onto each corridor or lobby it shares a wall with; a room drawn inside a department opens into it |
| Stairs and elevators | Traced; the same `shaft` name on each level joins them |
| Windows | The sides marked in the tracing; what each window looks at comes from OpenStreetMap, as for every other building |
| Storey heights | Assumed, 4 m per level |

Limits: department areas are one space each (the guide does not show their offices), the
plans are from 2016, and entrances are approximate. The building's own floor names are kept
(B, A, 1, 2, 3), so "first floor" means Level 1 and "basement" the B level. To add another
building, trace it into `configs/plan_buildings.yaml` the same way: put a pixel grid over
each plan image, read off the room corners, and match a few building corners to the map.

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
| `ingest/mp3d/` | Track A: one Matterport scan → surveyed `Model3D` or coarse `ScanModel`, aerial and street-bearing helpers, glTF export |
| `ingest/unified/` | Address or property id → `BuildingModel` at call time (index, roster, layout, stamping, context, projections) |
| `sim/` | Scripted demo calls, simulated callers and personas |
| `eval/` | Floor accuracy, top-k, 90% coverage, E[T]; `run_eval.py` CLI |
| `api/server.py` | FastAPI app, holds the single in-memory demo call |
| `web/` | React + Vite + Tailwind dispatcher console |
| `configs/` | `priors.yaml`, `clue_vocab.yaml`, `demo_buildings.yaml`, `real_buildings.yaml`, `plan_buildings.yaml` (buildings traced from floor plans): tune these, not code |
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
| GET | `/buildings/{id}/model3d` | 3D viewer payload for any building: generated model, surveyed scan, or extruded room outlines |
| GET | `/lookup?q=...` | Properties matching an address, a building name or a property id |
| POST | `/buildings/generate?q=...` | Generate a building now from an address, a property id or `mp:<model id>`; returns the run report with stage timings |
| GET | `/buildings/{id}/floorplan/{floor}` | Matterport's plan image of a floor (PNG); whole-building scans only |
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
| `LLM_PROVIDER`, `LLM_API_KEY` | `parser/agent.py`, `parser/extract.py` | `anthropic` and its key; only used when the parser is Claude Haiku |
| `PARSER_MODE` | `api/server.py` | Parser a new call starts with: `rules` or `llm` (Claude Haiku; `agent` is an older name for it). Default: `llm` when a key is configured, else `rules` |
| `LLM_MAX_CALLS` | `parser/extract.py` | Optional cap on requests per server process; the rules take over beyond it. Unset means no cap |
| `LLM_THINKING`, `LLM_EFFORT`, `LLM_STRICT_TOOLS` | `parser/agent.py` | Model settings of the Claude Haiku interpreter (`disabled` or `adaptive`; `low`, `medium` or `high`; `1` or `0`). The defaults in `parser/agent.py` were chosen by measurement |
| `LLM_PREWARM` | `api/server.py` | `0` stops the one no-output request that writes a building's prompt cache when a call starts |
| `MATTERPORT_API_KEY`, `MATTERPORT_API_TOKEN` | `ingest/matterport_api.py` | Model API token id and secret, used when building a real building |
| `MATTERPORT_SDK_KEY` | web viewer, via `/config` | Browser key that lets the page steer the embedded viewer; the viewer still loads without it |
| `VEXCEL_API_KEY` | `ingest/vexcel.py` | Aerial tiles; stays on the server |
| `DEMO_BUILDING_ID` | `api/server.py` | Building the demo starts on: `washington-5285`, `bundy-olympic` or `ptc-100` (default: `bundy-olympic` if built, else `ptc-100`) |
| `MATTERPORT_DATA_DIR`, `VEXCEL_DATA_DIR`, `COSTAR_DATA_DIR` | `ingest/` | Raw data and cache folders under `data/raw/` |

What calls the network, and how often:

- **Anthropic**: one small request per new caller utterance while the parser is set to
  Claude Haiku. Replies are cached in `data/processed/llm_clue_cache.json`, so replaying
  the demo costs nothing, and totals are kept in `data/processed/llm_usage.json`.
- **Matterport**: one request per scan, plus one download per floor plan image, when a
  real building is built; cached in `data/raw/matterport/`. The embedded viewer loads
  from Matterport in the browser.
- **Vexcel**: one request per aerial tile (49 for Bundy Olympic, 42 for 5285 W
  Washington Blvd), cached in `data/raw/vexcel/tiles/`.

## What is still stubbed

| Area | Today | Next step |
|---|---|---|
| Suite positions on a floor (Bundy Olympic) | Suite-number order on a schematic plate | CoStar's suite floor plans, or matching scan window walls to the roof outline |
| Facade views | Annotated per side in `configs/real_buildings.yaml` | `classify_surroundings` / `window_ray_intersections` in `ingest/vexcel.py` |
| Unscanned suites (Bundy Olympic) | One `suite` room each | More scans, or a typical-suite interior model |
| Unlabelled rooms in a scan | Typed `room`, which matches room clues only partially | Label them in Matterport, or classify the panoramas |
| Adding a whole-scan building | A config entry plus three matched corners | Automatic corner matching against the aerial |
| Call audio | Scripted turns and typed utterances | Live transcript feed |
| Eval harness | Synthetic building only | Simulated callers on the real building |

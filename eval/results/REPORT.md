# Response-time and survivability benchmark

14 November 2026 · 32,200 simulated 911 calls · 4 buildings · 7 call types.
Interactive version with all charts: `report.html` (also published as an artifact). Static charts: `charts/`.

## Question

When a 911 caller is somewhere inside a building, how much sooner does the crew reach them with the app's search plan, and what is that worth in survival?

## Method

- **Callers.** Each simulated call draws a true room from the app's occupancy model and a caller persona matched to the emergency (for example a panicked bystander, an elderly visitor or an injured caller). It also draws a phone fix: the height reading is present 90% of the time with ±3 m error, and the horizontal fix 50% of the time with σ 30 m. Finally it draws the location clues that persona would give, including wrong floors and hedges.
- **Three search methods**, all starting from the same entrance, with the same walking speed (1.2 m/s, 29 s per floor of stairs, heavy load) and the same room-search times:
  - **Address only:** a nearest-first sweep of the whole building.
  - **Informed dispatcher** (the main comparison): the floor and room type the caller stated are relayed by voice. The crew checks that room type on that floor first, then the rest of that floor, then that room type elsewhere, then everything else.
  - **App:** the app's search plan on the fused posterior. For cardiac arrest it maximises survival.
- **Clock.** Call-to-curb time is drawn from a lognormal with median 7 min and IQR 5–10 (NEMSIS urban; Mell et al. 2017). We add 1 min from the curb to the door, then the in-building time until the true room has been searched.
- **Outcomes.**
  - Cardiac arrest uses Valenzuela 1997 (witnessed) and De Maio 2003 (unwitnessed).
  - Fire, stroke and internal bleeding use linear per-minute rates from `configs/situations.yaml`.
  - Overdose is reported as time only, because there is no published curve.
  - For outcome calculations only, in-building time is capped at 60 min.
- **Statistics.** Every comparison is paired (same caller, entrance and drive time). 95% CIs come from a 2,000-resample bootstrap.

## Results

**Response time** (main benchmark, 11,200 calls):

| | Address only | Informed dispatcher | App |
|---|---|---|---|
| Median minutes from entrance to patient | 39.8 | 10.0 | **5.8** |
| Patients reached within 5 min of entering | 12% | 33% | **45%** |

- The app was faster than the dispatcher on 67% of calls, the same on 13%, and **slower on 20%**.
- The median saving per call was 1.8 to 3.2 minutes depending on call type, with every 95% CI above zero.
- The app ranked the true room first 35% of the time and in its top three 62% of the time.

**Survivability, app vs. informed dispatcher:**

| Call type | Dispatcher | App | Gain per call (95% CI) |
|---|---|---|---|
| Cardiac arrest, witnessed + bystander CPR | 9.72% | 12.38% | +2.66 pts (2.38–2.94) |
| Cardiac arrest, witnessed, no CPR | 3.16% | 4.39% | +1.23 pts (1.06–1.40) |
| Cardiac arrest, unwitnessed | 1.74% | 2.30% | +0.56 pts (0.47–0.65) |
| Fire / smoke, trapped occupant | | | +3.96 survival pts (rate range 2.44–7.84) |
| Stroke | | | +0.71 good-outcome pts (0.27–1.15) |
| Internal bleeding | | | 3.63 fewer mortality pts (2.07–5.19) |
| Overdose | 7.4 min | 4.6 min | time only |

- **Cardiac arrest overall:** about **15 more survivors per 1,000 arrests** than the informed dispatcher (95% CI 14–16), and 44 more than an address-only sweep.
- **Faster ambulances amplify the gain.** With a 4-minute ambulance the bystander-CPR gain rises to about 3.8 points, because the patient is still on the steep part of the curve.
- **Fire times are long for every method.** The model's smoke search rate is a conservative 0.04 m/s. The app still cut the median fire search from 2.2 h to 51 min (−61%).

**Stress tests** (mean minutes saved per call vs. the dispatcher, 4,200 calls each):

| Condition | Mean minutes saved |
|---|---|
| Main benchmark | 9.0 |
| Callers 2× as error-prone | 9.1 |
| Callers 4× as error-prone | 9.6 |
| No phone fix | 5.6 |
| No caller clues | 15.2 |
| **Caller anywhere, equal odds** | **1.6** (median 0; app slower on 40% of calls) |

**Held-out speech** (280 calls written by Claude Haiku 5.5 and read by the app's own interpreter):
- Median search time was **6.7 min**, against 6.4 with clean clues and 9.9 for the dispatcher. The app was faster than the dispatcher on 68% of calls and fell back from the model reader on none.
- Keyword rules did about as well as Haiku on this speech. On the team's 57 hand-written free-form calls, the Haiku interpreter ranked the true room first 94% of the time against 74% for rules.

**By building** (median saving per call): 43 s in the 38-room scanned office, 1.5 min in the synthetic office, 10.2 min in the 8-storey tower and 10.7 min in the 208-room university building.

## Limitations

1. **Everything is simulated.** Callers, their clues and their errors come from the project's persona model. No real calls were used.
2. **The true room is drawn from the app's own occupancy prior.** With equal odds per room the advantage mostly disappears, so that prior should be validated against real incident locations before claiming field benefit.
3. **One crew searches alone**, with no building staff guiding it. That inflates the long tails of every method, the address-only sweep most of all.
4. **Planning settings are unverified.** Room-search times and the 90% detection rate are unverified, and every look is assumed to find a person who is present.
5. **Outcome models are coarse.** Fire, stroke and bleeding use linear rates, and the cardiac curves assume 1.2 min from collapse to call.
6. **The app is slower than the dispatcher on about 1 call in 5.** All averages above include those calls.

## Reproduce

```
uv run python -m eval.benchmark.simulate --main 400 --robust 150    # about 5 min on 20 cores, no network
uv run python -m eval.benchmark.speech --per-cell 10                 # Claude Haiku; replies cached
uv run python -m eval.run_agent_eval --modes rules,llm --json eval/results/freeform_parser.json > eval/results/freeform_parser.txt
uv run --with matplotlib --with pandas python -m eval.benchmark.analyze
uv run python -m eval.benchmark.build_page
```

Raw data: `benchmark_trials.csv` (one row per call), `speech_calls.json` (generated transcripts), `speech_results.json`, `summary.json`.

# v0.2 robustness benchmark

Measured on 15 November 2026, on a shared Linux x86_64 runner with Python 3.12.14,
Gymnasium 1.2.3, Box2D 2.3.10, NumPy 2.5.4 and Pygame 2.6.1. Full per-seed results,
effective configurations and timing samples are recorded in
[`benchmarks/robustness.json`](../benchmarks/robustness.json).

## Matched wind comparison

Both versions use gravity -10, wind power 10, turbulence power 0.5 and a 1000-step
limit. The original controller retains its original 30×0.04 s, 96-sample,
three-iteration search. The new controller uses 20×0.06 s, 64 samples, two iterations,
five acceleration scenarios, compensation and the touchdown stabiliser.

| Controller | Wind seeds | Landed | Score ≥200 | Mean reward |
|---|---|---:|---:|---:|
| Original MPC | 0–49, development/reference | 24/50 (48%) | 14/50 (28%) | 92.7 |
| Disturbance-aware MPC | 0–49, development/regression | 44/50 (88%) | 44/50 (88%) | 222.1 |
| Original MPC | **50–99, new validation set** | **28/50 (56%)** | 22/50 (44%) | **117.2** |
| Disturbance-aware MPC | **50–99, new validation set** | **47/50 (94%)** | 47/50 (94%) | **235.4** |

Seeds 0–49 informed development; the gravity -10 controller was frozen before
testing 50–99. No seed-specific controller rules, wind-state access or simulator
cloning are used. The comparison measures the complete revised controller, not a
single-feature ablation. It does not isolate how much each individual change helps.

The three validation failures, seeds 69, 91 and 93, remain in the report. A 94%
sample success rate is not a guarantee on new seeds or different dynamics.

## Regression and stress cases

| Scenario | Settings | Seeds | Landed |
|---|---|---|---:|
| No wind | Gravity -10 | 0–49 | **50/50** |
| Strong wind | Gravity -10, wind 15, turbulence 1.5 | 100–109 | **6/10** |
| High gravity with wind | Gravity -11.5, wind 10, turbulence 0.5 | 100–109 | **6/10** |
| Low gravity with wind, regression | Gravity -6, wind 10, turbulence 0.5 | 100–109 | **10/10** |
| Low gravity with wind, new validation | Gravity -6, wind 10, turbulence 0.5 | 110–119 | **9/10** |

The no-wind regression retained 100% landings; 49/50 episodes scored at least 200,
with a mean reward of 249.7. In low-gravity validation, seed 113 rested outside the
pad and is counted as a failure even though it scored above 200.

An initial low-gravity stress check exposed an engine-mapping defect: the minimum
50% thrust exceeds hover demand, causing repeated upward-biased descent commands.
The revised low-gravity switching rule was checked on the original failing range
and then on 110–119. This branch is inactive at gravity -10 and -11.5, so it did not
change the frozen nominal-gravity wind validation controller.

Stronger gusts and high gravity are still difficult. The estimator lags changing
forces, the fixed-inertia model approximates articulated legs, and the five constant
acceleration hypotheses omit simultaneous extremes and future gust reversals.
Touchdown stabilisation can arrest some tipping, but cannot recover every contact.

## Planning time

A separate sequential fixed-state profile avoids mixing cheap feedback steps with
expensive planning calls. Each controller ran three warmups and 50 measured `plan`
calls at `[x,y,vx,vy,angle,omega] = [1,5,0.4,-1,0.1,0.2]`, using seed 50. The adaptive
controller used a fixed acceleration-bias estimate `[1,0,0.3]` for this timing-only
test. No landing results are inferred from that artificial fixed-state profile.

| Planning metric | Original | v0.2 |
|---|---:|---:|
| Mean | 25.20 ms | **14.83 ms** |
| 95th percentile | 27.87 ms | **16.91 ms** |
| Maximum | 34.76 ms | **17.07 ms** |
| Calls exceeding 20 ms | 50/50 | **0/50** |

The new mean is approximately 41% lower. Fewer prediction intervals and search
iterations, plus a coarse prediction step, offset the extra scenario evaluations.
Actual physics remains at 50 Hz, while planning is requested every three steps.

This is not a hard real-time guarantee. During overlapping episode evaluations,
scheduler contention produced controller calls above 20 ms. Full summaries retain
their p95, maximum and deadline-exceedance fraction; hardware, rendering and system
load can change interactive playback speed.

## Reproduce

```bash
./scripts/setup.sh --dev

# Matched validation and a separate sequential planning profile
./scripts/benchmark.sh --scenarios wind nominal-wind --seed 50 --episodes 50 --profile

# No-wind regression
./scripts/benchmark.sh --scenarios no-wind --seed 0 --episodes 50

# Stress and gravity regressions
./scripts/benchmark.sh --scenarios strong-wind high-gravity low-gravity --seed 100 --episodes 10
./scripts/benchmark.sh --scenarios low-gravity --seed 110 --episodes 10
```

The matrix saves CSV telemetry, predictions, configuration, runtime and per-episode
summaries, plus a checkpointed `benchmark.json`. Add `--plot` to save figures. Use
`--output` to choose a new empty destination; existing results are not overwritten.
Cross-platform bitwise equivalence is not promised.

The [original v0.1 benchmark](benchmark.md) remains available for historical context.

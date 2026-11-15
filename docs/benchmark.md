# v0.1 reference benchmark

Measured on **15 November 2026**, with the checked-in defaults, a shared Linux
x86_64 runner, Python 3.12.14, Gymnasium 1.2.3, Box2D 2.3.10, NumPy 2.5.4 and
Pygame 2.6.1. The full per-seed results and effective configurations are in
[`benchmarks/reference.json`](../benchmarks/reference.json).

| Scenario | Seeds | Landed | Score ≥ 200 | Mean reward | Mean control call |
|---|---|---:|---:|---:|---:|
| CEM MPC, no wind | 0–49 | 50/50 (100%) | 44/50 (88%) | 240.0 | 8.31 ms |
| Feedback only, no wind | 0–49 | 42/50 (84%) | 36/50 (72%) | 196.4 | 0.037 ms |
| CEM MPC, wind preset | 0–9 | 5/10 (50%) | 1/10 (10%) | 97.1 | 8.42 ms |

The nominal MPC check first covered seeds 0–19, then extended to 20–49 without
changing the control algorithm. The matched feedback comparison uses the same
environment settings and seed range. No model training or seed-specific tuning
was performed. These are empirical results, not a guarantee for untested seeds.

## Interpretation

The trajectory search improves both landing rate and average score over its own
feedback-only proposal in this sample. Landing and score are separate: a vehicle
can come to rest safely while spending enough time or engine power to score below
200. The controller optimises its own flight objective, not Gymnasium's exact reward.

The wind preset applies `wind_power=10.0` and `turbulence_power=0.5`. Its much lower
success rate exposes an important limit: the predictor does not estimate external
force or torque. Replanning and feedback help, but this implementation is not robust
MPC. Wind performance is reported without retuning or hiding failed episodes.

## Timing

The table averages all controller calls, including cheap feedback-only updates and
engine-off contact steps. It is **not** the mean optimisation time. In a separate
seed-0 diagnostic run:

| Metric | Measured value |
|---|---:|
| All-step mean controller latency | 8.44 ms |
| Mean planning-call latency | 27.09 ms |
| 95th-percentile controller latency | 29.25 ms |
| Maximum controller latency | 46.27 ms |
| Controller calls exceeding 20 ms | 31.0% |

The model advances at 50 physics steps/s, with a nominal planning interval of three
steps. A 27 ms planning call exceeds a single 20 ms physics tick even though it is
below the 60 ms planning interval. This synchronous implementation can therefore
slow down interactive playback. Timing was measured on shared infrastructure with
some overlapping validation work; repeat on your machine before drawing real-time
conclusions. There is no asynchronous solver or hard deadline fallback.

## Reproduce

These are historical v0.1 results. Current `main` defaults to the revised controller.
To reproduce this original table, create an isolated checkout of the original commit:

```bash
git worktree add --detach ../Lunar-Lander-v0.1 fcacfe05f2b158a2c94a7de9241da7d926c17a75
cd ../Lunar-Lander-v0.1
./scripts/setup.sh --dev
./scripts/evaluate.sh --episodes 50 --seed 0 --output runs/mpc-reference
./scripts/evaluate.sh --controller feedback --episodes 50 --seed 0 --output runs/feedback-reference
./scripts/evaluate.sh --config configs/wind.toml --episodes 10 --seed 0 --output runs/wind-reference
```

Use a matching software stack for close reproduction; platform and dependency
changes can alter stochastic trajectories. `runtime.json` records the stack used
for newly generated episodes. Each evaluation writes plots, CSV telemetry, predicted
trajectories, configuration and per-episode/aggregate JSON. Add `--no-plot` for faster
benchmark output when you only need numerical results.

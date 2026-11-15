# Lunar Lander · Model Predictive Control

[![Tests](https://github.com/CagriCatik/Lunar-Lander/actions/workflows/ci.yml/badge.svg)](https://github.com/CagriCatik/Lunar-Lander/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11–3.13-3776AB?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A5)
![Control](https://img.shields.io/badge/Control-CEM%20MPC-9B59B6)

**Land a rocket by predicting its flight and optimising its engine commands. No training required.**

A sampling-based MPC controller for the continuous-action version of Gymnasium's
`LunarLander-v3`. Watch the actual and predicted trajectories together, inspect engine
commands, and compare the predictive controller against its feedback-only baseline.

![Actual flight and MPC predictions](docs/assets/flight.png)

## Quick start

macOS or Linux, with **Python 3.11, 3.12 or 3.13**:

```bash
git clone https://github.com/CagriCatik/Lunar-Lander.git
cd Lunar-Lander
./scripts/setup.sh --dev
./scripts/run.sh --render
```

The setup script creates `.venv`, installs SWIG before Box2D, installs the project,
and checks that the environment can reset and step. It does not require activation,
global Python package installation, or changes to your shell profile.

Choose a specific interpreter if your default Python is outside the supported range:

```bash
PYTHON_BIN=python3.13 ./scripts/setup.sh --dev
```

The window shows a **cyan flight trail**, **yellow nominal prediction**, reward and
control timing. Close the window or press **Escape** to stop and save the episode.

![Live flight with predictive trajectory overlay](docs/assets/live.png)

## Run and compare

```bash
# Interactive MPC, different initial push and terrain
./scripts/run.sh --render --seed 7

# Headless run with saved plot and telemetry
./scripts/run.sh --seed 0

# Matched-seed comparison; evaluate is always headless
./scripts/evaluate.sh --episodes 50 --seed 0 --output runs/mpc-50
./scripts/evaluate.sh --controller feedback --episodes 50 --seed 0 --output runs/feedback-50

# Disturbance stress scenario
./scripts/run.sh --render --config configs/wind.toml --seed 3

# Development checks
./scripts/doctor.sh
./scripts/test.sh -q
```

Every run uses a new timestamped output directory unless `--output` is supplied.
Existing results are never overwritten. Use `--no-plot` to skip PNG generation.

You can also activate the environment and use the installed command:

```bash
source .venv/bin/activate
lunar-mpc run --render --config configs/default.toml
lunar-mpc evaluate --episodes 20 --controller mpc
```

## Configure locally

Copy the default configuration and pass it to either entry point:

```bash
cp configs/default.toml configs/local.toml
./scripts/run.sh --render --config configs/local.toml
```

| Setting | Default | Effect |
|---|---:|---|
| `environment.gravity` | `-10.0` | Vertical acceleration; the nominal model uses the same value |
| `environment.wind` | `false` | Enable simulator wind and rotational turbulence |
| `environment.wind_power` | `15.0` | Simulator wind-force magnitude |
| `environment.turbulence_power` | `1.5` | Simulator wind-torque magnitude |
| `environment.max_steps` | `1000` | Episode time limit, at 50 physics steps per second |
| `mpc.horizon` | `30` | Number of prediction intervals |
| `mpc.dt` | `0.04` | Seconds per prediction interval; multiple of `0.02` |
| `mpc.knots` | `6` | Piecewise-constant control-bias segments |
| `mpc.samples` | `96` | Candidate sequences per CEM iteration |
| `mpc.elites` | `12` | Best candidates used to update the search distribution |
| `mpc.iterations` | `3` | CEM refinements per optimisation |
| `mpc.replan_interval` | `3` | Simulator steps between optimisations |
| `mpc.noise` | `0.45` | Initial search standard deviation |
| `mpc.min_std` | `0.08` | Minimum search standard deviation |

The defaults look **1.2 seconds ahead** and replan every **0.06 simulated seconds**.
Feedback is recomputed at every physics step between replans. Larger sample counts,
horizons and iteration counts increase computation. Invalid and misspelled settings
fail explicitly. This is a CPU implementation; no CUDA or GPU setup is needed.

## What the controller does

1. Convert Gymnasium's observation into six physical flight states.
2. Build a stabilising feedback proposal from position, velocity, angle and angular velocity.
3. Sample piecewise control-bias sequences and simulate each proposal with the nominal model.
4. Rank trajectories by tracking, fuel-proxy and soft safety costs; refine the sampling distribution.
5. Apply the first optimised bias to the current feedback command and repeat.

The zero-bias feedback trajectory remains a candidate at every optimisation. The
selected plan therefore has no higher **predicted cost** than that proposal, although
model error means this does not guarantee a better real trajectory. On leg contact,
a separate touchdown mode switches the engines off and lets Box2D settle.

This is **feedback-shooting MPC**, not a trained RL agent or a pure heuristic with
an MPC label. The feedback baseline uses the same guidance without trajectory search.
See [controller and model details](docs/controller.md) for equations, costs and boundaries.

## Results and diagnostics

Each episode produces:

| File | Contents |
|---|---|
| `flight.png` | Flight path, sampled prediction snapshots, velocity, attitude and engine power |
| `telemetry.csv` | Pre-action physical state, contacts, actions, effective power and control latency |
| `predictions.json` | Nominal predicted state trajectories, indexed by simulator step |
| `summary.json` | Outcome, reward, first-contact approach motion, final state, fuel proxy and timing |
| `config.json` | Effective settings, including defaults |
| `runtime.json` | Python, platform, environment ID and simulator/package versions |
| `evaluation.json` | Aggregate results and per-episode summaries |

`landed` means the simulation terminated at rest on the pad with leg contact and
without a body crash. `score_at_least_200` is tracked separately. Contact velocity
is sampled **before the first contact step**, so it is an approach-speed estimate,
not an exact time-of-impact measurement. Fuel is infinite in this environment;
the reported fuel proxy integrates effective main and lateral power, not kilograms.

See the [reference benchmark](docs/benchmark.md) for measured results and limitations.

## Project structure

| Module | Responsibility |
|---|---|
| `config.py` | TOML loading and validation |
| `model.py` | State conversion, engine thresholds, vectorised flight prediction |
| `guidance.py` | Landing targets and stabilising feedback proposal |
| `controller.py` | CEM search and touchdown-mode switching |
| `environment.py` | Gymnasium construction and outcome classification |
| `runner.py` | Episodes, metrics and artifact writing |
| `visualization.py` / `plotting.py` | Live overlay / saved headless plots |
| `cli.py` | Local run and evaluation commands |

The model, guidance and optimiser do not import Gymnasium or access simulator
internals. Only the adapter, outcome classifier and live overlay use Box2D-specific
details. Tests cover engine boundaries, state scales, vectorisation, reproducibility,
configuration failures, nominal-model mismatch, timeout reporting and an actual landing.
CI runs on Linux and macOS with Python 3.11–3.13.

## Limits and troubleshooting

- The flight model approximates the articulated vehicle as a rigid body. It omits
  random exhaust dispersion, wind, terrain geometry and ground contacts.
- Attitude, ground and viewport penalties are **soft costs**. Only action bounds and
  engine activation rules are enforced directly. This is not a safety-certified or
  robust MPC implementation.
- Real-time execution is hardware-dependent. Planning calls can exceed one 20 ms
  physics tick; summaries report planning latency and the fraction of control calls
  above 20 ms. The simulator may run slower than wall-clock real time.
- A new seed changes terrain and the initial push. Equal seeds reproduce runs on a
  matching software/platform stack; cross-platform bitwise identity is not promised.
- If Box2D must compile, install a C++ compiler: `xcode-select --install` on macOS,
  or `sudo apt install build-essential python3-dev python3-venv` on Ubuntu/Debian.
- If `python3 -m venv` fails on Ubuntu/Debian, install `python3-venv` first.
- On a server without a display, omit `--render`. Plots work headlessly.
- Native Windows shell setup is not provided. Use Linux/WSL, or install manually
  in a Python virtual environment with SWIG before `pip install -e '.[dev]'`.
- Scripts accept `VENV_DIR` for a custom virtual environment; supply the same value
  to setup and subsequent commands.

## References

- [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Pinned Gymnasium 1.2.3 implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/box2d/lunar_lander.py)

Gymnasium and Box2D remain their respective authors' projects. This repository adds
the predictive controller, local workflow and evaluation tooling.

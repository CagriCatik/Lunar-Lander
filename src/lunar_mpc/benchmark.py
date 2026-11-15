"""Repeatable robustness matrix; benchmark failures remain measured outcomes."""

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import load_config
from .controller import MPCController
from .model import FlightModel
from .robust import RobustMPCController
from .runner import aggregate, run_episode, save_episode


def benchmark_cases():
    nominal = load_config("configs/default.toml")
    wind = load_config("configs/wind.toml")
    reference = load_config("configs/v1-wind.toml")
    # The reference has its original solver budget, but identical physical settings.
    reference = replace(reference, environment=wind.environment)
    return {
        "no-wind": ("mpc", nominal),
        "wind": ("mpc", wind),
        "nominal-wind": ("nominal", reference),
        "strong-wind": ("mpc", load_config("configs/strong-wind.toml")),
        "low-gravity": ("mpc", replace(wind, environment=replace(wind.environment, gravity=-6))),
        "high-gravity": (
            "mpc",
            replace(wind, environment=replace(wind.environment, gravity=-11.5)),
        ),
    }


def profile_planning(cases, seed=50, repeats=50):
    """Sequential, fixed-state timing; separate from closed-loop landing claims."""
    state = np.array([1, 5, 0.4, -1, 0.1, 0.2], dtype=float)
    bias = np.array([1, 0, 0.3], dtype=float)
    result = {"state": state.tolist(), "estimated_bias": bias.tolist(), "repeats": repeats}
    for name, cls in (("nominal-wind", MPCController), ("wind", RobustMPCController)):
        _, config = cases[name]
        model = FlightModel(gravity=config.environment.gravity)
        if cls is RobustMPCController:
            controller = cls(config.mpc, model, config.robust, seed)
            controller.observer.estimate[:] = bias
        else:
            controller = cls(config.mpc, model, seed)
        for _ in range(3):
            controller.plan(state)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            controller.plan(state)
            durations.append((time.perf_counter() - start) * 1000)
        result[name] = {
            "mean_ms": float(np.mean(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "max_ms": float(max(durations)),
            "over_20ms_fraction": sum(t > 20 for t in durations) / repeats,
        }
    return result


def main(argv=None):
    cases = benchmark_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", choices=list(cases), nargs="+", default=list(cases))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot", action="store_true", help="Save plots as well as numerical results"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Also time sequential fixed-state planning"
    )
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed must be nonnegative")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or Path("runs") / f"{stamp}-robustness"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output path must be an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed_start": args.seed, "episodes_per_scenario": args.episodes, "scenarios": {}}
    for name in dict.fromkeys(args.scenarios):
        controller, config = cases[name]
        summaries = []
        for seed in range(args.seed, args.seed + args.episodes):
            summary, rows, predictions = run_episode(config, seed, controller)
            save_episode(
                output / name / f"seed-{seed}", summary, rows, predictions, config, args.plot
            )
            summaries.append(summary)
            print(
                f"{name} seed={seed} {summary['outcome']} reward={summary['reward']:.1f}",
                flush=True,
            )
        report["scenarios"][name] = {"config": asdict(config), **aggregate(summaries)}
        # Checkpoint completed scenarios so an interrupted long benchmark retains its results.
        (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.profile:
        report["planning_profile"] = profile_planning(cases, args.seed)
        (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()

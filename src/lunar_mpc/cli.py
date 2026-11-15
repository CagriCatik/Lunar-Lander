"""Command line entry point for interactive runs and headless benchmarks."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .runner import aggregate, run_episode, save_episode


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "evaluate"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--controller", choices=["mpc", "feedback"], default="mpc")
    parser.add_argument(
        "--seed", type=nonnegative_int, default=0, help="First environment and controller seed"
    )
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--render", action="store_true", help="Open the Gymnasium window")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate" and args.render:
        parser.error("evaluate is headless; use run --render for an interactive episode")
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or Path("runs") / f"{stamp}-{args.controller}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output path must be an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in range(args.seed, args.seed + args.episodes):
        summary, rows, predictions = run_episode(config, seed, args.controller, args.render)
        save_episode(
            output / f"seed-{seed}", summary, rows, predictions, config, plot=not args.no_plot
        )
        summaries.append(summary)
        print(
            f"seed={seed} outcome={summary['outcome']} reward={summary['reward']:.1f} "
            f"steps={summary['steps']} control={summary['mean_control_ms']:.2f} ms",
            flush=True,
        )
    report = aggregate(summaries)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Landed {report['landing_rate']:.0%}; mean reward {report['mean_reward']:.1f}")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()

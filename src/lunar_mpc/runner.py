"""Episode orchestration and reproducible telemetry, isolated from optimisation."""

import csv
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .controller import FeedbackController, MPCController
from .environment import classify_outcome, make_environment
from .model import FlightModel, decode_observation, engine_power
from .robust import RobustMPCController


def run_episode(config, seed, controller_name="mpc", render=False):
    if controller_name not in {"mpc", "nominal", "feedback"}:
        raise ValueError("controller_name must be 'mpc', 'nominal' or 'feedback'")
    model = FlightModel(gravity=config.environment.gravity)
    if controller_name == "mpc":
        controller = RobustMPCController(config.mpc, model, config.robust, seed)
    elif controller_name == "nominal":
        controller = MPCController(config.mpc, model, seed)
    else:
        controller = FeedbackController(model)
    env = make_environment(config.environment, render)
    rows, predictions, latencies = [], [], []
    reward_total = 0.0
    fuel = 0.0
    touchdown = None
    visualizer = None
    user_closed = False
    planning_latencies = []
    touchdown_control_steps = 0
    try:
        observation, _ = env.reset(seed=seed)
        if render:
            from .visualization import FlightOverlay

            visualizer = FlightOverlay(env)
        for step in range(config.environment.max_steps):
            plans_before = getattr(controller, "plans", 0)
            before = time.perf_counter()
            action = controller.act(observation)
            latency = (time.perf_counter() - before) * 1000
            latencies.append(latency)
            if any(observation[6:]) and abs(action[1]) > 0.5:
                touchdown_control_steps += 1
            replanned = getattr(controller, "plans", 0) > plans_before
            if replanned:
                planning_latencies.append(latency)
            state = decode_observation(observation)
            main, side = engine_power(action)
            fuel += float(main + 0.1 * abs(side)) * 0.02
            rows.append(
                [
                    step,
                    step * 0.02,
                    *state,
                    *observation[6:],
                    *action,
                    float(main),
                    float(side),
                    latency,
                    *getattr(controller, "disturbance", np.zeros(3)),
                ]
            )
            if replanned:
                predictions.append({"step": step, "states": controller.prediction.tolist()})
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += reward
            if touchdown is None and any(observation[6:]):
                # Capture incoming motion, before the contact impulse reduces velocity.
                touchdown = {
                    "pre_contact_vx": float(state[2]),
                    "pre_contact_vy": float(state[3]),
                    "pre_contact_angle": float(state[4]),
                    "time_s": (step + 1) * 0.02,
                }
            if visualizer is not None:
                user_closed = not visualizer.draw(
                    state,
                    controller.prediction,
                    reward_total,
                    latency,
                    action,
                    getattr(controller, "disturbance", None),
                )
            if terminated or truncated or user_closed:
                break
        outcome = "user_closed" if user_closed else classify_outcome(env, terminated, truncated)
        summary = {
            "seed": seed,
            "controller": controller_name,
            "outcome": outcome,
            "reward": float(reward_total),
            "score_at_least_200": bool(reward_total >= 200),
            "steps": step + 1,
            "fuel_proxy": fuel,
            "touchdown": touchdown,
            "mean_control_ms": float(np.mean(latencies)),
            "p95_control_ms": float(np.percentile(latencies, 95)),
            "max_control_ms": float(max(latencies)),
            "plans": len(planning_latencies),
            "mean_plan_ms": float(np.mean(planning_latencies)) if planning_latencies else None,
            "control_over_20ms_fraction": sum(t > 20 for t in latencies) / len(latencies),
            "final_state": decode_observation(observation).tolist(),
            "final_contacts": [bool(v) for v in observation[6:]],
            "touchdown_control_steps": touchdown_control_steps,
        }
        return summary, rows, predictions
    finally:
        env.close()


COLUMNS = [
    "step",
    "time_s",
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "omega",
    "left_contact",
    "right_contact",
    "main_action",
    "side_action",
    "main_power",
    "side_power",
    "control_ms",
    "estimated_ax",
    "estimated_ay",
    "estimated_angular_accel",
]


def save_episode(directory, summary, rows, predictions, config, plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment_id": "LunarLander-v3",
        "packages": {
            name: version(name)
            for name in ("lunar-lander-mpc", "gymnasium", "numpy", "Box2D", "pygame")
        },
    }
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (directory / "predictions.json").write_text(json.dumps(predictions) + "\n")
    if plot:
        from .plotting import plot_episode

        plot_episode(directory, summary, rows, predictions)


def aggregate(summaries):
    plans = sum(s.get("plans", 0) for s in summaries)
    steps = sum(s["steps"] for s in summaries)
    return {
        "episodes": len(summaries),
        "landing_rate": sum(s["outcome"] == "landed" for s in summaries) / len(summaries),
        "score_200_rate": sum(s["score_at_least_200"] for s in summaries) / len(summaries),
        "mean_reward": float(np.mean([s["reward"] for s in summaries])),
        "mean_control_ms": float(np.mean([s["mean_control_ms"] for s in summaries])),
        "mean_plan_ms": (
            sum((s.get("mean_plan_ms") or 0) * s.get("plans", 0) for s in summaries) / plans
            if plans
            else None
        ),
        "max_control_ms": max(s["max_control_ms"] for s in summaries),
        "control_over_20ms_fraction": sum(
            s.get("control_over_20ms_fraction", 0) * s["steps"] for s in summaries
        )
        / steps,
        "episodes_detail": summaries,
    }

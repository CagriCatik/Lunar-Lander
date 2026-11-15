"""Gymnasium boundary and simulator-specific outcome classification."""

import gymnasium as gym

from .config import EnvironmentConfig


def make_environment(config: EnvironmentConfig, render=False):
    return gym.make(
        "LunarLander-v3",
        continuous=True,
        render_mode="human" if render else None,
        gravity=config.gravity,
        enable_wind=config.wind,
        wind_power=config.wind_power,
        turbulence_power=config.turbulence_power,
        max_episode_steps=config.max_steps,
    )


def classify_outcome(env, terminated, truncated):
    """Reward >= 200 is a score threshold, not a reliable touchdown detector."""
    base = env.unwrapped
    if base.game_over:
        return "crashed"
    if abs(base.lander.position.x - 10) >= 10:
        return "out_of_bounds"
    if terminated and not base.lander.awake:
        leg_contact = any(leg.ground_contact for leg in base.legs)
        in_pad = base.helipad_x1 <= base.lander.position.x <= base.helipad_x2
        return "landed" if leg_contact and in_pad else "rested_elsewhere"
    return "timeout" if truncated else "unknown"

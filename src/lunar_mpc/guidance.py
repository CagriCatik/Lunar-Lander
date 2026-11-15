"""Feedback landing policy used as a baseline and MPC rollout proposal."""

import numpy as np

from .model import FlightModel


def targets(state):
    x, y, vx = state[..., 0], state[..., 1], state[..., 2]
    target_angle = np.clip(0.10 * x + 0.20 * vx, -0.45, 0.45)
    # Remain above terrain while translating, then slow down near the pad.
    height = 0.55 * np.abs(x)
    target_vy = -np.clip(0.75 * (y - height) + 0.12, -1.5, 2.5)
    return target_angle, target_vy


def feedback(state, model: FlightModel):
    state = np.asarray(state)
    target_angle, target_vy = targets(state)
    alpha = 8 * (target_angle - state[..., 4]) - 4 * state[..., 5]
    side = np.clip(-alpha / 4.9, -1, 1)
    # Pulse the lateral jet when requested acceleration exceeds half its authority.
    side = np.where(np.abs(side) > 0.25, np.sign(side) * np.maximum(0.501, np.abs(side)), 0)
    ay = 2.5 * (target_vy - state[..., 3])
    thrust = (
        (ay - model.gravity)
        * model.mass
        / ((4 / 30) * 13 * 50 * np.maximum(0.3, np.cos(state[..., 4])))
    )
    main = np.where(thrust > 0.25, np.clip(2 * thrust - 1, 0.001, 1), -1)
    return np.stack([main, side], axis=-1)

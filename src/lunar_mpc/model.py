"""Vectorised nominal free-flight model, in physical Box2D units.

Engine constants and observation scales follow Gymnasium 1.2.3's lunar_lander.py.
Leg articulation, contact and random exhaust dispersion are intentionally omitted.
"""

from dataclasses import dataclass

import numpy as np

OBS_SCALE = np.array([10.0, 20.0 / 3.0, 5.0, 7.5, 1.0, 2.5])


def decode_observation(observation):
    obs = np.asarray(observation, dtype=float)
    if obs.shape != (8,) or not np.isfinite(obs).all():
        raise ValueError("Expected eight finite observation values")
    return obs[:6] * OBS_SCALE


def engine_power(actions):
    """Exact continuous-action activation thresholds, including their endpoints."""
    actions = np.clip(np.asarray(actions, dtype=float), -1, 1)
    main = np.where(actions[..., 0] > 0, (actions[..., 0] + 1) / 2, 0)
    side = np.where(np.abs(actions[..., 1]) > 0.5, actions[..., 1], 0)
    return main, side


@dataclass(frozen=True)
class FlightModel:
    gravity: float = -10.0
    mass: float = 4.958888843655586
    inertia: float = 0.90

    def step(self, state, action, dt=0.02, disturbance=None):
        """Semi-implicit Euler with averaged impulse forces at 50 Hz."""
        state = np.asarray(state, dtype=float)
        result = state.copy()
        theta = state[..., 4]
        sine, cosine = np.sin(theta), np.cos(theta)
        main, side = engine_power(action)
        main_force = (4 / 30) * 13 * 50 * main
        side_force = (12 / 30) * 0.6 * 50 * side
        result[..., 2] += dt * (-sine * main_force + cosine * side_force) / self.mass
        result[..., 3] += dt * (
            (cosine * main_force + sine * side_force) / self.mass + self.gravity
        )
        # Gymnasium uses 17/30 in x but 14/30 in y for the side impulse point.
        arm = (14 / 30) * cosine**2 + (17 / 30) * sine**2 - 0.101307
        result[..., 5] -= dt * arm * side_force / self.inertia
        if disturbance is not None:
            result[..., 2:4] += dt * np.asarray(disturbance)[..., :2]
            result[..., 5] += dt * np.asarray(disturbance)[..., 2]
        result[..., :2] += dt * result[..., 2:4]
        result[..., 4] += dt * result[..., 5]
        return result

    def coarse_step(self, state, action, dt, disturbance):
        """One force evaluation per interval, retaining 50 Hz position integration.

        Accelerations are frozen over the interval. Correcting positions matches
        repeated 0.02 s semi-implicit steps under constant acceleration exactly.
        """
        state = np.asarray(state, dtype=float)
        result = self.step(state, action, dt, disturbance)
        correction = 0.5 * (dt - 0.02)
        result[..., :2] -= correction * (result[..., 2:4] - state[..., 2:4])
        result[..., 4] -= correction * (result[..., 5] - state[..., 5])
        return result

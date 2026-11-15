"""Disturbance-compensated CEM with a small set of force/torque scenarios.

All disturbance information comes from past measured transitions. No wind state,
simulator body, future random number or terrain is available to this module.
"""

import numpy as np

from .config import RobustConfig
from .controller import MPCController
from .model import decode_observation
from .observer import DisturbanceObserver


class RobustMPCController(MPCController):
    def __init__(self, config, model, robust_config=None, seed=0):
        self.robust_config = robust_config or RobustConfig()
        self.observer = DisturbanceObserver(model, self.robust_config)
        super().__init__(config, model, seed)

    @property
    def disturbance(self):
        return self.observer.estimate.copy()

    def reset(self):
        super().reset()
        self.observer.reset()
        self._scenario_disturbance = np.zeros(3)

    def _targets(self, states):
        x, y, vx = states[..., 0], states[..., 1], states[..., 2]
        dx, dy, _ = self.observer.estimate
        gravity = max(3.0, -self.model.gravity - dy)
        angle = np.clip(0.10 * x + 0.20 * vx + np.arctan2(dx, gravity), -0.55, 0.55)
        height = 0.55 * np.abs(x)
        vy = -np.clip(0.75 * (y - height) + 0.12, -1.5, 2.5)
        return angle, vy

    def _feedback(self, states):
        angle_ref, vy_ref = self._targets(states)
        _, dy, angular_accel = self.observer.estimate
        alpha = 8 * (angle_ref - states[..., 4]) - 4 * states[..., 5]
        side = np.clip((angular_accel - alpha) / 4.9, -1, 1)
        side = np.where(np.abs(side) > 0.25, np.sign(side) * np.maximum(0.501, np.abs(side)), 0)
        ay = 2.5 * (vy_ref - states[..., 3])
        thrust = (
            (ay - self.model.gravity - dy)
            * self.model.mass
            / ((4 / 30) * 13 * 50 * np.maximum(0.3, np.cos(states[..., 4])))
        )
        threshold = 0.25
        if -self.model.gravity * self.model.mass < 0.5 * (4 / 30) * 13 * 50:
            # In low gravity, minimum engine power exceeds hover demand. Switch
            # around zero requested acceleration, rather than biasing descent upward.
            threshold = np.clip(
                (-self.model.gravity - dy)
                * self.model.mass
                / ((4 / 30) * 13 * 50 * np.maximum(0.3, np.cos(states[..., 4]))),
                0,
                0.5,
            )
        main = np.where(thrust > threshold, np.clip(2 * thrust - 1, 0.001, 1), -1)
        return np.stack([main, side], axis=-1)

    def _advance(self, states, action):
        return self.model.coarse_step(states, action, self.config.dt, self._scenario_disturbance)

    def _rollout(self, state, offsets, record=False):
        c = self.robust_config
        variations = np.array(
            [
                [0, 0, 0],
                [c.wind_margin, 0, 0],
                [-c.wind_margin, 0, 0],
                [0, 0, c.torque_margin],
                [0, 0, -c.torque_margin],
            ]
        )
        count = len(offsets)
        scenarios = self.observer.estimate + variations
        self._scenario_disturbance = np.repeat(scenarios, count, axis=0)
        costs, path = super()._rollout(state, np.tile(offsets, (len(scenarios), 1, 1)), record)
        costs = costs.reshape(len(scenarios), count)
        mean = costs.mean(axis=0)
        risk_cost = mean + c.risk_weight * (costs.max(axis=0) - mean)
        # Only the central estimate is drawn, not an uncertainty guarantee.
        return risk_cost, path[:, :count] if record else None

    def act(self, observation):
        state = decode_observation(observation)
        if observation[6] or observation[7]:
            # Contact impulses are not wind; discard history before a possible bounce.
            self.observer.reset()
            self.tick = 0
            self.bias[:] = 0
            self.prediction = np.empty((0, 6))
            return touchdown_action(state)
        self.observer.update(state)
        action = super().act(observation)
        self.observer.remember(state, action)
        return action


def touchdown_action(state):
    """Level a tilted first-leg contact, then release controls so Box2D can sleep."""
    angle, omega = state[4], state[5]
    if abs(angle) < 0.06 and abs(omega) < 0.25:
        return np.array([-1, 0], dtype=np.float32)
    side = np.clip((16 * angle + 6 * omega) / 4.9, -1, 1)
    side = np.sign(side) * max(0.501, abs(side)) if abs(side) > 0.2 else 0
    return np.array([-1, side], dtype=np.float32)

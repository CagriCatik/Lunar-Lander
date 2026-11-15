"""Causal, bounded acceleration-bias observer using measured transition residuals."""

import numpy as np

from .config import RobustConfig
from .model import FlightModel


class DisturbanceObserver:
    def __init__(self, model: FlightModel, config: RobustConfig):
        self.model = model
        self.config = config
        self.limits = np.array(
            [config.max_linear_accel, config.max_linear_accel, config.max_angular_accel]
        )
        self.reset()

    def reset(self):
        self.estimate = np.zeros(3)
        self.previous_state = None
        self.previous_action = None
        self.transitions = 0

    def update(self, state):
        if self.previous_state is None:
            return self.estimate.copy()
        predicted = self.model.step(self.previous_state, self.previous_action)
        residual = (np.asarray(state)[[2, 3, 5]] - predicted[[2, 3, 5]]) / 0.02
        self.transitions += 1
        if self.transitions > self.config.warmup_steps:
            # Vertical exhaust dispersion is especially noisy; filter it more strongly.
            tau = self.config.observer_tau * np.array([1.0, 2.0, 1.0])
            gain = 1 - np.exp(-0.02 / tau)
            residual = np.clip(residual, -3 * self.limits, 3 * self.limits)
            self.estimate += gain * (residual - self.estimate)
            self.estimate = np.clip(self.estimate, -self.limits, self.limits)
        return self.estimate.copy()

    def remember(self, state, action):
        self.previous_state = np.asarray(state).copy()
        self.previous_action = np.asarray(action).copy()

"""CEM feedback-shooting MPC; no simulator access or learned policy."""

import numpy as np

from .config import MPCConfig
from .guidance import feedback, targets
from .model import FlightModel, decode_observation, engine_power


class MPCController:
    def __init__(self, config: MPCConfig, model: FlightModel, seed=0):
        self.config = config
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.bias = np.zeros(2)
        self.tick = 0
        self.prediction = np.empty((0, 6))
        self.plans = 0
        self.last_cost = None

    def _feedback(self, states):
        return feedback(states, self.model)

    def _targets(self, states):
        return targets(states)

    def _advance(self, states, action):
        for _ in range(round(self.config.dt / 0.02)):
            states = self.model.step(states, action)
        return states

    def _rollout(self, state, offsets, record=False):
        c = self.config
        states = np.broadcast_to(state, (len(offsets), 6)).copy()
        costs = np.zeros(len(offsets))
        path = [states.copy()] if record else None
        for t in range(c.horizon):
            knot = min(t * c.knots // c.horizon, c.knots - 1)
            action = np.clip(self._feedback(states) + offsets[:, knot], -1, 1)
            main, side = engine_power(action)
            states = self._advance(states, action)
            angle_ref, vy_ref = self._targets(states)
            stage = (
                0.15 * states[:, 0] ** 2
                + 0.25 * states[:, 2] ** 2
                + 2.0 * (states[:, 3] - vy_ref) ** 2
                + 10 * (states[:, 4] - angle_ref) ** 2
                + 0.3 * states[:, 5] ** 2
                + 0.04 * main
                + 0.01 * np.abs(side)
            )
            # Soft safety costs: terrain is not represented by this flight model.
            stage += 200 * np.maximum(-states[:, 1], 0) ** 2
            stage += 100 * np.maximum(np.abs(states[:, 0]) - 9, 0) ** 2
            stage += 100 * np.maximum(np.abs(states[:, 4]) - 0.7, 0) ** 2
            costs += c.dt * stage
            if record:
                path.append(states.copy())
        costs += 0.5 * states[:, 0] ** 2 + 0.2 * states[:, 1] ** 2
        return costs, np.array(path) if record else None

    def plan(self, state):
        c = self.config
        mean = np.zeros((c.knots, 2))
        std = np.full_like(mean, c.noise)
        best = mean.copy()
        best_cost = self._rollout(state, best[None])[0][0]
        for _ in range(c.iterations):
            samples = np.clip(self.rng.normal(mean, std, size=(c.samples, c.knots, 2)), -1, 1)
            samples[0] = best
            samples[1] = 0
            costs, _ = self._rollout(state, samples)
            elite_indices = np.argsort(costs)[: c.elites]
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]]
                best = samples[elite_indices[0]].copy()
            elite = samples[elite_indices]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), c.min_std)
        _, path = self._rollout(state, best[None], record=True)
        self.prediction = path[:, 0]
        self.bias = best[0].copy()
        self.last_cost = float(best_cost)
        self.plans += 1

    def act(self, observation):
        state = decode_observation(observation)
        # Contact is a different dynamical mode: stop thrust and let Box2D settle.
        if observation[6] or observation[7]:
            self.prediction = np.empty((0, 6))
            return np.array([-1, 0], dtype=np.float32)
        if self.tick % self.config.replan_interval == 0:
            self.plan(state)
        self.tick += 1
        return np.clip(self._feedback(state) + self.bias, -1, 1).astype(np.float32)


class FeedbackController:
    def __init__(self, model):
        self.model = model
        self.prediction = np.empty((0, 6))

    def act(self, observation):
        state = decode_observation(observation)
        if observation[6] or observation[7]:
            return np.array([-1, 0], dtype=np.float32)
        return feedback(state, self.model).astype(np.float32)

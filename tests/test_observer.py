import numpy as np

from lunar_mpc.config import RobustConfig
from lunar_mpc.model import FlightModel
from lunar_mpc.observer import DisturbanceObserver


def test_observer_recovers_constant_force_and_torque_without_simulator_access():
    model = FlightModel()
    observer = DisturbanceObserver(model, RobustConfig(warmup_steps=0))
    state = np.array([0, 20, 0, 0, 0, 0], dtype=float)
    disturbance = np.array([1.4, -0.5, 0.3])
    action = np.array([0.2, 0.7])
    for _ in range(220):
        observer.update(state)
        observer.remember(state, action)
        state = model.step(state, action, disturbance=disturbance)
    np.testing.assert_allclose(observer.estimate, disturbance, atol=1e-4)


def test_observer_filters_noise_and_bounds_extreme_innovations():
    model = FlightModel()
    observer = DisturbanceObserver(model, RobustConfig(warmup_steps=0))
    rng = np.random.default_rng(41)
    state = np.array([0, 20, 0, 0, 0, 0], dtype=float)
    disturbance = np.array([1.2, 0, 0.2])
    estimates = []
    for i in range(400):
        observer.update(state)
        observer.remember(state, [-1, 0])
        noise = rng.normal(0, [2, 4, 0.7])
        state = model.step(state, [-1, 0], disturbance=disturbance + noise)
        if i > 100:
            estimates.append(observer.estimate.copy())
    estimates = np.array(estimates)
    np.testing.assert_allclose(estimates.mean(axis=0), disturbance, atol=0.35)
    assert np.all(estimates.std(axis=0) < [0.65, 0.85, 0.25])
    state[[2, 3, 5]] = 1e9
    assert np.all(np.abs(observer.update(state)) <= observer.limits)


def test_reset_and_bootstrap_cannot_reuse_old_contact_or_episode_history():
    observer = DisturbanceObserver(FlightModel(), RobustConfig())
    state = np.zeros(6)
    observer.remember(state, [1, 1])
    state[2] = 20
    assert observer.previous_state[2] == 0
    observer.estimate[:] = 2
    observer.reset()
    np.testing.assert_array_equal(observer.update(state), [0, 0, 0])
    assert observer.transitions == 0
    assert observer.previous_action is None


def test_startup_warmup_rejects_unmodelled_initial_leg_motion():
    observer = DisturbanceObserver(FlightModel(), RobustConfig(warmup_steps=2))
    state = np.zeros(6)
    for _ in range(2):
        observer.remember(state, [-1, 0])
        state = state + np.array([0, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(observer.update(state), [0, 0, 0])

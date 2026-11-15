import json
from pathlib import Path

import numpy as np

from lunar_mpc.config import MPCConfig
from lunar_mpc.controller import MPCController
from lunar_mpc.model import FlightModel, decode_observation, engine_power


def make_controller(seed=7):
    return MPCController(
        MPCConfig(samples=16, elites=4, horizon=12, knots=3, iterations=2),
        FlightModel(),
        seed,
    )


def test_seeded_controller_repeatability_and_action_bounds():
    observation = np.array([0.1, 1, 0.2, -0.4, 0.1, 0.1, 0, 0])
    left, right = make_controller(), make_controller()
    for _ in range(4):
        action = left.act(observation)
        np.testing.assert_array_equal(action, right.act(observation))
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all() and np.all(np.abs(action) <= 1)
    assert left.prediction.shape == (13, 6)
    assert left.plans == 2


def test_contact_mode_cuts_thrust_without_optimisation():
    controller = make_controller()
    for contacts in ([1, 0], [0, 1], [1, 1]):
        obs = np.array([0, 0.01, 0, -0.01, 0, 0, *contacts])
        np.testing.assert_array_equal(controller.act(obs), [-1, 0])
        assert controller.plans == 0
        assert controller.prediction.size == 0


def test_cem_keeps_a_candidate_at_least_as_good_as_baseline():
    controller = make_controller()
    obs = np.array([0.2, 0.9, 0.2, -0.3, 0, 0, 0, 0])
    state = decode_observation(obs)
    baseline, _ = controller._rollout(state, np.zeros((1, 3, 2)))
    controller.plan(state)
    # The best recorded path must start from the measured state, not stale episode state.
    np.testing.assert_allclose(controller.prediction[0], state)
    assert controller.last_cost <= baseline[0] + 1e-12


def test_reset_drops_previous_episode_state():
    controller = make_controller()
    controller.act(np.array([0, 1, 0, -0.2, 0, 0, 0, 0]))
    controller.reset()
    assert controller.tick == 0 and controller.plans == 0
    assert controller.prediction.size == 0
    np.testing.assert_array_equal(controller.bias, [0, 0])


def test_nominal_controller_matches_original_commit_on_fixed_observations():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "v1_actions.json").read_text())
    config = MPCConfig(**fixture["mpc"])
    for case in fixture["cases"]:
        controller = MPCController(config, FlightModel(), seed=case["seed"])
        actual = controller.act(np.asarray(case["observation"], dtype=float))
        # Commands inside a jet's dead zone are physically equivalent.
        np.testing.assert_allclose(
            engine_power(actual), engine_power(case["action"]), atol=1e-6, rtol=1e-6
        )

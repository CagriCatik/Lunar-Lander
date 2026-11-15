import numpy as np

from lunar_mpc.config import MPCConfig, RobustConfig
from lunar_mpc.model import FlightModel
from lunar_mpc.robust import RobustMPCController, touchdown_action


def make_controller(seed=7, risk_weight=0.5):
    return RobustMPCController(
        MPCConfig(samples=16, elites=4, horizon=12, knots=3, iterations=2),
        FlightModel(),
        RobustConfig(risk_weight=risk_weight),
        seed,
    )


def test_scenario_rollout_cost_and_nominal_path_shapes():
    controller = make_controller()
    state = np.array([1, 4, 0.2, -1, 0.1, 0])
    offsets = np.zeros((3, 3, 2))
    costs, path = controller._rollout(state, offsets, record=True)
    assert costs.shape == (3,)
    assert path.shape == (13, 3, 6)
    assert np.isfinite(costs).all() and np.isfinite(path).all()
    controller.plan(state)
    assert controller.prediction.shape == (13, 6)
    assert controller.last_cost <= costs[0] + 1e-12


def test_worst_case_weight_cannot_lower_the_cost_of_an_identical_candidate():
    state = np.array([0, 1, 0, -0.5, 0, 0], dtype=float)
    offsets = np.zeros((2, 3, 2))
    mean, _ = make_controller(risk_weight=0)._rollout(state, offsets)
    worst, _ = make_controller(risk_weight=1)._rollout(state, offsets)
    assert np.all(worst >= mean)


def test_bias_compensation_has_the_correct_force_and_torque_signs():
    controller = make_controller()
    state = np.array([0, 4, 0, -1, 0, 0], dtype=float)
    angle0, _ = controller._targets(state)
    controller.observer.estimate[:] = [1, 0, 2]
    angle1, _ = controller._targets(state)
    assert angle1 > angle0  # Positive wind requires tilt producing negative thrust acceleration.
    controller.observer.estimate[:] = [0, 0, 2]
    action = controller._feedback(state)
    assert action[1] > 0.5  # Positive disturbance torque requires negative jet torque.


def test_contact_resets_observer_and_forces_replanning_after_bounce():
    controller = make_controller()
    observation = np.array([0, 1, 0, -0.2, 0, 0, 0, 0], dtype=float)
    controller.act(observation)
    controller.observer.estimate[:] = 2
    observation[6] = 1
    np.testing.assert_array_equal(controller.act(observation), [-1, 0])
    np.testing.assert_array_equal(controller.disturbance, [0, 0, 0])
    assert controller.prediction.size == 0
    observation[6] = 0
    controller.act(observation)
    assert controller.plans == 2
    assert controller.observer.transitions == 0


def test_touchdown_stabiliser_reduces_tilt_and_releases_a_settled_lander():
    state = np.array([0, 0, 0, 0, 0.2, 0.4])
    assert touchdown_action(state)[1] > 0.5
    state[4:] *= -1
    assert touchdown_action(state)[1] < -0.5
    state[4:] = 0
    np.testing.assert_array_equal(touchdown_action(state), [-1, 0])


def test_repeatable_adaptive_actions_and_episode_reset():
    left, right = make_controller(), make_controller()
    observation = np.array([0.1, 1, 0.2, -0.4, 0.1, 0.1, 0, 0])
    for _ in range(4):
        np.testing.assert_array_equal(left.act(observation), right.act(observation))
    left.reset()
    np.testing.assert_array_equal(left.disturbance, [0, 0, 0])
    assert left.tick == 0 and left.plans == 0


def test_low_gravity_uses_velocity_switching_below_minimum_hover_thrust():
    controller = RobustMPCController(MPCConfig(), FlightModel(gravity=-6))
    state = np.array([0, 2, 0, 0, 0, 0], dtype=float)
    _, desired_vy = controller._targets(state)
    state[3] = desired_vy + 0.05
    assert controller._feedback(state)[0] == -1  # Descend faster: engine must switch off.
    state[3] = desired_vy - 0.05
    assert controller._feedback(state)[0] > 0  # Falling too fast: minimum thrust arrests it.

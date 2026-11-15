import json

import numpy as np
import pytest

from lunar_mpc.config import Config, EnvironmentConfig
from lunar_mpc.environment import classify_outcome, make_environment
from lunar_mpc.model import FlightModel, decode_observation
from lunar_mpc.runner import run_episode, save_episode


@pytest.mark.parametrize("action", [[-1, 0], [0, 0], [1, 0], [-1, 1], [-1, -1]])
def test_one_step_prediction_matches_box2d_with_bounded_model_error(action):
    env = make_environment(EnvironmentConfig())
    try:
        obs, _ = env.reset(seed=7)
        # Let articulated legs settle before comparing the rigid nominal model.
        for _ in range(20):
            obs, *_ = env.step(np.array([-1, 0], dtype=np.float32))
        predicted = FlightModel().step(decode_observation(obs), action)
        obs, *_ = env.step(np.array(action, dtype=np.float32))
        actual = decode_observation(obs)
        # Exhaust dispersion is stochastic and deliberately absent from the model.
        np.testing.assert_allclose(predicted[:2], actual[:2], atol=0.005)
        np.testing.assert_allclose(predicted[2:4], actual[2:4], atol=0.18)
        assert abs(predicted[4] - actual[4]) < 0.003
        assert abs(predicted[5] - actual[5]) < 0.06
    finally:
        env.close()


def test_timeout_telemetry_and_saved_artifacts(tmp_path):
    config = Config(environment=EnvironmentConfig(max_steps=3))
    summary, rows, predictions = run_episode(config, seed=0)
    assert summary["outcome"] == "timeout"
    assert len(rows) == 3
    save_episode(tmp_path, summary, rows, predictions, config, plot=True)
    assert json.loads((tmp_path / "summary.json").read_text())["seed"] == 0
    assert (tmp_path / "flight.png").stat().st_size > 1000
    assert (tmp_path / "diagnostics.png").stat().st_size > 1000
    assert len(rows[0]) == 18
    assert len((tmp_path / "telemetry.csv").read_text().splitlines()) == 4


def test_seed_zero_lands_without_training():
    summary, _, _ = run_episode(Config(), seed=0)
    assert summary["outcome"] == "landed"
    assert summary["reward"] > 200
    assert summary["touchdown"] is not None


def test_crash_takes_precedence_over_timeout():
    env = make_environment(EnvironmentConfig())
    try:
        env.reset(seed=0)
        env.unwrapped.game_over = True
        assert classify_outcome(env, True, True) == "crashed"
    finally:
        env.close()


def test_window_close_returns_a_saved_outcome(monkeypatch):
    from lunar_mpc.visualization import FlightOverlay

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(FlightOverlay, "draw", lambda *args: False)
    summary, rows, _ = run_episode(
        Config(environment=EnvironmentConfig(max_steps=10)), seed=0, render=True
    )
    assert summary["outcome"] == "user_closed"
    assert len(rows) == 1


@pytest.mark.parametrize("seed", [0, 8])
def test_wind_landings_cover_previously_failing_contact_cases(seed):
    from lunar_mpc.config import load_config

    summary, _, _ = run_episode(load_config("configs/wind.toml"), seed=seed)
    assert summary["outcome"] == "landed"
    assert summary["reward"] > 200


def test_nominal_controller_preserves_v1_reference_behaviour():
    from dataclasses import replace

    from lunar_mpc.config import load_config

    config = load_config("configs/v1-wind.toml")
    config = replace(config, environment=replace(config.environment, wind=False))
    summary, _, _ = run_episode(config, seed=0, controller_name="nominal")
    assert summary["outcome"] == "landed"
    # Box2D/platform floating-point differences need not preserve every reward digit.
    assert summary["reward"] == pytest.approx(244.610411, abs=1.0)


def test_low_gravity_lands_instead_of_hovering_above_the_minimum_throttle_limit():
    from lunar_mpc.benchmark import benchmark_cases

    controller, config = benchmark_cases()["low-gravity"]
    summary, _, _ = run_episode(config, seed=110, controller_name=controller)
    assert summary["outcome"] == "landed"

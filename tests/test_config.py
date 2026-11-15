import pytest

from lunar_mpc.config import EnvironmentConfig, MPCConfig, RobustConfig, load_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 1},
        {"samples": 0},
        {"samples": True},
        {"elites": 200},
        {"horizon": -1},
        {"knots": 31},
        {"dt": 0.03},
        {"dt": float("nan")},
        {"min_std": 0},
        {"replan_interval": 0},
        {"iterations": 1.5},
    ],
)
def test_invalid_solver_configuration(kwargs):
    with pytest.raises(ValueError):
        MPCConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gravity": 0},
        {"gravity": -12},
        {"gravity": float("nan")},
        {"wind": "false"},
        {"wind_power": -1},
        {"max_steps": 0},
    ],
)
def test_invalid_environment_configuration(kwargs):
    with pytest.raises(ValueError):
        EnvironmentConfig(**kwargs)


def test_unknown_keys_fail_instead_of_silently_using_defaults(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[mpc]\nsampels = 10\n")
    with pytest.raises(ValueError, match="Unknown mpc keys"):
        load_config(config)


def test_partial_configuration_preserves_defaults(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[mpc]\nsamples = 24\nelites = 4\n")
    loaded = load_config(config)
    assert loaded.mpc.samples == 24
    assert loaded.mpc.horizon == 20
    assert loaded.environment.gravity == -10


def test_shipped_presets_load():
    for path in (
        "configs/default.toml",
        "configs/wind.toml",
        "configs/strong-wind.toml",
        "configs/v1-wind.toml",
    ):
        assert load_config(path).mpc.samples > 1


@pytest.mark.parametrize("dt", [1e-12, 0.01, True])
def test_prediction_interval_cannot_skip_physics(dt):
    with pytest.raises(ValueError):
        MPCConfig(dt=dt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observer_tau": 0},
        {"wind_margin": -1},
        {"torque_margin": float("nan")},
        {"risk_weight": 1.1},
        {"warmup_steps": -1},
        {"warmup_steps": True},
        {"max_linear_accel": float("inf")},
        {"max_angular_accel": False},
    ],
)
def test_invalid_robust_configuration(kwargs):
    with pytest.raises(ValueError):
        RobustConfig(**kwargs)


def test_benchmark_reference_has_matched_environment_and_original_solver_budget():
    from lunar_mpc.benchmark import benchmark_cases

    cases = benchmark_cases()
    _, candidate = cases["wind"]
    controller, reference = cases["nominal-wind"]
    assert controller == "nominal"
    assert candidate.environment == reference.environment
    assert candidate.mpc.samples == 64
    assert reference.mpc.samples == 96

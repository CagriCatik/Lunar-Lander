import numpy as np
import pytest

from lunar_mpc.model import FlightModel, decode_observation, engine_power


@pytest.mark.parametrize("value,expected", [(-1, 0), (0, 0), (1e-6, 0.5000005), (1, 1)])
def test_main_engine_threshold(value, expected):
    main, side = engine_power([value, 0])
    assert main == pytest.approx(expected)
    assert side == 0


@pytest.mark.parametrize("value,expected", [(-1, -1), (-0.5, 0), (0.5, 0), (0.5001, 0.5001)])
def test_lateral_engine_threshold(value, expected):
    assert engine_power([-1, value])[1] == pytest.approx(expected)


def test_observation_scales_and_no_mutation():
    obs = np.ones(8)
    np.testing.assert_allclose(decode_observation(obs), [10, 20 / 3, 5, 7.5, 1, 2.5])
    np.testing.assert_array_equal(obs, np.ones(8))


@pytest.mark.parametrize("obs", [np.zeros(6), np.full(8, np.nan), np.full(8, np.inf)])
def test_invalid_observation_rejected(obs):
    with pytest.raises(ValueError, match="eight finite"):
        decode_observation(obs)


def test_ballistic_motion_and_vectorised_predictions():
    model = FlightModel()
    state = np.array([0, 5, 1, -1, 0, 0], dtype=float)
    result = model.step(state, [-1, 0])
    np.testing.assert_allclose(result, [0.02, 4.976, 1, -1.2, 0, 0])
    batch = model.step(np.tile(state, (3, 1)), [[-1, 0], [1, 0], [-1, 1]])
    np.testing.assert_allclose(batch[0], result)
    assert batch[1, 3] > result[3]
    assert batch[2, 2] > result[2]
    assert batch[2, 5] < 0  # Positive lateral command produces negative torque.
    np.testing.assert_array_equal(state, [0, 5, 1, -1, 0, 0])

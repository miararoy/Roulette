import numpy as np
from sklearn.metrics import mean_squared_error

from roulette.evaluation.utils import close_enough
from roulette.evaluation.experiment import Experiment, reg_mean, binary_mean, BASE_DIST
from roulette.evaluation.constants import ExperimentConstants


real = np.asarray([1, 2, 3])
model = np.asarray([1.1, 2.2, 3.3])
rand = np.asarray([4, 5, 6])


def test_experiment_init_happy():
    exp = Experiment(
        "reg",
        real,
        real,
        model,
    )
    assert set(exp.experiment_data.Real) & set(real)
    assert set(exp.experiment_data.Model) & set(model)
    assert set(exp.experiment_data.Mean) & set([2, 2, 2])
    for x in exp.experiment_data.Rand:
        assert real.min() <= x <= real.max()


def test_experiment_init_unhappy():
    try:
        Experiment(
            "reg",
            real[:-1],
            real[:-1],
            model,
        )
    except Exception as e:
        assert isinstance(e, ValueError)


def test_score_calculation():
    exp = Experiment(
        "reg",
        real,
        real,
        model
    )
    score = exp.score(mean_squared_error)
    assert close_enough(score.Model, 0.04666666)
    assert close_enough(score.Mean, 2 / 3)


def test_mean_arr():
    reg_arr = np.asarray([0.1, 0.2, 0.3, 0.4, 0.0])
    bin_arr = np.asarray([1, 1, 1, 0, 0])
    assert np.array_equal(reg_mean(reg_arr, 3), np.asarray([0.2, 0.2, 0.2]))
    assert np.array_equal(binary_mean(bin_arr, 3), np.asarray([1, 1, 1]))


def test_exp_type_match():
    for _type in ExperimentConstants.TYPES:
        assert _type in BASE_DIST
        assert "mean" in BASE_DIST[_type] and "rand" in BASE_DIST[_type]

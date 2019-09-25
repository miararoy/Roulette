import numpy as np
from sklearn.metrics import mean_squared_error

from bliz.evaluation.utils import close_enough
from bliz.evaluation.experiment import Experiment

real = [1, 2, 3]
model = [1.1, 2.2, 3.3]
rand = [4, 5, 6]


def test_experiment_init_happy():
    exp = Experiment(real, np.asanyarray([real]), model, 1)
    assert set(exp.experiment_data.Real) & set(real)
    assert set(exp.experiment_data.Model) & set(model)
    assert set(exp.experiment_data.Mean) & set([2, 2, 2])
    for x in exp.experiment_data.Rand:
        assert 0 <= x < 1


def test_experiment_init_scaled_random_happy():
    rand_scale = 5
    exp = Experiment(real, np.asanyarray([real]), model, rand_scale)
    for x in exp.experiment_data.Rand:
        assert 0 <= x < 1 * rand_scale


def test_experiment_others():
    exp = Experiment(
        real,
        np.asanyarray([real]),
        model,
        1,
        other_models={
            "A": real,
            "B": real
        })
    assert set(exp.experiment_data.OtherModels["A"]) & set(real)
    assert set(exp.experiment_data.OtherModels["B"]) & set(real)


def test_experiment_init_unhappy():
    try:
        Experiment(real[:-1], np.asanyarray([real][:-1]), model, 1)
    except Exception as e:
        assert isinstance(e, ValueError)


def test_score_calculation():
    exp = Experiment(
        real,
        np.asanyarray([real]),
        model,
        1,
        other_models={
            "A": real,
            "B": real
        })
    score = exp.score(mean_squared_error)
    assert close_enough(score.Model, 0.04666666)
    assert close_enough(score.Div, 5)
    assert close_enough(score.Mean, 2 / 3)
    assert close_enough(score.OtherModels["A"], 0.0)
    assert close_enough(score.OtherModels["B"], 0.0)

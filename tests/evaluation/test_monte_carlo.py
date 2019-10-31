import numpy as np
from sklearn.metrics import mean_squared_error


from roulette.evaluation.monte_carlo import MonteCarloSimulation

real = np.asarray([1, 2, 3])
model = np.asarray([1.1, 2.2, 3.3])
rand = np.asarray([4, 5, 6])

MC = MonteCarloSimulation(exp_type="reg")
MC.load_experiment(
    real,
    real,
    model
)
MC.load_experiment(
    real,
    real,
    model
)
MC.digest(mean_squared_error)


def test_monte_carlo_score():
    expected = {
        'model': [0.04666666666666666, 0.04666666666666666],
    }
    for k in expected:
        if k != "rand":
            assert set(MC.scores[k]) & set(expected[k])


def test_monte_carlo_metrics():
    metrics = MC.get_metrics()
    assert metrics.Certainty == -1


def test_as_dict():
    MC_as_dict = MC.metrics_as_dict()
    assert isinstance(MC_as_dict, dict)
    assert ("discriminability" in MC_as_dict and "certainty" in MC_as_dict)


def test_to_json():
    passed = True
    try:
        MC.metrics_to_json("/tmp/test.json")
    except Exception as e:
        print(e)
        passed = False
    assert passed


def test_summery():
    passed = True
    try:
        MC.save_experiment_summery("/tmp/test_summery.json")
    except Exception as e:
        print(e)
        passed = False
    assert passed


def test_plotting():
    passed = True
    MC.load_experiment([4.5, 5.4, 1.1],
                       np.asarray([4.5, 5.4, 1.1]), [4, 2, 1],
                       )
    MC.load_experiment([5, 3, 1],
                       np.asarray([5, 3, 1]), [1, 1, 1],
                       )
    MC.digest(mean_squared_error)
    try:
        MC.plot("/tmp")
    except Exception:
        passed = False
    assert passed

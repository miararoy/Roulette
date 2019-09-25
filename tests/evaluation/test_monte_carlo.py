import numpy as np
from sklearn.metrics import mean_squared_error


from bliz.evaluation.monte_carlo import MonteCarloSimulation

real = [1, 2, 3]
model = [1.1, 2.2, 3.3]
rand = [4, 5, 6]

MC = MonteCarloSimulation()
MC.load_experiment(
    real,
    np.asanyarray([real]),
    model,
    others={
        "rand_a": rand,
        "rand_b": [x * 2 for x in rand]
    })
MC.load_experiment(
    real,
    np.asanyarray([real]),
    model,
    others={
        "rand_a": rand,
        "rand_b": real
    })
MC.digest(mean_squared_error)


def test_monte_carlo_score():
    expected = {
        'model': [0.04666666666666666, 0.04666666666666666],
        'rand_a': [9.0, 9.0],
        'rand_b': [64.66666666666667, 0.0]
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
    assert ("discriminability" in MC_as_dict and "certainty" in MC_as_dict
            and "divergency" in MC_as_dict)


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
                       others={
                           "rand_a": [1, 1, 1],
                           "rand_b": [x * 1.22 for x in rand]
    })
    MC.load_experiment([5, 3, 1],
                       np.asarray([5, 3, 1]), [1, 1, 1],
                       others={
                           "rand_a": [0.3, 0.3, 0.3],
                           "rand_b": [0.1, 0.1, 9]
    })
    MC.digest(mean_squared_error)
    try:
        MC.plot("/tmp")
    except Exception:
        passed = False
    assert passed

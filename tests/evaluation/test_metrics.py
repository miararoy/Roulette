import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ehmodelevaluation import _interpolate_bins,\
                              close_enough,\
                              weighted_interpolated_error,\
                              get_weight_metric


def test_interploate_bins_unitary():
    bins = [0, 0.25, 0.5, 1]
    assert (close_enough((_interpolate_bins(bins, len(bins))), bins, 2))


def test_interploate_bins():
    bins = [0.0, 0.3, 0.7, 1]
    new_bins = _interpolate_bins(bins, 5)
    assert len(new_bins) == 5
    assert max(bins) == max(new_bins) and min(bins) == min(new_bins)
    assert [(new_bins[i] <= bins[i] <= new_bins[i + 1])
            for i, x in enumerate(bins)].count(True) == len(bins)


def test_weighted_interpolated_error_sanity():
    real = [0.1, 0.3, 0.5, 0.7, 0.4, 0.2, 0.45, 0.8, 0.9]
    pred = [0.4, 0.5, 0.2, 1.0, 0.4, 0.3, 0.15, 0.5, 0.7]
    bins = [0, 0.3, 0.7, 1]
    W = np.asarray([
        [1, 3, 5],
        [2, 1, 2],
        [3, 2, 1],
    ], order='C')

    assert mean_squared_error(real, pred) < weighted_interpolated_error(
        real, bins, W, 'mse')(real, pred)
    assert mean_absolute_error(real, pred) < weighted_interpolated_error(
        real, bins, W, 'abs')(real, pred)


def test_get_weight_metric():
    bins = [0, 0.3, 0.7, 1]
    W = np.asarray([
        [1, 3, 5],
        [2, 1, 2],
        [3, 2, 1],
    ], order='C')
    weight_function, nw = get_weight_metric(bins, W, 3)
    print(nw)
    assert close_enough(weight_function(1, 1), 1)

import scipy as sp
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .utils import close_enough,\
                   validate_multiple_lists_length,\
                   samples_to_bin_numbers

MSE = mean_squared_error
ABS_ERR = mean_absolute_error
ERR_TYPES = ["mse", "abs"]
WD = sp.stats.wasserstein_distance


def discriminability(
        a,
        mean,
        rand,
):
    """calculates the discrimination of model given error distribution a,
    against random and mean `models`

    Args:
        a (array-like): the error distribution of model
        mean (array-like): the error distribution against `guessing` the mean value
        rand (array-like): the error distribution against `guessing` random values

    Returns:
        d(float): the rate of discrimination

    """
    if np.mean(a) > np.mean(mean):
        return 0.0
    else:
        MX =  WD(np.asarray([0.0] * len(a)), mean)
        B = WD(a, mean)
        C = WD(mean, rand)

        return (B/C) / (MX/C)


def certainty(
        a,
        rand,
):
    """the inverse of the standard deviation of the error distribution of a model

    Args:
        a (array-like): the error distribution of model
        rand (array-like): the error distribution against `guessing` random values

    Returns:
        c (float): the std^-1
    """
    std_a = np.std(a)
    std_rand = np.std(rand)
    if close_enough(std_a, 0.0, 16):
        return -1
    else:
        return float(std_rand / std_a)


def divergency(s):
    """the average distribution distance between the ground truth and the predictions
    veotrs

    Args:
        s (array-like): vector of the distances of each sample

    Returns:
        d (float): mean value of distnace
    """
    return float(np.mean(s))


def _weighted_error(
        bins,
        weights: np.ndarray,
        how: str,
):
    """returns an error calculating function according to weights of samples
    given by the weights matrix

    Args:
        bins (array-like): the bin edges for grouping the r, p vectors
        weights (np.ndarray): an N X N matrix representing the cost of predicting 
                              in different bin
        how (str): should the error be absolute / squared, gets values in ['abs', 'mse']
                   defaults to 'mse'
    
    Returns:
        _calc_metric (callable): a functions that gets two vectors and returns the calculated error 
                                 with accordance to bins, weights and how
    """
    if weights.shape == (len(bins) - 1, len(bins) - 1):
        _step = (lambda i, j: abs(i - j)) if how == 'abs' else (
            lambda i, j: ((i - j)**2))

        def _calc_metric(r, p):
            if validate_multiple_lists_length(r, p):
                weighted_sum = 0.0
                sum_of_weights = 0.0
                r_bins, p_bins = samples_to_bin_numbers(r, p, bins=bins)
                for i, j, ib, jb in zip(r, p, r_bins, p_bins):
                    w = weights[ib, jb]
                    diff = _step(i, j)
                    weighted_sum += diff * w
                    sum_of_weights += w
                return (weighted_sum / sum_of_weights)
            else:
                raise ValueError(
                    "r, p should be of same length but len(r) = {lr}, len(p) = {lp}"
                    .format(lr=len(r), lp=len(p)))

        return _calc_metric
    else:
        raise IndexError(
            "size of weights matrix should be same as len(bins) - 1: {} X {}".
            format(len(bins) - 1,
                   len(bins) - 1))


def _resize_matrix_function(mx: np.ndarray, kind='linear'):
    """creates an interpolation function with accordance to the matrix mx

    Args:
        mx (np.ndarray): a cubic matrix
        kind (str): type of interpolation, gets values in linear, cubic

    Returns:
        f (callable): an interpolation function 
    """
    x = np.linspace(0, 1, mx.shape[0])
    y = np.linspace(0, 1, mx.shape[1])
    f = sp.interpolate.interp2d(x, y, mx, kind=kind)
    return f


def _resize_vector_function(v):
    """creates an interpolation function with accordance to the vector v.
    with which we could expand the vector v

    Args:
        v (array-like): a 1d vector
        
    Returns:
        f (callable): an interpolation function in 1d
    """
    _v = np.asarray(v)
    f = sp.interpolate.interp1d(
        np.linspace(_v.min(), _v.max(), len(v)), v, kind='cubic')
    return f


def _interpolate_weights(
        w: np.ndarray,
        new_size: int,
):
    """extends a cubic weight matrix to new size

    Args:
        w (np.ndarray): a cubic matrix of weights
        new_size (int): the new size of output matrix
        
    Returns:
        _w (np.ndarray): a cubic weight matrix with size new_size
    """
    if new_size == w.shape[0]:
        return w
    else:
        _w = w
        if max(w.shape) <= 3:
            _w = _resize_matrix_function(w)(np.linspace(0, 1, 4),
                                            np.linspace(0, 1, 4))
        _w = _resize_matrix_function(
            _w, kind='cubic')(np.linspace(0, 1, new_size),
                              np.linspace(0, 1, new_size))
        for i in range(new_size):
            _w[i, i] = 1
    return _w


def _interpolate_bins(bins, new_size: int):
    """extends a bins vector

    Args:
        bins (array-like): bin edges vector
        new_size (int): the new size of output matrix
        
    Returns:
        new_bins (array-like): a cubic weight matrix with size new_size
    """
    bins_as_vec = np.asarray(bins)
    new_bins_x = np.linspace(bins_as_vec.min(), bins_as_vec.max(), new_size)
    new_bins = _resize_vector_function(bins)(new_bins_x)
    return new_bins


def get_weight_metric(bins, weights: np.ndarray,
                      new_size) -> (callable, np.ndarray):
    """returns an interpolated weight metric

    Args:
        bins(array-like): array of bin edges
        weights(np.ndarray): a small (3X3 usually) matrix of bin-weight errors
        new_size(int): interpolation new size ( > len(bins)-1)
    
    Returns:
        T(tuple[callable, np.ndarray]): a tuple where the first arg is a function to calculate
            weights between two scores and the second is matrix of new weights
    """
    new_bins = _interpolate_bins(bins, new_size + 1)
    new_weights = _interpolate_weights(weights, new_size)

    def _sample_weight(a, b):
        a_bins, b_bins = samples_to_bin_numbers(
            [a], [b], bins=new_bins
        )  # TODO: change samples to bins to handle list and scalar
        return new_weights[a_bins[0], b_bins[0]]

    return _sample_weight, new_weights


def weighted_interpolated_error(
        size,
        bins,
        weights: np.ndarray,
        error_type: str,
):
    """returns a error calculation function

    Args:
        r_vector(array-like): result vector
        bins(list): bin edges vector
        weights(np.ndarray): matrix of weights between bins
        error_type(str): type of error to use, (abs/mse)

    Returns:
        _calc_metric (callable): a functions that gets two vectors and returns the calculated error 
                                 with accordance to bins, weights and how
    """
    if error_type in ERR_TYPES:
        new_size = max(9, size)
        new_bins = _interpolate_bins(bins, new_size + 1)
        new_weights = _interpolate_weights(weights, new_size)
        return _weighted_error(new_bins, new_weights, how=error_type)
    else:
        raise ValueError("error type must be in {}".format(ERR_TYPES))


__all__ = [
    'discriminability', 'certainty', 'divergency',
    'weighted_interpolated_error'
]

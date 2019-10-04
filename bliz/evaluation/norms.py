import numpy as np


def min_max_norm(y):
    """calculates the min max norm of an array

    Args:
        y(array-like): an array of int or floats

    Returns:
        out(array-like): a normalized array

    Raises:
        ValueError: if y is not array-like
    """
    try:
        _y = np.asarray(y)
    except BaseException as be:
        raise ValueError(
            "y must be scalar or array like :: {}".format(
                str(be)))
    return (_y - _y.min()) / (_y.max() - _y.min())


NORM_MAP = {
    "min_max": min_max_norm
}


def get_normalizer(normalizer: str):
    """returns a normalizer function using a string

    Args:
        normalizer(str): normalizer identifier key

    Returns:
        normalizer(Callable): a normalizing method

    Raises:
        KeyError: if normalizer is not available or key not found
    """
    if normalizer in NORM_MAP:
        return NORM_MAP[normalizer]
    else:
        raise KeyError("normalizer is not in available normalizers: [{}]".format(
            NORM_MAP.keys()
        ))

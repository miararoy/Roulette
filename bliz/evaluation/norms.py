import numpy as np


def min_max_norm(y):
    try:
        _y = np.asarray(y)
    except BaseException as be:
        raise ValueError(
            "y must be scalar or array like :: {}".format(
                str(be)))
    return (_y - _y.min()) / (_y.max() - _y.min())

import numpy as np


def min_max_norm(y):
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
    if normalizer in NORM_MAP:
        return NORM_MAP[normalizer]
    else:
        raise ValueError("normalizer is not in available normalizers: [{}]".format(
            NORM_MAP.keys()
        ))
import numpy as np


def validate_multiple_lists_length(*lists) -> bool:
    """Validates that a list of lists is of the same length

    Args:
        lists(list): a list of lists to be checked

    Retuens:
        b(bool): True if all list of the same length, False else. 

    """
    list_len = -1
    for l in lists:
        try:
            iter(l)
        except BaseException:
            return False
        if list_len == -1:  # first list
            list_len = len(l)
        else:
            if list_len != len(l):
                return False
    return True


def close_enough(a, b, precision=3) -> bool:
    """checks weather to Array-like are close enough to each other

    Args:
        a: first argument
        b: second argument
        precision(int): the decimal level of accuracy, 3 == 0.001

    Returns:
        out(bool): True, if a and b are close enough on precision, False else.

    """

    def _check_precision(x, y):
        if round(x, precision) == round(y, precision):
            return True
        else:
            return False

    try:
        if a == b:
            return True
    except ValueError:  # comparing numpy.ndarray for example
        pass
    try:
        iter(a)
        iter(b)
    except BaseException:  # objects are not iterable == scalar
        return _check_precision(a, b)
    out = True
    for i, j in zip(
            a,
            b):  # if we got here we need to iterate over all args in the array
        out = out and _check_precision(i, j)
    return out


def _sample_to_bin(a, bins):
    """return the bin number of each sample

    Args:
        a: list of array like, same length
        bins: list of bin edges
    Retrurns:
        l: list same length as 'a' containing the bin number instead of values
    """
    bins_list = []
    for x in a:
        x_bin, _ = np.histogram(x, bins)
        bins_list.append(x_bin.argmax())
    return bins_list


def samples_to_bin_numbers(*lists, bins):
    """return the bin number of each sample

    Args:
        lists: list of array like, same length
        bins: list of bin edges
    Raises:
        ValueError: if lists not of same length
    Retruns:
        lists_out: list same length as *lists containing the bin number instead of values
    """
    if validate_multiple_lists_length(lists):
        lists_out = []
        for l in lists:
            lists_out.append(_sample_to_bin(l, bins))
        return tuple(lists_out)
    else:
        raise ValueError("lists should all have the same length!")


def parse_ndarray_as_float_list(arr: np.ndarray) -> list:
    """returns a list of python floats rather than numpy.ndarray of float32

    Args:
        arr(numpy.ndarray): an array of float32

    Returns:
        lst(list): list of float
    """
    try:
        lst = [float(x) for x in arr]
    except Exception as e:
        raise e
    return lst


def is_binary(a) -> bool:
    if len(np.asarray(a).shape) >= 1:
        return False
    if (np.bincount(a) == 2
        and (np.max(a) == 1 or np.max(a) == 0)
        and (np.min(a) == 0 or np.min(a) == 1)):
        return True
    else:
        return False


__all__ = [
    "samples_to_bin_numbers", "_sample_to_bin", "close_enough",
    "validate_multiple_lists_length"
]

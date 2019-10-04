import pandas as pd
from sklearn.model_selection import train_test_split
from time import time


def prepare_data_for_training(
    df: pd.core.frame.DataFrame,
    target: str,
    index_column: str = None,
    validation_test_size: float = 0.2,
    verbose: bool = False,
):
    """takes input data and setting index, seperating target column
    and retuns a random split of given size

    Args:
        df(pd.core.frame.DataFrame): input data
        target(str): target column name
        index_column(str): index column name, default to None (no indexing)
        validation_test_size(float): test set size 0.0-1.0 if 0 all will be training set
            defaults to 0.2
        verbose(bool): prints out the shape of test and train datasets

    Returns:
        _x, test_x, _y, test_y: a tuple of datasets - train, test

    """
    if index_column:
        _df = df.set_index(index_column)
    else:
        _df = df
    _target = _df[target]
    _data = _df.drop(target, axis=1)
    if validation_test_size == 0:
        _x, test_x, _y, test_y = _data, pd.DataFrame(), _target, pd.Series()
    else:
        _x, test_x, _y, test_y = train_test_split(
            _data,
            _target,
            test_size=validation_test_size,
            random_state=int(time()),
        )
    if verbose:
        print("shape of training data = {}".format(_x.shape))
        print("shape of training data target = {}".format(_y.shape))
        print("shape of validation data = {}".format(test_x.shape))
        print(
            "shape of validation data target = {}\n".format(
                test_y.shape))
    return _x, _y.values, test_x, test_y.values

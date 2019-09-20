import pandas as pd
from sklearn.model_selection import train_test_split
from time import time


def prepare_data_for_training(
    df: pd.core.frame.DataFrame,
    target: str,
    index_column: str=None,
    validation_test_size: float=0.2,
    verbose: bool=False,
):  
    if index_column:
        _df = df.set_index(index_column)
    else:
        _df = df
    _target = _df[target]
    _data = _df.drop(target, axis=1)
    _x, validation_x, _y, validation_y = train_test_split(
        _data,
        _target,
        test_size=validation_test_size,
        random_state=int(time()),
    )
    if verbose:
        print("shape of training data = {}".format(_x.shape))
        print("shape of training data target = {}".format(_y.shape))
        print("shape of validation data = {}".format(validation_x.shape))
        print("shape of validation data target = {}\n".format(validation_y.shape))
    return _x, _y.values, validation_x, validation_y.values

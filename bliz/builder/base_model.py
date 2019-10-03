import os
import pickle
from typing import Union
from time import time
from abc import ABC, abstractmethod

import joblib
import numpy as np
from pandas.core.frame import DataFrame

from bliz.builder.save_load_model import ModelFileHandler


class BaseModel(ABC):
    def __init__(
        self,
    ):
        self.model_name = str(int(time()))
        self.model = None

    @abstractmethod
    def fit(
        self,
        X,
        y
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        X,
    ) -> Union[DataFrame, np.ndarray, list]:
        pass

    def save(
        self,
        base_path,
    ):
        model_dir = os.path.join(base_path, self.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "{}.joblib".format(self.model_name))
        try:
            with open(model_path, 'wb') as model_file:
                joblib.dump(
                    self,
                    model_file,
                )
            return model_dir
        except pickle.PicklingError as e:
            print("Cannot picke model at {p} due to {e}".format(
                p=model_path, e=e))
            raise e

    def load(
        self,
        model_path,
    ):
        self.model = ModelFileHandler().load(model_path)

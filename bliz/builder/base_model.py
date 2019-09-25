from typing import Union
from time import time
from abc import ABC, abstractmethod

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
        return ModelFileHandler(model=self).save(base_path)

    def load(
        self,
        model_path,
    ):
        self.model = ModelFileHandler().load(model_path)

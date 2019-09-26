import os
from typing import Union
import random

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from bliz.builder.data_prep import prepare_data_for_training
from bliz.builder.save_load_model import load_model
from bliz.builder.utils import is_regression_metric, is_binary_classification_metric
from bliz.evaluation import MonteCarloSimulation
from bliz.evaluation.metrics import get_regression_metric, get_binary_classification_metric
from bliz.logger import Logger

logger = Logger("builder").get_logger()

BUILD_DIR_NAME = "build"


class Builder(object):

    logger = Logger("Builder").get_logger()

    def __init__(
        self,
        path_to_model: str,
        data: pd.core.frame.DataFrame,
        target: str,
        metric: callable,
        index: str = None
    ):
        """Base builder class, handles the MC iteration 

        Args:
            data(pd.DataFrame): 
            path_to_model(str): path to a folder containing model.py file
            target(str): name of column in data to target for
            index(str): name of colummn in data to use as index, default None
        """
        self.data = data
        self.path_to_model = path_to_model
        self.target = target
        self.index_column = index
        self.final_model = None
        self.result = None

    def _build(
        self,
        n_experiments: int,
        metric,
    ):
        """Build and evaluates the model loaded

        Args:
            n_experiments(int): number of experiment to execute
        """
        self.logger.info("Initiating {} Epochs".format(n_experiments))
        Model = load_model(self.path_to_model)
        self.MC_simulation = MonteCarloSimulation()
        with tqdm(total=n_experiments, desc=" Training Model") as bar:
            for _ in range(n_experiments):
                X, y, v_X, v_y = prepare_data_for_training(
                    df=self.data,
                    target=self.target,
                    index_column=self.index_column,
                    validation_test_size=random.uniform(0.15, 0.25),
                )
                this_model = Model()
                this_model.fit(X, y)
                this_prediction = this_model.predict(v_X)
                self.MC_simulation.load_experiment(
                    v_y,
                    y,
                    this_prediction
                )
                bar.update(1)
        self.MC_simulation.digest(metric=metric)
        self.result = self.MC_simulation.metrics_as_dict()

    def finalize_model(self,):
        X, y, _, _ = prepare_data_for_training(
            df=self.data,
            target=self.target,
            index_column=self.index_column,
            validation_test_size=0,
        )
        self.logger.info("Finalzing model")
        Model = load_model(self.path_to_model)
        self.logger.info("Training model on all data")
        self.final_model = Model()
        self.final_model.fit(X, y)

    def get_results(self) -> dict:
        if self.result:
            return self.result
        else:
            raise RuntimeError("You must use build() to get results")

    def plot(self, title=None):
        plt.clf()
        self.MC_simulation.plot(title=title)
        plt.show()

    def save(self, plot=True, summery=False, data=False):
        if self.final_model:
            model_dir = self.final_model.save(
                os.path.join(self.path_to_model, BUILD_DIR_NAME))
            self.logger.info("saved model to {}".format(model_dir))
        else:
            raise RuntimeError(
                "You did not finalize model thus no model will be saved, use .finalize_model() method to save model")
        if self.result:
            self.logger.info("saving model metrics")
            self.MC_simulation.metrics_to_json(os.path.join(
                model_dir, "{}_metadata.json".format(self.final_model.model_name)))
            if plot:
                self.logger.info("saving simultion plot")
                self.MC_simulation.plot(path=model_dir)
            else:
                self.logger.info("plot=False will not save evaluation plot")
        else:
            raise RuntimeError("You must use build() to save")
        if summery:
            self.logger.info("saving experiment summery")
            self.MC_simulation.save_experiment_summery(os.path.join(
                model_dir, "{}_summery.json".format(self.final_model.model_name)))
        else:
            self.logger.info(
                "summery = False, will not save experiment summery")
        if data:
            self.logger.info("saving input data")
            self.data.to_csv(os.path.join(
                model_dir, "{}_data.csv".format(self.final_model.model_name)))
        else:
            self.logger.info("data = False, will not save experiment data")
        return model_dir


class RegressionBuilder(Builder):
    def __init__(
        self,
        path_to_model: str,
        data: pd.core.frame.DataFrame,
        target: str,
        metric: Union[str, callable],
        index: str = None
    ):
        super().__init__(path_to_model, data, target, metric)
        if hasattr(metric, "__call__"):
            assert is_regression_metric(metric)
            self._metric = metric
        elif isinstance(metric, str):
            self._metric = get_regression_metric(metric)
        else:
            raise ValueError("metric should be str or callable, not {}".format(type(metric)))

    def build(
        self,
        n_experiments: int,
    ):
        self._build(n_experiments, self._metric)
        

class BinaryClassificationBuilder(Builder):
    def __init__(
        self,
        path_to_model: str,
        data: pd.core.frame.DataFrame,
        target: str,
        metric: Union[str, callable],
        index: str = None
    ):
        super().__init__(path_to_model, data, target, metric)
        if isinstance(metric, callable):
            assert is_binary_classification_metric(metric)
            self._metric = metric
        elif isinstance(metric, str):
            self._metric = get_binary_classification_metric(metric)
        else:
            raise ValueError("metric should be str or callable, not {}".format(type(metric)))

    def build(
        self,
        n_experiments: int,
    ):
        self._build(n_experiments, self._metric)


# class RegressionBuilder(object):

#     logger = Logger("RegressionBuilder").get_logger()

#     def __init__(
#         self,
#         path_to_model: str,
#         data: pd.core.frame.DataFrame,
#         target: str,
#         metric: callable,
#         index: str = None
#     ):
#         self.data = data
#         self.path_to_model = path_to_model
#         self.target = target
#         self.index_column = index
#         assert is_regression_metric(metric)
#         self._metric = metric
#         self.final_model = None
#         self.result = None

#     def build(
#         self,
#         n_experiments: int,
#     ):
#         self.logger.info("Initiating {} Epochs".format(n_experiments))
#         Model = load_model(self.path_to_model)
#         self.MC_simulation = MonteCarloSimulation()
#         with tqdm(total=n_experiments, desc=" Training Model") as bar:
#             for _ in range(n_experiments):
#                 X, y, v_X, v_y = prepare_data_for_training(
#                     df=self.data,
#                     target=self.target,
#                     index_column=self.index_column,
#                     validation_test_size=random.uniform(0.15, 0.25),
#                 )
#                 this_model = Model()
#                 this_model.fit(X, y)
#                 this_prediction = this_model.predict(v_X)
#                 self.MC_simulation.load_experiment(
#                     v_y,
#                     y,
#                     this_prediction
#                 )
#                 bar.update(1)
#         self.MC_simulation.digest(metric=self._metric)
#         self.result = self.MC_simulation.metrics_as_dict()

#     def finalize_model(self,):
#         X, y, _, _ = prepare_data_for_training(
#             df=self.data,
#             target=self.target,
#             index_column=self.index_column,
#             validation_test_size=0,
#         )
#         self.logger.info("Finalzing model")
#         Model = load_model(self.path_to_model)
#         self.logger.info("Training model on all data")
#         self.final_model = Model()
#         self.final_model.fit(X, y)

#     def get_results(self) -> dict:
#         if self.result:
#             return self.result
#         else:
#             raise RuntimeError("You must use build() to get results")

#     def plot(self, title=None):
#         plt.clf()
#         self.MC_simulation.plot(title=title)
#         plt.show()

#     def save(self, plot=True, summery=False, data=False):
#         if self.final_model:
#             model_dir = self.final_model.save(
#                 os.path.join(self.path_to_model, BUILD_DIR_NAME))
#             self.logger.info("saved model to {}".format(model_dir))
#         else:
#             raise RuntimeError(
#                 "You did not finalize model thus no model will be saved, use .finalize_model() method to save model")
#         if self.result:
#             self.logger.info("saving model metrics")
#             self.MC_simulation.metrics_to_json(os.path.join(
#                 model_dir, "{}_metadata.json".format(self.final_model.model_name)))
#             if plot:
#                 self.logger.info("saving simultion plot")
#                 self.MC_simulation.plot(path=model_dir)
#             else:
#                 self.logger.info("plot=False will not save evaluation plot")
#         else:
#             raise RuntimeError("You must use build() to save")
#         if summery:
#             self.logger.info("saving experiment summery")
#             self.MC_simulation.save_experiment_summery(os.path.join(
#                 model_dir, "{}_summery.json".format(self.final_model.model_name)))
#         else:
#             self.logger.info(
#                 "summery = False, will not save experiment summery")
#         if data:
#             self.logger.info("saving input data")
#             self.data.to_csv(os.path.join(
#                 model_dir, "{}_data.csv".format(self.final_model.model_name)))
#         else:
#             self.logger.info("data = False, will not save experiment data")
#         return model_dir

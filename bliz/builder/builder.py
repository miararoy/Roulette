import os
# import json
# import operator
import random
# import pickle
# from pprint import pprint
# from time import time, sleep
# import importlib.util
# import argparse

import pandas as pd
# import numpy as np
# import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
# from sklearn.utils import shuffle
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression

from bliz.builder.data_prep import prepare_data_for_training
# from bliz.builder.constants import BuildConstants
from bliz.builder.save_load_model import load_model
from bliz.builder.utils import is_regression_metric
from bliz.evaluation import MonteCarloSimulation  # , weighted_interpolated_error
from bliz.logger import Logger

logger = Logger("builder").get_logger()

BUILD_DIR_NAME = "build"


class RegressionBuilder(object):

    logger = Logger("RegressionBuilder").get_logger()

    def __init__(
        self,
        path_to_model: str,
        data: pd.core.frame.DataFrame,
        target: str,
        metric: callable,
        index: str = None
    ):
        self.data = data
        self.path_to_model = path_to_model
        self.target = target
        self.index_column = index
        assert is_regression_metric(metric)
        self._metric = metric
        self.final_model = None
        self.result = None

    def build(
        self,
        n_experiments: int,
    ):
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
        self.MC_simulation.digest(metric=self._metric)
        self.result = self.MC_simulation.metrics_as_dict()

    # def evaluate(self, weights=None, bins=None, error_type='mse'):
    #     if bins is None:
    #         self.logger.warn("bins value is None using default == [0, 0.3, 0.7, 1]")
    #         _bins = [0, 0.3, 0.7, 1]
    #     elif len(bins) == BuildConstants.WEIGHT_MATRIX_SIZE + 1:
    #         _bins = bins
    #     else:
    #         raise ValueError("bin length should be {} + 1".format(BuildConstants.WEIGHT_MATRIX_SIZE))
    #     if weights is None:
    #         self.logger.warn("weights value is None using default == [[1,1,1],[1,1,1],[1,1,1]]")
    #         _weights = np.asarray(
    #             [
    #                 [1,1,1],
    #                 [1,1,1],
    #                 [1,1,1],
    #             ],
    #             order='C'
    #         )
    #     elif (isinstance(weights, np.ndarray) and
    #           weights.shape == (BuildConstants.WEIGHT_MATRIX_SIZE, BuildConstants.WEIGHT_MATRIX_SIZE)):
    #         _weights = weights
    #     else:
    #         raise ValueError("Weights size should be {}X{}".format(
    #             BuildConstants.WEIGHT_MATRIX_SIZE,
    #             BuildConstants.WEIGHT_MATRIX_SIZE
    #     ))
    #     metric = weighted_interpolated_error(20, bins=_bins, weights=_weights, error_type=error_type)
    #     self.MC_simulation.digest(metric=metric)
    #     self.metrics = self.MC_simulation.metrics_as_dict()
    #     return self.metrics

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

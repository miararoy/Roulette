import os
import json
import operator
import random
import pickle
from pprint import pprint
from time import time, sleep
import importlib.util
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from bliz.builder.data_prep import prepare_data_for_training
from bliz.builder.save_load_model import load_model
from bliz.builder.utils import min_max_norm
from bliz.evaluation import MonteCarloSimulation, weighted_interpolated_error
from bliz.logger import Logger

logger = Logger("builder").get_logger()


BUILD_DIR_NAME = "build"

class RegressionBuilder(object):

    logger = Logger("RegressionBuilder").get_logger()

    def __init__(
        self,
        path_to_model_folder: str,
        train_data: pd.core.frame.DataFrame,
        eval_data: pd.core.frame.DataFrame=None,
        sim_weights=None):

        self.train_data = train_data
        self.eval_data = eval_data
        if sim_weights is None:
            self.logger.warn("sim_weights is None, loading default value == [0.3, 0.3, 0.4]")
            self.sim_weights = [0.3, 0.3, 0.4]
        self.path_to_model_folder = path_to_model_folder

    def build(
        self, 
        n_experiments: int,
        target: str,
        index: str=None
    ):  
        self.target_col = target
        self.candidate_col = index
        self.logger.info("Initiating {} Epochs".format(n_experiments))
        Model = load_model(self.path_to_model_folder)
        self.MC_simulation = MonteCarloSimulation(self.sim_weights)
        with tqdm(total=n_experiments) as bar:
            i = 0
            while i < n_experiments:

                X, y, v_X, v_y = prepare_data_for_training(
                    df=self.train_data,
                    target=target,
                    index_column=index,
                    validation_test_size=random.uniform(0.1, 0.2),
                )

                this_model = Model()
                this_model.fit(X, y)
                this_prediction = this_model.predict(v_X)
                
                try:
                    lr = LinearRegression().fit(X,y)
                except BaseException as be:
                    self.logger.warn("could not perform linear regression model, maybe data is not numeric-only")
                    self.logger.warn(str(be))
                    lr = None
                if lr is None:
                    other_models = {}
                else:
                    other_models = {
                        "linear_regression":lr.predict(v_X)
                    }
                self.MC_simulation.load_experiment(
                    v_y,
                    y,
                    this_prediction,
                    others=other_models
                )
                bar.update(1)
                i+=1
    
    def evaluate(self, weights=None, bins=None, error_type='mse'):
        if bins is None:
            self.logger.warn("bins value is None using default == [0, 0.3, 0.7, 1]")
            _bins = [0, 0.3, 0.7, 1]
        if weights is None:
            self.logger.warn("weights value is None using default == [[1,1,1],[1,1,1],[1,1,1]]")
            _weights = np.asarray(
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1],
                ],
                order='C'
            )

        metric = weighted_interpolated_error([1.0]*20, bins=_bins, weights=_weights, error_type=error_type)
        self.MC_simulation.digest(metric=metric)            
        return self.MC_simulation.metrics_as_dict()

    def finalize_model(self,):
        X, y, _, _ = prepare_data_for_training(
            df=self.train_data,
            target=self.target_col,
            index_column=self.candidate_col,
            validation_test_size=0.0000001,
        )
        Model = load_model(self.path_to_model_folder)
        self.final_model = Model()
        self.final_model.fit(X, y)
        if self.eval_data is not None:
            try:
                eval_X, eval_y, _, _ = prepare_data_for_training(
                    df=self.eval_data,
                    target=self.target_col,
                    index_column=self.candidate_col,
                    validation_test_size=0.0000001,
                )
                self.eval_y = eval_y
                self.eval_y_pred = self.final_model.predict(eval_X)
                self.eval_plot, self.precision, self.recall ,self.n_precision, self.specificity = _create_evaluation_plot_with_rp(self.eval_y, self.eval_y_pred)

                data_for_auc = pd.DataFrame([self.eval_y, self.eval_y_pred], index=["y", "pred"]).T.sort_values(by="y")
                # fpr, tpr, _ = roc_curve(np.round(self.eval_y),  self.eval_y_pred)
                # self.auc = auc(fpr, tpr)

                self.auc = auc(data_for_auc.y, data_for_auc.pred)
                self.rp_dict = {"precision":self.precision, "recall": self.recall, "n_precision": self.n_precision, "specificity": self.specificity, "auc": self.auc}
                self.is_eval_predictions = True
                return self.rp_dict
            except BaseException as be:
                self.logger.error("could not predict on eval set!")
                self.logger.error(str(be))
                self.is_eval_predictions = False
                return None
        else:
            self.is_eval_predictions = False
            return None
    
    def get_evaluation_plot(self,):
        if self.eval_plot:
            return self.eval_plot
        else:
            self.logger.error("evaluation plot does not exist, if there is an evaluation set or use .finalize_model() method")
            return None

    def get_plot(self, title=None):
        return self.MC_simulation.plot(title=title)

    def save(self, path:str=None):
        if path is None:
            save_path = os.path.join(self.path_to_model_folder, BUILD_DIR_NAME)
        else:
            save_path = os.path.join(path, BUILD_DIR_NAME)
        model_dir = self.final_model.save(save_path)
        self.logger.info("saved model to {}".format(model_dir))
        _ = self.MC_simulation.get_metrics()
        if self.eval_data is not None:
            self.logger.info("saving eval input data")
            self.eval_data.to_csv(os.path.join(model_dir, "{}_eval_data.csv".format(self.final_model.model_name)))
        if self.is_eval_predictions:
            # plot, precision, recall = _create_evaluation_plot_with_rp(self.eval_y, self.eval_y_pred)
            self.eval_plot.savefig(os.path.join(model_dir, "{}_evaluation.png".format(self.final_model.model_name)))
            with open(os.path.join(model_dir, "{}_evaluation_metadata.json".format(self.final_model.model_name)), "w+") as rp_json:
                rp_json.write(
                    json.dumps(
                        self.rp_dict, 
                        sort_keys=True, 
                        indent=4))
            self.eval_plot.clf()
        self.logger.info("saving simultion plot")
        self.MC_simulation.plot(model_dir)
        self.logger.info("saving model metrics(model's kpis)")
        self.MC_simulation.metrics_to_json(os.path.join(model_dir, "{}_metadata.json".format(self.final_model.model_name)))
        self.logger.info("saving experiment summery")
        self.MC_simulation.save_experiment_summery(os.path.join(model_dir, "{}_summery.json".format(self.final_model.model_name)))
        self.logger.info("saving input data")
        self.train_data.to_csv(os.path.join(model_dir, "{}_data.csv".format(self.final_model.model_name)))
                
            
        

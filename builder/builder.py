from time import time, sleep
import importlib.util
import argparse
from pprint import pprint
import os
import json
import operator
import random
import pickle
import progressbar

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ehmodelbuilder.data_prep import prepare_data_for_training
from ehmodelbuilder.save_load_model import load_model
from ehmodelbuilder.utils import min_max_norm

from ehmodelevaluation import MonteCarloSimulation, weighted_interpolated_error
from ehetlextract.candidates import Candidates
from ehlogger.ehlogger import Logger


BUILD_DIR_NAME = "build"


def _create_evaluation_plot_with_rp(y: np.ndarray, y_pred: np.ndarray):
    _y_pred = min_max_norm(y_pred)
    logger = Logger("evaluation plot").get_logger()
    val_pred = pd.concat([pd.Series(y),pd.Series(_y_pred)],axis=1)
    val_pred.columns = ['test','pred']


    total_points = len(y)
    tp = val_pred[(val_pred.test>0.5)&(val_pred.pred>0.5)].shape[0]
    tn = val_pred[(val_pred.test<0.5)&(val_pred.pred<0.5)].shape[0]
    fn = val_pred[(val_pred.test>0.5)&(val_pred.pred<0.5)].shape[0]
    fp = val_pred[(val_pred.test<0.5)&(val_pred.pred>0.5)].shape[0]
    try:
        recall = round(tp/(tp+fn),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin recall")
        recall = -1.0
    try:
        precision = round(tp/(tp+fp),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin precision")
        precision = -1.0
    try:
        n_precision = round(tn/(tn+fn),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin precision")
        precision = -1.0
    try:
        specificity = round(tn/(tn+fp),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin specificity")
        precision = -1.0

    sns_plot = sns.regplot(
        y=_y_pred, 
        x=y,
        color="skyblue",
        fit_reg=False
    ).set_title(label="test set: recall = {rec}, precision = {pre}, specificity = {specificity}, n_precision = {n_pre}, N = {n}".format(
        rec=recall, pre=precision, specificity=specificity, n_pre=n_precision, n=total_points
    ),
        fontdict={'fontsize': 10}
    )

    axes = sns_plot.axes
    axes.set_ylim(0,1)
    axes.set_xlim(0,1)
    axes.set(xlabel='real', ylabel='prediction')

    axes.axhline(0.5, ls='--', c="r")
    axes.axvline(0.5, ls='--', c='r')
    plt.text(0.75, 0.75, round(tp/total_points,2), color='r')
    plt.text(0.2, 0.75, round(fp/total_points,2), color='r')
    plt.text(0.75, 0.2, round(fn/total_points,2), color='r')
    plt.text(0.2, 0.2, round(tn/total_points,2), color='r')
    return plt, precision, recall, n_precision, specificity



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
        target_col: str,
        candidate_col: str
    ):  
        self.target_col = target_col
        self.candidate_col = candidate_col
        self.logger.info("Initiating {} Epochs".format(n_experiments))
        Model = load_model(self.path_to_model_folder)
        self.MC_simulation = MonteCarloSimulation(self.sim_weights)
        with progressbar.ProgressBar(max_value=n_experiments) as bar:
            i = 0
            while i < n_experiments:

                X, y, v_X, v_y = prepare_data_for_training(
                    df=self.train_data,
                    kpi_column=target_col,
                    index_column=candidate_col,
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
                print(other_models)
                self.MC_simulation.load_experiment(
                    v_y,
                    y,
                    this_prediction,
                    others=other_models
                )
                bar.update(i)
                i+=1
    
    def evaluate(self, weights=None, bins=None, error_type='abs'):
        if bins is None:
            self.logger.warn("bins value is None using default == [0, 0.3, 0.7, 1]")
            _bins = [0, 0.3, 0.7, 1]
        if weights is None:
            self.logger.warn("weights value is None using default == [[1,3,5],[2,1,2],[3,2,1]]")
            _weights = np.asarray(
                [
                    [1,3,5],
                    [2,1,2],
                    [3,2,1],
                ],
                order='C'
            )

        metric = weighted_interpolated_error([1.0]*20, bins=_bins, weights=_weights, error_type=error_type)
        self.MC_simulation.digest(metric=metric)            
        return self.MC_simulation.metrics_as_dict()

    def finalize_model(self,):
        X, y, _, _ = prepare_data_for_training(
            df=self.train_data,
            kpi_column=self.target_col,
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
                    kpi_column=self.target_col,
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
                
            
        

import random
from time import time
from collections import namedtuple

import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import entropy

from bliz.evaluation.simulation_data import ExperimentData, Score
from bliz.evaluation.utils import validate_multiple_lists_length
from bliz.evaluation.constants import MetricsConstants
from bliz.evaluation.metrics import WD

random.seed(int(time()) % 10**4)


def length_error(data_length):
    raise ValueError(
        "all data should be of the same length - {}".format(data_length))


def _divergence_by_wd(dist, ref):
    """calculated the divergence between two distributions, meaning:
    the inverse of the distance between them, the more they are evenly
    distributed == divergence is higher

    Args:
        dist(array-like): original distribution
        ref(array-like): refrence distribution
    
    Returns:
        abs_wd
    """
    abs_wd = 1 / abs(WD(dist, ref))
    if abs_wd > 0.0:
        return abs_wd
    else:
        return np.inf


class Experiment(object):
    """
    gets monte carlo exp results and returns metrics
    """

    ExperimentData = namedtuple('ExperimentData', [
        "Real",
        'Model',
        'Rand',
        'Mean',
        'OtherModels',
    ])

    Score = namedtuple('Score', [
        'Model',
        'Rand',
        'Mean',
        'Div',
        'OtherModels',
    ])

    def __init__(
            self,
            real: list,
            real_trained: np.ndarray,
            model: list,
            rand_scale: int,
            other_models: dict = {},
    ):
        self.experiment_data = None
        self._load(
            real_results=real,
            real_trained_results=real_trained,
            model_prediction=model,
            random_scale=rand_scale,
            other_models_predictions=other_models)
        self.experiment_results = None

    def _load(
            self,
            real_results: list,
            real_trained_results: np.ndarray,
            model_prediction: list,
            random_scale,  # if not passed will generate a list in len(real_results) of [0,1) 
            other_models_predictions: dict):
        """loads experiment data into ExperimentData object

        Args:
            real_results(list): real results of this experiments
            real_trained_results(np.ndarray): target vector of the trained data in this experiment
            model_prediction(list): predictions of the model 
            random_scale(int): what is the scale [0, X] from which random results would be selected
            other_models_predictions(dict): a dictionary of other models model_name->list_of_socres

        Raises:
            ValueError: if there is a mismatch in length of any of the arguments
        """
        if validate_multiple_lists_length(
                real_results,
                model_prediction,
                *other_models_predictions.values()):
            random_data = [
                random_scale * random.random()
                for i in range(len(real_results))
            ]
            mean_result = real_trained_results.mean()
            self.experiment_data = ExperimentData(
                real_results, model_prediction, random_data,
                [mean_result] * len(real_results), other_models_predictions
                or {})
        else:
            raise length_error(len(real_results))

    def score(self, metric) -> Score:
        """calculates the score of this model based on the metrics

        Args:
            metric(callable): which metric should calcultae the error bw 2 results sets

        Returns:
            score(Score): score object with all the metric calculated scores
        """
        divergence = _divergence_by_wd(self.experiment_data.Model,
                                       self.experiment_data.Real)
        self.experiment_results = Score(
            metric(self.experiment_data.Real, self.experiment_data.Model),
            metric(self.experiment_data.Real, self.experiment_data.Rand),
            metric(self.experiment_data.Real, self.experiment_data.Mean),
            divergence, {
                k: metric(self.experiment_data.Real, v)
                for k, v in self.experiment_data.OtherModels.items()
            })
        return self.experiment_results

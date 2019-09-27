import random
from time import time
from collections import namedtuple

import numpy as np

from bliz.evaluation.simulation_data import ExperimentData, Score
from bliz.evaluation.utils import validate_multiple_lists_length
from bliz.evaluation.metrics import WD
from bliz.evaluation.constants import ExperimentConstants

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


def reg_mean(y, size):
    return np.full(size, np.asarray(y).mean())


def binary_mean(y, size):
    return np.full(size, np.bincount(y).argmax())


def choice_rand(y, size):
    return np.random.choice(y, size=size)


BASE_DIST = {
    "reg": {
        "mean": reg_mean,
        "rand": choice_rand
    },
    "binary": {
        "mean": binary_mean,
        "rand": choice_rand
    }
}


class Experiment(object):
    """
    gets monte carlo exp results and returns metrics
    """

    ExperimentData = namedtuple('ExperimentData', [
        "Real",
        'Model',
        'Rand',
        'Mean',
        # 'OtherModels',
    ])

    Score = namedtuple('Score', [
        'Model',
        'Rand',
        'Mean',
    ])

    def __init__(
            self,
            exp_type: str,
            real: list,
            real_trained: np.ndarray,
            model: list,
    ):
        self.experiment_data = None
        if exp_type in ExperimentConstants.TYPES:
            self.exp_type = exp_type
        else:
            raise ValueError("exp_type must be one of: [{}]".format(
                ExperimentConstants.TYPES))
        self._load(
            real_results=real,
            real_trained_results=real_trained,
            model_prediction=model,
        )
        self.experiment_results = None

    def _load(
        self,
        real_results: list,
        real_trained_results: np.ndarray,
        model_prediction: list,
    ):
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
        ):
            size = len(real_results)
            random_data = BASE_DIST[self.exp_type]["rand"](
                real_trained_results, size)
            mean_data = BASE_DIST[self.exp_type]["mean"](
                real_trained_results, size)
            self.experiment_data = ExperimentData(
                real_results, model_prediction, random_data,
                mean_data)
        else:
            raise length_error(len(real_results))

    def score(self, metric) -> Score:
        """calculates the score of this model based on the metrics

        Args:
            metric(callable): which metric should calcultae the error bw 2 results sets

        Returns:
            score(Score): score object with all the metric calculated scores
        """
        self.experiment_results = Score(
            metric(self.experiment_data.Real, self.experiment_data.Model),
            metric(self.experiment_data.Real, self.experiment_data.Rand),
            metric(self.experiment_data.Real, self.experiment_data.Mean),
        )
        return self.experiment_results

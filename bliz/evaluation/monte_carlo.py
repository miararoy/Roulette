# import os
import json
import threading

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error

from bliz.evaluation.utils import close_enough, parse_ndarray_as_float_list
from bliz.evaluation.experiment import Experiment
from bliz.evaluation.simulation_data import Metrics
from bliz.evaluation.metrics import discriminability, divergency, certainty
from bliz.evaluation.plotting.hist import single_hist
from bliz.evaluation.plotting.result_data import ResultData


class MonteCarloSimulation(object):
    """facilitates the experiments conducted, and calculating the metrics
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if MonteCarloSimulation._instance is None:
            with MonteCarloSimulation._lock:
                if MonteCarloSimulation._instance is None:
                    MonteCarloSimulation._instance = super(
                        MonteCarloSimulation, cls).__new__(cls)
        return MonteCarloSimulation._instance

    def __init__(
            self,
            W: list = [0.33333, 0.33333, 0.33333],
    ):
        """initiates monte carlo simulation

        Args:
            W: weights for ratio between metrics
            bins: list of bin edges
        Raises:
            ValueError: if lists not of same length
        """
        self.experiments = []
        self.scores = {}
        self.metrics = None
        if len(W) == len(Metrics._fields):
            if close_enough(sum(W), 1.0, 4):
                self.W = W
            else:
                ValueError("W, weights should sum to 1.0 and not {}".format(
                    sum(W)))
        else:
            raise ValueError("W should be of len {}".format(
                len(Metrics._fields)))

    def load_experiment(self,
                        real: list,
                        real_trained: np.ndarray,
                        model: list,
                        rand=1,
                        others: dict = {}):
        """loading a single experiment to simulation

        Args:
            real(list): list of ground truth results of the test set
            real_trained(list): list of ground truth results of the training set
            model(list): list of subjected-model predictions
            rand(int): scale of random samples, ir R in (0, n), defaults to 1
            others(dict): dictionary of other models predictions.
        """
        self.experiments.append(
            Experiment(real, real_trained, model, rand, others))

    def digest(self, metric):
        """calculates the full simulation results on the experiments
        loaded thus far

        Args:
            metric(callable): the metric to calculate results on (array-like, array-like) -> float
                              defaults to sklearn.metrics.mean_squared_error
        """
        _scores = []
        for exp in self.experiments:
            _scores.append(exp.score(metric))
        self.scores["model"] = []
        self.scores["rand"] = []
        self.scores["mean"] = []
        self.scores["div"] = []
        for s in _scores:
            self.scores["model"].append(s.Model)
            self.scores["rand"].append(s.Rand)
            self.scores["mean"].append(s.Mean)
            self.scores["div"].append(s.Div)
            if s.OtherModels:
                for k in s.OtherModels:
                    self.scores[k] = [s.OtherModels[k] for s in _scores]
        self.metrics = Metrics(
            discriminability(self.scores["model"], self.scores["mean"],
                             self.scores["rand"]),
            certainty(self.scores["model"], self.scores["rand"]),
            divergency(self.scores['div']))
        self.scores.pop('div', None)

    def get_metrics(self):
        """returns the Metrics namedTuple

        Returns:
            metrics (Metrics)
        """
        return self.metrics

    def metrics_as_dict(self):
        """returns the Metrics as dict

        Returns:
            metrics (dict): dictionary of metrics
        """
        if self.metrics:
            return {
                "discriminability": self.metrics.Discriminability,
                "certainty": self.metrics.Certainty,
                "divergency": self.metrics.Divergency,
            }
        else:
            return None

    def metrics_to_json(
            self,
            path: str,
            # filename: str,
    ):
        """saves MC result metrics as .json file

        Args:
            path(str): path to save json file
            filename(str): filename to be used in saving
        """
        # with open(os.path.join(path, "{}.json".format(filename)), 'w+') as
        # output_file:
        with open(path, 'w+') as output_file:
            output_file.write(json.dumps(self.metrics_as_dict()))

    def save_experiment_summery(
            self,
            path: str,
    ):
        """saves a summery report of the experiments

        Args:
            path(str): path to save summry report json file to
        """
        experiment_summery = {}
        for i, exp in enumerate(self.experiments):
            experiment_id = "experiment_{}".format(i)
            experiment_summery[experiment_id] = {
                "real": parse_ndarray_as_float_list(exp.experiment_data.Real),
                "model": parse_ndarray_as_float_list(exp.experiment_data.Model),
                "mean": parse_ndarray_as_float_list(exp.experiment_data.Mean),
                "rand": parse_ndarray_as_float_list(exp.experiment_data.Rand),
                "others": {
                    k: parse_ndarray_as_float_list(v)
                    for k, v in exp.experiment_data.OtherModels.items()
                }
            }
        with open(path, 'w+') as output_file:
            output_file.write(json.dumps(experiment_summery))

    def plot(self, path=None, title=None):
        """plots simulation histograms

        Args:
            path(str): path to save plots to
        """
        try:
            max_scores = [max(v) for k, v in self.scores.items()]
            bins = np.linspace(0, min(1.0, max(max_scores)),
                               max(int(len(self.scores["model"]) / 10), 100))
            plots = [ResultData(k, v, None) for k, v in self.scores.items()]
            if path:
                single_hist(data=plots, bins=bins, path=path, title=title)
            else:
                return single_hist(data=plots, bins=bins, title=title)
        except Exception as e:
            raise e

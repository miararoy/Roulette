import json

import numpy as np

from bliz.evaluation.utils import parse_ndarray_as_float_list
from bliz.evaluation.experiment import Experiment
from bliz.evaluation.simulation_data import Metrics
from bliz.evaluation.metrics import discriminability, certainty
from bliz.evaluation.plotting.hist import single_hist
from bliz.evaluation.plotting.result_data import ResultData


class MonteCarloSimulation(object):
    """facilitates the experiments conducted, and calculating the metrics
    """

    def __init__(
            self,
            exp_type: str
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
        self.exp_type = exp_type

    def load_experiment(self,
                        real: list,
                        real_trained: np.ndarray,
                        model: list,
                        ):
        """loading a single experiment to simulation

        Args:
            real(list): list of ground truth results of the test set
            real_trained(list): list of ground truth results of the training set
            model(list): list of subjected-model predictions
            rand(int): scale of random samples, ir R in (0, n), defaults to 1
            others(dict): dictionary of other models predictions.
        """
        self.experiments.append(
            Experiment(self.exp_type, real, real_trained, model))

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
        for s in _scores:
            self.scores["model"].append(s.Model)
            self.scores["rand"].append(s.Rand)
            self.scores["mean"].append(s.Mean)
        self.metrics = Metrics(
            discriminability(self.scores["model"], self.scores["mean"],
                             self.scores["rand"]),
            certainty(self.scores["model"], self.scores["rand"]),
        )

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
            }
        else:
            return None

    def metrics_to_json(
            self,
            path: str,
    ):
        """saves MC result metrics as .json file

        Args:
            path(str): path to save json file
            filename(str): filename to be used in saving
        """
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

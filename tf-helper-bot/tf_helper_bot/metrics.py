import warnings
from typing import Tuple, Union

import numpy as np
# import tensorflow as tf
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


class Metric:
    name = "metric"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> Tuple[float, str]:
        """Calculate the metric from truth and prediction tensors

        Parameters
        ----------
        truth : numpy.ndarray
        pred : numpy.ndarray

        Returns
        -------
        Tuple[float, str]
            (metric value(to be minimized), formatted string)
        """
        raise NotImplementedError()


class FBeta(Metric):
    """FBeta for binary targets"""
    name = "fbeta"

    def __init__(self, step, beta=2, average="binary"):
        self.step = step
        self.beta = beta
        self.average = average

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> Tuple[float, str]:
        best_fbeta, best_thres = self.find_best_fbeta_threshold(
            truth, pred,
            step=self.step, beta=self.beta)
        return best_fbeta * -1, f"{best_fbeta:.4f} @ {best_thres:.2f}"

    def find_best_fbeta_threshold(self, truth, probs, beta=2, step=0.05):
        best, best_thres = 0, -1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            for thres in np.arange(step, 1, step):
                current = fbeta_score(
                    truth, (probs >= thres).astype("int8"),
                    beta=beta, average=self.average)
                if current > best:
                    best = current
                    best_thres = thres
        return best, best_thres


class AUC(Metric):
    """AUC for binary targets"""
    name = "auc"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> Tuple[float, str]:
        auc_score = roc_auc_score(
            truth.astype("int"), pred)
        return auc_score * -1, f"{auc_score * 100:.2f}"

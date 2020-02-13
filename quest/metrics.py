import numpy as np
import sklearn.metrics
from scipy.stats import spearmanr
from scipy.special import expit
from tf_helper_bot import Metric


class SpearmanCorr(Metric):
    name = "spearman"

    def __init__(self, add_sigmoid: bool = False):
        self.add_sigmoid = add_sigmoid

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        if self.add_sigmoid:
            pred = expit(pred)
        corrs = []
        for i in range(pred.shape[1]):
            if len(np.unique(truth[:, i])) == 1:
                continue
            corrs.append(
                spearmanr(
                    truth[:, i],
                    pred[:, i]
                ).correlation

            )
        score = np.mean(corrs)
        return score * -1, f"{score * 100:.2f}"

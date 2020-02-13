import copy

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler


def prevent_nan(pred):
    for i in range(pred.shape[1]):
        if len(np.unique(pred[:, i])) == 1:
            pred[0, i] = np.random.rand()
            pred[-1, i] = np.random.rand()
    return pred


def find_best_bins(y_true, y_pred):
    scaler = MinMaxScaler()
    y_pred = scaler.fit_transform(y_pred)
    y = np.copy(y_pred)
    list_of_bins = []
    for i in (range(y_pred.shape[1])):
        best_score = 0  # initilize score for the the column i
        best_bins = 1
        history_score = []
        for max_voters in range(2, 200):
            y[:, i] = np.round(
                y_pred[:, i] * max_voters
            ) / max_voters
            y[:, i] = prevent_nan(y[:, i:i+1])[:, 0]
            score = spearmanr(y_true[:, i], y[:, i]).correlation
            history_score.append(score)
            if score > best_score:
                best_score = score
                best_bins = max_voters
        list_of_bins.append(best_bins)
        y[:, i] = np.round(y_pred[:, i] * best_bins) / best_bins
    return np.mean([
        spearmanr(y_true[:, ind], y[:, ind]).correlation
        for ind in range(y.shape[1])
    ]), list_of_bins, scaler

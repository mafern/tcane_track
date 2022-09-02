"""Custom metrics for model performance evaluation."""

import numpy as np
import mahalanobis
from scipy.stats import multivariate_normal

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "24 August 2022"

def get_mv_normal(y_data):
    mu_u, mu_v, sigma_u, sigma_v, rho = (
        y_data[0],
        y_data[1],
        y_data[2],
        y_data[3],
        y_data[4],
    )
    cov = np.array(
        [[sigma_u ** 2, rho * sigma_v * sigma_u], [rho * sigma_v * sigma_u, sigma_v ** 2]])
    rv = multivariate_normal([mu_u, mu_v], cov)

    return rv


def compute_nll(y_data, onehot_data):
    nll_vec = np.zeros((y_data.shape[0],))

    for isample in np.arange(0, y_data.shape[0]):
        rv = get_mv_normal(y_data[isample, :])
        nll = -rv.logpdf([onehot_data[isample, 0], onehot_data[isample, 1]])
        nll_vec[isample] = nll

    return nll_vec


def get_errors(y_data, onehot_data):
    error_vec = np.zeros((y_data.shape[0], 2))
    for isample in np.arange(0, y_data.shape[0]):
        rv = get_mv_normal(y_data[isample, :])
        pred_x, pred_y = (rv.mean[0], rv.mean[1])
        error = [pred_x - onehot_data[isample, 0], pred_y - onehot_data[isample, 1]]
        error_vec[isample] = error

    rmse = np.sqrt(error_vec[:, 0] ** 2 + error_vec[:, 1] ** 2)
    rmse_cons = np.sqrt(onehot_data[:, 0] ** 2 + onehot_data[:, 1] ** 2)

    return rmse, rmse_cons


def compute_pit(y_data, onehot_data):
    bins = np.linspace(0, 1, 11)

    F = mahalanobis.compute_cdf(
        y_data[:, 0],
        y_data[:, 1],
        y_data[:, 2],
        y_data[:, 3],
        y_data[:, 4],
        onehot_data[:, 0],
        onehot_data[:, 1],
    )
    pit_hist = np.histogram(
        F,
        bins,
        weights=np.ones_like(F) / float(len(F)),
    )

    B = len(pit_hist[0])
    D = np.sqrt(1 / B * np.sum((pit_hist[0] - 1 / B) ** 2))
    EDp = np.sqrt((1. - 1 / B) / (onehot_data.shape[0] * B))

    return bins, pit_hist, D, EDp

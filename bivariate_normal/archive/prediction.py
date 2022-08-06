"""Prediction functions.

Functions
---------
params
percentile_value

"""
import numpy as np
import shash_tfp

__author__ = "Elizabeth Barnes and Randal J Barnes"
__version__ = "05 August 2022"


def params(x_inputs, model):
    """Funtion to make parameter predictions for the bivariate 
    normal distribution.

    Arguments
    ---------
    x_inputs : floats
        matrix of inputs to the network

    model : tensorflow model
        neural network for predictions

    Returns
    -------
    vector of predicted parameters

    """
    y_pred = model.predict(x_inputs)
    y_pred[:, 0]
    sigma_pred = y_pred[:, 1]

    
    return mu_pred, sigma_pred, gamma_pred, tau_pred


def percentile_value(mu_pred, sigma_pred, gamma_pred, tau_pred, percentile_frac=0.5):
    """Function to obtain percentile value of the shash distribution."""
    
    dist = shash_tfp.Shash(mu_pred, sigma_pred, gamma_pred, tau_pred)
    
    return dist.quantile(percentile_frac).numpy()

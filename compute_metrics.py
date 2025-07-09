"""Collate and save model metrics."""

import model_diagnostics
import pandas as pd
import numpy as np

__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "12 November 2022"


def compute_metrics(df_predictions):
    """
    Compute model metrics.

    Arguments
    ---------
    df_predictions : data frame of predictions from
        the trained neural network

    Return
    ------
    metric : dictionary

    """
    mean_error, mean_error_reduction = model_diagnostics.compute_average_errors(df_predictions)
    iqr_capture = model_diagnostics.compute_iqr_capture(df_predictions)
    bins, hist_shash, pit_D, EDp_shash = model_diagnostics.compute_pit(df_predictions)
    sign_test = model_diagnostics.compute_sign_test(df_predictions)

    # Write metrics dictionary and return
    metrics = {
        'pit_D': pit_D,
        'sign_test': sign_test,
        'iqr_capture': iqr_capture,
        'mean_error': mean_error,
        'mean_error_reduction': mean_error_reduction,
        }

    return metrics


def save_metrics(settings, exp_name, metric_filename, df_train, df_val, df_test, df_valtest):
    """
    Build and save a combined dataframe of model metrics for four subsets: test, val, train, and valtest.

        Arguments
        ---------
        settings : dict
            The parameters defining the current experiment.
        exp_name : str
            The name of the experiment.
        metric_filename : str
            The name of the created pickle file.
        df_train : numpy.ndarray
            The training split.
        df_val : numpy.ndarray
            The validation split.
        df_test : numpy.ndarray
            The test split.
        df_valtest : numpy.ndarray
            The union of the test and validation splits.
        Returns
        -------
        None.
    """

    # create the metrics dataframe
    d = {}
    for leadtime in np.unique(df_test['FHOUR']):
        d[leadtime] = {}
        d[leadtime]['uncertainty_type'] = settings["uncertainty_type"]
        d[leadtime]['rng_seed'] = settings['rng_seed']
        d[leadtime]['exp_name'] = exp_name + str(leadtime)
        d[leadtime]['basin_lead'] = exp_name[exp_name.rfind('_') + 1:]
        d[leadtime]['testing_years'] = settings["years_test"][0]

        # compute the metrics
        metrics_test = compute_metrics(df_test[df_test['FHOUR'] == leadtime])
        metrics_val = compute_metrics(df_val[df_val['FHOUR'] == leadtime])
        metrics_train = compute_metrics(df_train[df_train['FHOUR'] == leadtime])
        metrics_valtest = compute_metrics(df_valtest[df_valtest['FHOUR'] == leadtime])

        for k in metrics_test.keys():
            k_key = k + '_test'
            d[leadtime][k_key] = metrics_test[k]
        for k in metrics_val.keys():
            k_key = k + '_val'
            d[leadtime][k_key] = metrics_val[k]
        for k in metrics_train.keys():
            k_key = k + '_train'
            d[leadtime][k_key] = metrics_train[k]
        for k in metrics_valtest.keys():
            k_key = k + '_valtest'
            d[leadtime][k_key] = metrics_valtest[k]

    # save the dataframe
    df = pd.DataFrame(data=d).transpose()
    df.to_pickle(metric_filename)

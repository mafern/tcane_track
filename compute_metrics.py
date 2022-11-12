"""Collate and save model metrics."""

import model_diagnostics
import pandas as pd

__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "12 November 2022"


def compute_metrics(model, x_data, label_data):
    """Compute model metrics.

    Arguments
    ---------
    model : tensorflow model
        trained neural network for predictions

    x_data : numpy.ndarray
        array of observed predictors
        shape = [n_data, n_features].

    label_data : numpy.ndarray
        array of observed predictands
        shape = [n_data, 2].

    Return
    ------
    metric : dictionary

    """
    mean_error = model_diagnostics.compute_average_errors(model, x_data, label_data)
    iqr_capture = model_diagnostics.compute_iqr_capture(model, x_data, label_data)
    bins, hist_shash, pit_D, EDp_shash = model_diagnostics.compute_pit(model, x_data, label_data)

    # Write metrics dictionary and return
    metrics = {
        'pit_D': pit_D,
        'iqr_capture': iqr_capture,
        'mean_error': mean_error,
    }

    return metrics


def save_metrics(
        model,
        settings,
        exp_name,
        metric_filename,
        x_train,
        label_train,
        x_val,
        label_val,
        x_test,
        label_test,
        x_valtest,
        label_valtest
):
    # compute the metrics
    metrics_test = compute_metrics(model, x_test, label_test)
    metrics_val = compute_metrics(model, x_val, label_val)
    metrics_train = compute_metrics(model, x_train, label_train)
    metrics_valtest = compute_metrics(model, x_valtest, label_valtest)

    # create the metrics dataframe
    d = {}
    d['uncertainty_type'] = settings["uncertainty_type"]
    d['rng_seed'] = settings['rng_seed']
    d['exp_name'] = exp_name
    d['basin_lead'] = exp_name[exp_name.rfind('_') + 1:]
    d['testing_years'] = settings["years_test"]

    for k in metrics_test.keys():
        k_key = k + '_test'
        d[k_key] = metrics_test[k]
    for k in metrics_val.keys():
        k_key = k + '_val'
        d[k_key] = metrics_val[k]
    for k in metrics_train.keys():
        k_key = k + '_train'
        d[k_key] = metrics_train[k]
    for k in metrics_valtest.keys():
        k_key = k + '_valtest'
        d[k_key] = metrics_valtest[k]

    # save the dataframe
    df = pd.DataFrame(data=d, index=[0])
    df.to_pickle(metric_filename)

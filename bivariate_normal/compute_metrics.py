import numpy as np
# import silence_tensorflow.auto
# import tensorflow as tf
# import prediction
import model_diagnostics
import pandas as pd

def compute_metrics(model, x_data, onehot_data):
    # """Metrics for SHASH models."""
    #
    # mu_pred, sigma_pred, gamma_pred, tau_pred = prediction.params(x_data, model)
    # dist = shash_tfp.Shash(mu_pred, sigma_pred, gamma_pred, tau_pred)
    # shash_mean = dist.mean()
    # shash_med = dist.median()
    # shash_mode = dist.mode()
    #
    # mean_error, median_error, mode_error = model_diagnostics.compute_errors(
    #     onehot_data,
    #     shash_mean,
    #     shash_med,
    #     shash_mode
    # )
    # bins, hist_shash, pit_D, EDp_shash = model_diagnostics.compute_pit(
    #     onehot_data,
    #     x_data=x_data,
    #     model_shash=model
    # )
    # iqr_capture = model_diagnostics.compute_interquartile_capture(
    #     onehot_data,
    #     x_data=x_data,
    #     model_shash=model
    # )
    # iqr_error_spearman, iqr_error_pearson = model_diagnostics.compute_iqr_error_corr(
    #     onehot_data=onehot_data,
    #     pred_median=shash_med,
    #     x_data=x_data,
    #     model_shash=model,
    # )
    #
    # # by definition Consensus is a correction of zero
    # cons_error = np.mean(np.abs(0.0 - onehot_data[:, 0]))
    #
    # # write metrics dictionary and return
    # metrics = {
    #     'pit_D': pit_D,
    #     'iqr_capture': iqr_capture,
    #
    #     'iqr_error_spearman': iqr_error_spearman[0],
    #     'iqr_error_pearson': iqr_error_pearson[0],
    #     'iqr_error_spearman_p': iqr_error_spearman[1],
    #     'iqr_error_pearson_p': iqr_error_pearson[1],
    #
    #     'cons_error': cons_error,
    #     'mean_error': mean_error,
    #     'median_error': median_error,
    #     'mode_error': mode_error,
    #
    #     'mean_error_reduction': cons_error - mean_error,
    #     'median_error_reduction': cons_error - median_error,
    #     'mode_error_reduction': cons_error - mode_error,
    # }
    metrics = {}
    return metrics

def save_metrics(model, settings, exp_name, metric_filename, x_train, onehot_train, x_val, onehot_val, x_test, onehot_test, x_valtest, onehot_valtest):
    # compute the metrics
    metrics_test = compute_metrics(model, x_test, onehot_test)
    metrics_val = compute_metrics(model, x_val, onehot_val)
    metrics_train = compute_metrics(model, x_train, onehot_train)
    metrics_valtest = compute_metrics(model, x_valtest, onehot_valtest)

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

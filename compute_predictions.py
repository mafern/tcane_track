import numpy as np
# import silence_tensorflow.auto
# import tensorflow as tf
# import prediction, shash_tfp
import pandas as pd


# leadtime [hours] : intensification [knots]
RI_THRESH_DICT = {
    24: 30,
    48: 55,
    72: 65,
    96: None,
    120: None,
}

def save_predictions(model, settings, predictions_filename, df_data, x_data, y_data):

    # Make the predictions and compute the associated prediction metrics.
    mu_pred, sigma_pred, gamma_pred, tau_pred = prediction.params(x_data, model)
    dist = shash_tfp.Shash(mu_pred, sigma_pred, gamma_pred, tau_pred)

    if (ri_threshold := RI_THRESH_DICT[settings["leadtime"]]) is not None:
        theta = df_data["VMAX0"].to_numpy() + ri_threshold - df_data["VMXC"].to_numpy()
        shash_pr_ri = 1.0 - dist.cdf(theta)

        clim_pr_ri = np.empty((x_data.shape[0],))
        for j, t in enumerate(theta):
            clim_pr_ri[j] = sum(y_data > t) / y_data.shape[0]
    else:
        shash_pr_ri = np.full((x_data.shape[0],), np.nan)
        clim_pr_ri = np.full((x_data.shape[0],), np.nan)

    df_predictions = df_data.copy()
    df_predictions["shash_mu"] = mu_pred
    df_predictions["shash_sigma"] = sigma_pred
    df_predictions["shash_gamma"] = gamma_pred
    df_predictions["shash_tau"] = tau_pred

    df_predictions["shash_median"] = dist.median()
    df_predictions["shash_mean"] = dist.mean()
    df_predictions["shash_mode"] = dist.mode()
    df_predictions["shash_25p"] = dist.quantile(0.25)
    df_predictions["shash_75p"] = dist.quantile(0.75)
    df_predictions["shash_90p"] = dist.quantile(0.90)
    df_predictions["shash_pr_ri"] = shash_pr_ri
    df_predictions["clim_pr_ri"] = clim_pr_ri

    df_predictions["shash_error"] = dist.median().numpy() - y_data[:,0]
    df_predictions["cons_error"] = 0.0 - y_data
    df_predictions["shash_improvement"] = (
            df_predictions["cons_error"].abs() -
            df_predictions["shash_error"].abs()
    )

    df_predictions.to_csv(predictions_filename)

    return None

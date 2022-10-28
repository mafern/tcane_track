"""Plot the model.fit training history and other training and analysis metrics."""
import matplotlib.pyplot as plt
import numpy as np

#import prediction
#import shash_tfp

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "28 October 2022"


def plot_history(history, model_name):
    """Plot the model.fit training history and save the resulting figure.

    Creates a 2-by-2 block of subplots.  The four plots are:
        1 -- training and validations loss history.
        2 -- training and validations customMAE history.
        3 -- training and validations InterquartileCapture history.
        4 -- training and validations SignTest history.

    Arguments
    ---------
    history : tf.keras.callbacks.History
        The history must have at least the following eight items
        in the history.history.keys()
            "loss",
            "val_loss",
            "custom_mae",
            "val_custom_mae",
            "interquartile_capture",
            "val_interquartile_capture",
            "sign_test",
            "val_sign_test"

    model_name : str
        The resulting figure is saved to:
            "figures/model_diagnostics/" + model_name + ".png"

    Returns
    -------
    None

    """
    TRAIN_COLOR = "#7570b3"
    VALID_COLOR = "#e7298a"
    FIGSIZE = (14, 10)
    FONTSIZE = 12
    DPIFIG = 300.0

    best_epoch = np.argmin(history.history["val_loss"])

    plt.figure(figsize=FIGSIZE)

    # Plot the training and validations loss history.
    plt.subplot(2, 2, 1)
    plt.plot(
        history.history["loss"],
        "o",
        color=TRAIN_COLOR,
        markersize=3,
        label="train",
    )

    plt.plot(
        history.history["val_loss"],
        "o",
        color=VALID_COLOR,
        markersize=3,
        label="valid",
    )

    plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
    plt.title("Log-likelihood Loss Function")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FONTSIZE)

    # Plot the training and validations customMAE history.
    try:
        plt.subplot(2, 2, 2)
        plt.plot(
            history.history["custom_mae"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_custom_mae"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Mean |true - median|")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no mae metric, skipping plot')

    # Plot the training and validations InterquartileCapture history.
    try:
        plt.subplot(2, 2, 3)
        plt.plot(
            history.history["interquartile_capture"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_interquartile_capture"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Fraction Between 25 and 75 Percentile")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no interquartile_capture, skipping plot')

    # Plot the training and validations SignTest history.
    try:
        plt.subplot(2, 2, 4)
        plt.plot(
            history.history["sign_test"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_sign_test"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Fraction Above the Median")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no sign-test, skipping plot')

    # Draw and save the plot.
    plt.tight_layout()
    plt.savefig("figures/model_diagnostics/" + model_name + ".png", dpi=DPIFIG)
    # plt.show()


def compute_iqr(x_data=None, model_shash=None):
    mu, sigma, gamma, tau = prediction.params(x_data, model_shash)

    dist = shash_tfp.Shash(mu, sigma, gamma, tau)
    lower = dist.quantile(0.25)
    upper = dist.quantile(0.75)

    return lower, upper


def compute_interquartile_capture(onehot_data, x_data=None,
        model_shash=None):
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1] - bins[0]

    lower, upper = compute_iqr(
        x_data=x_data,
        model_shash=model_shash
    )
    iqr_capture = np.logical_and(onehot_data[:, 0] > lower, onehot_data[:, 0] < upper)

    return np.sum(iqr_capture.astype(int)) / np.shape(iqr_capture)[0]


def compute_pit(onehot_data, x_data=None, model_shash=None):
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1] - bins[0]

    mu, sigma, gamma, tau = prediction.params(x_data, model_shash)
    dist = shash_tfp.Shash(mu, sigma, gamma, tau)
    F = dist.cdf(onehot_data[:, 0])

    pit_hist = np.histogram(
        F,
        bins,
        weights=np.ones_like(F) / float(len(F)),
    )

    # pit metric from Bourdin et al. (2014) and Nipen and Stull (2011)
    # compute expected deviation of PIT for a perfect forecast
    B = len(pit_hist[0])
    D = np.sqrt(1 / B * np.sum((pit_hist[0] - 1 / B) ** 2))
    EDp = np.sqrt((1. - 1 / B) / (onehot_data.shape[0] * B))

    return bins, pit_hist, D, EDp


def compute_nll(onehot_data, model_shash=None, x_data=None):

    mu, sigma, gamma, tau = prediction.params(x_data, model_shash)

    dist = shash_tfp.Shash(mu, sigma, gamma, tau)
    nloglike = -dist.log_prob(onehot_data[:, 0])

    return nloglike


def compute_errors(onehot_data, pred_mean, pred_median, pred_mode):
    mean_error = np.mean(np.abs(pred_mean - onehot_data[:, 0]))
    median_error = np.mean(np.abs(pred_median - onehot_data[:, 0]))
    mode_error = np.mean(np.abs(pred_mode - onehot_data[:, 0]))

    return mean_error, median_error, mode_error


def compute_iqr_error_corr(onehot_data, pred_median=None,
        x_data=None, model_shash=None):
    from scipy import stats

    # compute IQR
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1] - bins[0]

    lower, upper = compute_iqr(
        x_data=x_data,
        model_shash=model_shash
    )

    iqr = upper - lower

    # compute median_errors
    median_errors = np.abs(pred_median - onehot_data[:, 0])

    # compute correlation between median error and IQR
    iqr_error_spearman = stats.spearmanr(iqr, median_errors)
    try:
        iqr_error_pearson = stats.pearsonr(iqr, median_errors)
    except:
        iqr_error_pearson = [np.nan, np.nan]

    return iqr_error_spearman, iqr_error_pearson

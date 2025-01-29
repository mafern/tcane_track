"""Produce various model diagnostics, including plots and metrics.

Functions
---------
plot_history(history, model_name)
compute_average_errors(model, x_data, label_data)
compute_iqr_capture(model, x_data, label_data)
compute_pit(model, x_data, label_data)
compute_sign_test(model, x_data, label_data)

"""
import matplotlib.pyplot as plt
import numpy as np
import os

import mahalanobis

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "12 November 2022"


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
    plt.savefig(os.path.dirname(__file__)+"/figures/model_diagnostics/" + model_name + ".png", dpi=DPIFIG)
    plt.close()
    # plt.show()


def compute_average_errors(model, x_data, label_data):
    """Compute the average Euclidean distance between the conditional means
    and the labels.

    Arguments
    ---------
    model : tensorflow model
        trained neural network for predictions

    x_data : numpy.ndarray
        array of observed predictors
        shape = [n_data, n_features].

    label_data : numpy.ndarray
        array of observed predictands
        shape = [n_data,].

    Return
    ------
    mean_error : float
        average Euclidean distance between the conditional means
        and the labels.

    Note
    ----
    * The Euclidean distance does not account for the elliptical
    scaling of the x and y components. It is a crude measure.
    """
    y_pred = model.predict(x_data)
    mean_error = np.mean(
        np.hypot(
            y_pred[:, 0] - label_data[:, 0],
            y_pred[:, 1] - label_data[:, 1],
            )
        )

    mean_official_error = np.mean(
        np.hypot(
            label_data[:, 0],
            label_data[:, 1],
            )
        )

    mean_error_reduction = mean_official_error - mean_error

    return mean_error, mean_error_reduction


def compute_iqr_capture(model, x_data, label_data):
    """Compute the interquartile capture using the Mahalanobis distance.

    Computes the fraction of label_data that fall between the Mahalanobis
    ellipse that captures 25% of the bivariate probability and the Mahalanobis
    ellispe that captures 75% of the bivariate probability.

    Arguments
    ---------
    model : tensorflow model
        trained neural network for predictions

    x_data : numpy.ndarray
        array of observed predictors
        shape = [n_data, n_features].

    label_data : numpy.ndarray
        array of observed predictands
        shape = [n_data,].

    Return
    ------
    iqr_capture : float

    """
    y_pred = model.predict(x_data)
    cdf = mahalanobis.compute_cdf(
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
        y_pred[:, 4],
        label_data[:, 0],
        label_data[:, 1],
    )
    iqr_capture = np.logical_and(cdf > 0.25, cdf < 0.75)

    return np.mean(iqr_capture.astype(int))


def compute_pit(model, x_data, label_data):
    """Compute the PIT histogram using the Mahalanobis cdf."""
    bins = np.linspace(0, 1, 11)

    y_pred = model.predict(x_data)
    F = mahalanobis.compute_cdf(
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
        y_pred[:, 4],
        label_data[:, 0],
        label_data[:, 1],
    )
    pit_hist = np.histogram(
        F,
        bins,
        weights=np.ones_like(F) / float(len(F)),
    )

    # Pit metric from Bourdin et al. (2014) and Nipen and Stull (2011)
    # compute expected deviation of PIT for a perfect forecast
    B = len(pit_hist[0])
    D = np.sqrt(1 / B * np.sum((pit_hist[0] - 1 / B) ** 2))
    EDp = np.sqrt((1.0 - 1 / B) / (label_data.shape[0] * B))

    return bins, pit_hist, D, EDp


def compute_sign_test(model, x_data, label_data):
    """Compute the fraction of label values falling outside of
    the 0.50 Mahalanobis ellipse.

    """
    y_pred = model.predict(x_data)
    cdf = mahalanobis.compute_cdf(
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
        y_pred[:, 4],
        label_data[:, 0],
        label_data[:, 1],
    )
    outside = (cdf > 0.50)

    return np.mean(outside.numpy().astype(int))

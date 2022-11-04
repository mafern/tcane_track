"""Plot the model.fit training history and other training and analysis metrics."""
import matplotlib.pyplot as plt
import numpy as np

import mahalanobis

#import prediction

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "01 November 2022"


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
    plt.close()
    # plt.show()


def compute_interquartile_capture(onehot_data, x_data, model):
    """Compute the interquartile capture using the Mahalanobis distance.

    Computes the fraction of onehot_data that fall between the Mahalanobis
    ellipse that captures 25% of the bivariate probability and the Mahalanobis
    ellispe that captures 75% of the bivariate probability.
    """
    y_pred = model.predict(x_data)
    U = (onehot_data[:, 0] - y_pred[:, 0]) / y_pred[:, 2]
    V = (onehot_data[:, 1] - y_pred[:, 1]) / y_pred[:, 3]
    rho = y_pred[:, 4]
    r_sqr = 1.0 / (1.0 - rho * rho) * (U * U - 2 * rho * U * V + V * V)

    r_sqr_25 = -2.0 * np.log(1.0 - 0.25)
    r_sqr_75 = -2.0 * np.log(1.0 - 0.75)
    iqr_capture = np.logical_and(r_sqr > r_sqr_25, r_sqr < r_sqr_75)

    return np.sum(iqr_capture.astype(int)) / np.shape(iqr_capture)[0]


def compute_pit(onehot_data, x_data, model):
    """Compute the PIT histogram using the Mahalanobis distance."""
    bins = np.linspace(0, 1, 11)

    y_pred = model.predict(x_data)
    F = mahalanobis.compute_cdf(
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
        y_pred[:, 4],
        onehot_data[:, 0],
        onehot_data[:, 1],
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
    EDp = np.sqrt((1.0 - 1 / B) / (onehot_data.shape[0] * B))

    return bins, pit_hist, D, EDp

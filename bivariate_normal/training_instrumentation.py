"""Custom real-time training instrumentation."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "24 August 2022"


class TrainingInstrumentation(tf.keras.callbacks.Callback):
    """Plot real-time training instrumentation panel.

    If x_data and onehot_data are not given, the instrumentation panel
    includes only the real-time plot of the training and validation loss.

    If the x_data and onehot_data are given, the instrumentation panel also
    includes histogram plots for each of the local conditional distribution
    parameters, updated in real time.

    Parameters
    ----------
    x_data : tensor, default=None
        The x_train (or x_valid) tensor.  If either x_data or onehot_data
        is specified, then both must be specified, and they must have the
        same number of rows.

    onehot_data : tensor, default=None
        The onehot_train (or onehot_valid) tensor. If either x_data or
        onehot_data is specifed, then both must be specified, and they must
        have the same number of rows.

    figsize: (float, float), default=(13, 7)
        Size of the instrumentation panel.

    interval: int, default=1
        Number of epochs (steps) between refreshing the instruments.  By
        default, interval=1, and the instruments are updated ever epoch.

    Notes
    -----
    * This Class is explicitly designed for the bivariate normal
        distribution, with parameter names 'mu_u', 'mu_v', 'sigma_u',
        'sigma_v', and 'rho'.

    * Usage: include TrainingInstrumentation() as a callback in model.fit; e.g.
        training_callback = TrainingInstrumentation(
            x_train_std, onehot_train, interval=10
        )
        ...
        history = model.fit(
            ...
            callbacks=[training_callback],
        )

    """

    def __init__(
            self,
            x_data=None,
            onehot_data=None,
            figsize=(13, 7),
            interval=1,
    ):
        super().__init__()
        self.x_data = x_data
        self.onehot_data = onehot_data
        self.figsize = figsize
        self.interval = interval
        self.loss = []
        self.val_loss = []

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        if logs is None:
            logs = {}

        if epoch % self.interval == 0:
            clear_output(wait=True)
            plt.figure(figsize=self.figsize)

            best_epoch = np.argmin(self.val_loss)
            plt.subplot(3, 2, 1)
            plt.plot(self.loss, "o", color="#7570b3", label="train", markersize=2)
            plt.plot(self.val_loss, "o", color="#e7298a", label="valid", markersize=2)
            plt.axvline(x=best_epoch, linestyle="--", color="gray")
            plt.title(f"Loss After {epoch} Epochs")
            plt.grid(True)
            plt.legend(
                [
                    f"train = {logs.get('loss'):.3f}",
                    f"valid = {logs.get('val_loss'):.3f}",
                ]
            )

            if (self.x_data is not None) and (self.onehot_data is not None):
                preds = self.model.predict(self.x_data)

                mu_u = preds[:, 0]
                plt.subplot(3, 2, 3)
                plt.hist(mu_u, bins=30, color="#7fc97f", edgecolor="k")
                plt.legend(["mu_u"])

                mu_v = preds[:, 1]
                plt.subplot(3, 2, 4)
                plt.hist(mu_v, bins=30, color="#beaed4", edgecolor="k")
                plt.legend(["mu_v"])

                sigma_u = preds[:, 2]
                plt.subplot(3, 2, 5)
                plt.hist(sigma_u, bins=30, color="#fdc086", edgecolor="k")
                plt.legend(["sigma_u"])

                sigma_v = preds[:, 3]
                plt.subplot(3, 2, 6)
                plt.hist(sigma_v, bins=30, color="#fdc086", edgecolor="k")
                plt.legend(["sigma_v"])

                rho = preds[:, 4]
                plt.subplot(3, 2, 2)
                plt.hist(rho, bins=30, color="#ffff99", edgecolor="k")
                plt.legend(["rho"])

            plt.show()

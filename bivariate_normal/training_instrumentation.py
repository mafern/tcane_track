"""Custom real-time training instrumentation."""
from IPython.display import clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "28 October 2022"

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150


class TrainingInstrumentation(tf.keras.callbacks.Callback):
    """Plot real-time training instrumentation panel.

    Parameters
    ----------
    verbose: int, dafault=1
        0 = no output
        1 = plot at the end of training only (default)
        2 = update text each interval + plot at the end of training
        3 = plot each interval

    interval: int, default=1
        Number of epochs (steps) between refreshing the instruments.  By
        default, interval=1 and the instruments are updated every epoch.

    figsize: (float, float), default=(13, 7)
        Size of the instrumentation panel.

    Notes
    -----
    * Usage: include TrainingInstrumentation() as a callback in model.fit; e.g.
        training_callback = TrainingInstrumentation(
            x_train, y_train, interval=10
        )
        ...
        history = model.fit(
            ...
            callbacks=[training_callback],
        )

    """

    def __init__(
        self,
        verbose=1,
        interval=None,
        figsize=(13, 7),
    ):
        super().__init__()
        self.interval = interval
        self.verbose = verbose
        self.figsize = figsize
        self.loss = []
        self.val_loss = []

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        if (
            (self.interval is not None)
            and (epoch != 0)
            and (epoch % self.interval == 0)
        ):
            if self.verbose == 2:
                print(
                    f"{epoch = }, loss = {self.loss[-1]:.5f}, val_loss = {self.val_loss[-1]:.5f}"
                )

            elif self.verbose > 2:
                self.make_plot(epoch, logs)

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            epoch = len(self.val_loss)
            self.make_plot(epoch, logs)

    def make_plot(self, epoch, logs):
        clear_output(wait=True)
        plt.figure(figsize=self.figsize)

        # Eliminate the first two loss values, since they tend
        # to screw up the auto-scaling of the vertical axis.
        loss = self.loss.copy()
        loss[0:2] = [np.nan, np.nan]
        val_loss = self.val_loss.copy()
        val_loss[0:2] = [np.nan, np.nan]

        # Plot the taining loss and validation loss.
        best_epoch = np.argmin(self.val_loss)
        plt.plot(loss, "o", color="#7570b3", label="train", markersize=2)
        plt.plot(val_loss, "o", color="#e7298a", label="valid", markersize=2)
        plt.axvline(x=best_epoch, linestyle="--", color="gray")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss After {epoch} Epochs")
        plt.grid(True)
        plt.legend(
            [
                f"train = {logs.get('loss'):.3f}",
                f"valid = {logs.get('val_loss'):.3f}",
            ]
        )

        plt.show()

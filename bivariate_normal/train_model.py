"""Train the bivariate normal model.

Functions
---------
train_model(model, x_train, onehot_train, x_val, onehot_val, settings, verbose, interval)

"""
import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from training_instrumentation import TrainingInstrumentation

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Mark DeMaria"
__version__ = "28 October 2022"

def train_model(model, x_train, onehot_train, x_val, onehot_val, settings, verbose, interval):
    """Train the bivariate normal model.

    Arguments
    ---------
    model : tensorflow.keras.models.Model

    x_train : tensorflow.Tensor
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : tensorflow.Tensor
        The training split of the y data. The first column holds ODBX,
        the second column holds ODBY. The remaining three columns are
        filled with zeros.
        shape = [n_train, 5].

    x_val : tensorflow.Tensor
        The validation split of the x data.
        shape = [n_valid, n_features].

    onehot_val : tensorflow.Tensor
        The training split of the y data. The first column holds ODBX,
        the second column holds ODBY. The remaining three columns are
        filled with zeros.
        shape = [n_train, 5].

    settings : dict
        Dictionary of experiment settings for the run.

    verbose: int, dafault=0
        0 = no output
        1 = plot at the end of training only
        2 = update text each interval + plot at the end of training
        3 = plot each interval

    interval: int, default=1
        Number of epochs (steps) between refreshing the instruments.  By
        default, interval=1, and the instruments are updated ever epoch.
    """
    earlystoping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=settings["patience"],
        restore_best_weights=True,
        verbose=1,
    )

    training_callback = TrainingInstrumentation(
        verbose=verbose,
        interval=interval,
    )

    callbacks = [
        earlystoping_callback,
        training_callback,
    ]

    # train the network
    start_time = time.time()
    history = model.fit(
        x_train,
        onehot_train,
        validation_data=(x_val, onehot_val),
        batch_size=settings["batch_size"],
        epochs=settings["n_epochs"],
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )
    stop_time = time.time()

    # Display the results, and save the model rum.
    best_epoch = np.argmin(history.history["val_loss"])
    fit_summary = {
        "elapsed_time": stop_time - start_time,
        "best_epoch": best_epoch,
        "loss_train": history.history["loss"][best_epoch],
        "loss_valid": history.history["val_loss"][best_epoch],
    }

    return model, fit_summary

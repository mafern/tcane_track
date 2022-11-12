"""Train the bivariate normal model.

Functions
---------
train_model(model, x_train, label_train, x_val, label_val, settings)

"""
import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Mark DeMaria"
__version__ = "12 November 2022"


def train_model(model, x_train, label_train, x_val, label_val, settings):
    """Train the bivariate normal model.

    Arguments
    ---------
    model : tensorflow.keras.models.Model

    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    label_train : numpy.ndarray
        The training split of the predictands.
        shape = [n_train, 2].

    x_val : numpy.ndarray
        The validation split of the x data.
        shape = [n_val, n_features].

    label_val : numpy.ndarray
        The validation split of the predictands.
        shape = [n_val, 2].

    settings : dict
        Dictionary of experiment settings for the run.

    """
    earlystoping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=settings["patience"],
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [
        earlystoping_callback,
    ]

    # train the network
    start_time = time.time()
    history = model.fit(
        x_train,
        label_train,
        validation_data=(x_val, label_val),
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

    return model, fit_summary, history

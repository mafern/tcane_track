"""Run the defined set of experiments

"""
import os
import random

import numpy as np
from pprint import pprint
import silence_tensorflow.auto
import tensorflow as tf

from build_data import build_data
import build_model
import compute_metrics
import compute_predictions
import experiment_settings
import model_diagnostics
from save_model_run import save_model_run
from save_transfer_blueprint import save_transfer_blueprint
from train_model import train_model

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Mark DeMaria"
__version__ = "30 October 2022"


def train_experiments(
    exp_name_list,
    data_path,
    model_path,
    metrics_path,
    predictions_path,
    overwrite_model=False,
):

    """Loop through the defined experiments."""
    for exp_name in exp_name_list:
        settings = experiment_settings.get_settings(exp_name)

        # Set testing years based on the specified test conditions.
        if settings["test_condition"] == "leave-one-out":
            testing_years_list = np.arange(2013, 2022)
        elif settings["test_condition"] == "years":
            testing_years_list = np.copy(settings["years_test"])
        else:
            raise NotImplementedError("no such testing condition")

        # Loop through the testing years.
        for testing_years in testing_years_list:
            settings["years_test"] = (testing_years,)

            # Loop through the random seeds.
            for rng_seed in settings["rng_seed_list"]:
                settings["rng_seed"] = rng_seed
                np.random.seed(rng_seed)
                random.seed(rng_seed)
                tf.random.set_seed(rng_seed)

                # Build the track data tensors for a bivariate normal model.
                # create valtest set
                (
                    data_summary,
                    x_train,
                    onehot_train,
                    x_val,
                    onehot_val,
                    x_test,
                    onehot_test,
                    x_valtest,
                    onehot_valtest,
                    df_train,
                    df_val,
                    df_test,
                    df_valtest,
                ) = build_data(data_path, settings, verbose=0)

                # Create the model name.
                model_name = (
                    exp_name
                    + "_"
                    + str(testing_years)
                    + "_"
                    + settings["uncertainty_type"]
                    + "_"
                    + f"rng_seed_{settings['rng_seed']}"
                )

                # Make, compile, and train the model.
                tf.keras.backend.clear_session()
                model = build_model.make_model(
                    settings,
                    x_train,
                    onehot_train,
                    model_compile=True,
                )

                # Check if the model exists and overwirte is off.
                model_savename = model_path + model_name + "_weights.h5"
                if os.path.exists(model_savename) and overwrite_model is False:
                    print(f"Saved {model_name} already exists. Skipping...")
                    continue
                else:
                    print(f"Training {model_name}")

                model, fit_summary, history = train_model(
                    model,
                    x_train,
                    onehot_train,
                    x_val,
                    onehot_val,
                    settings,
                )
                pprint(fit_summary, width=80)

                save_model_run(
                    data_summary,
                    fit_summary,
                    model,
                    model_path,
                    model_name,
                    settings,
                    __version__,
                )

                save_transfer_blueprint(
                    data_summary,
                    model,
                    model_path,
                    model_name,
                    settings,
                    x_test,
                )

                # Additional plots and metrics
                model_diagnostics.plot_history(history, model_name)

                # save metrics
                # metric_filename = metrics_path + model_name + '_metrics.pickle'
                # __ = compute_metrics.save_metrics(
                #     model, settings, exp_name, metric_filename, x_train, onehot_train, x_val, onehot_val, x_test, onehot_test, x_valtest,
                #     onehot_valtest
                #     )

                # # save predictions
                # predictions_filename = predictions_path + model_name + '_testing_predictions.csv'
                # __ = compute_predictions.save_predictions(
                #     model, settings, predictions_filename, df_test, x_test, onehot_test,
                #     )

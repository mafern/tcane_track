# DRIVE THE TRAIN
# Run the selected suite of predefined experiments

import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from train_experiments import train_experiments

DATA_PATH = "data/"
MODEL_PATH = "saved_models/"
METRICS_PATH = "saved_metrics/"
PREDICTIONS_PATH = "saved_predictions/"

tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Mark DeMaria"
__version__ = "4 November 2022"

# List of experiments to run
EXP_NAME_LIST = (
    # "bivariate_normal_000_EPCP24",

    # "centered_bivariate_normal_100_EPCP12",
    # "centered_bivariate_normal_101_EPCP24",
    # "centered_bivariate_normal_102_EPCP36",
    # "centered_bivariate_normal_103_EPCP48",
    # "centered_bivariate_normal_104_EPCP60",
    "centered_bivariate_normal_105_EPCP72",
    "centered_bivariate_normal_106_EPCP84",
    "centered_bivariate_normal_107_EPCP96",
    "centered_bivariate_normal_108_EPCP108",
    "centered_bivariate_normal_109_EPCP120",
)

if __name__ == "__main__":
    start_time = time.time()
    train_experiments(
        EXP_NAME_LIST,
        DATA_PATH,
        MODEL_PATH,
        METRICS_PATH,
        PREDICTIONS_PATH,
        overwrite_model=False,
        )
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

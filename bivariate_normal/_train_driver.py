# DRIVE THE TRAIN
# Run the selected suite of predefined experiments

import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from train_experiments import train_experiments

DATA_PATH = "../data/"
MODEL_PATH = "saved_models/"
METRICS_PATH = "saved_metrics/"
PREDICTIONS_PATH = "saved_predictions/"

tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Mark DeMaria"
__version__ = "28 October 2022"

# List of experiments to run
EXP_NAME_LIST = (
    "bivariate_normal_101_EPCP24",

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
        verbose=2,
        interval=200,
    )
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

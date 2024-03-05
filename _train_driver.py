# DRIVE THE TRAIN
# Run the selected suite of predefined experiments

import time
import numpy as np
import warnings
import silence_tensorflow.auto
import tensorflow as tf
import experiment_settings
from train_experiments import train_experiments
import os

tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Elizabeth A. Barnes, Randal J Barnes, Mark DeMaria, and Martin A. Fernandez"
__version__ = "5 March 2024"

testing = experiment_settings.Experiments()
print('index range and short names for experiments')
for expname in testing.get_exp_list_short:
    exp_inds = [index for index, exp_string in enumerate(testing.get_exp_list) if expname in exp_string]
    print(np.min(exp_inds), '-', np.max(exp_inds)+1, expname)

# pick the experiment
expname = str(input("which experiment to run?: "))
assert np.isin(expname, testing.get_exp_list_short), "chosen experiment does not yet exist"
# get the correct indices for these experiments
exp_inds = [index for index, exp_string in enumerate(testing.get_exp_list) if expname in exp_string]
EXP_NAME_VEC = testing.get_exp_list[np.min(exp_inds):np.max(exp_inds)+1]
print(EXP_NAME_VEC)

ow_model = input("overwrite existing models? (True, False): ").lower() == 'true'
ow_preds = input("overwrite existing predictions? (True, False): ").lower() == 'true'

DATA_PATH = "data/"
MODEL_PATH = os.path.join("saved_models", expname)+'/'
METRICS_PATH = os.path.join("saved_metrics/", expname)+'/'
PREDICTIONS_PATH = os.path.join("saved_predictions/", expname)+'/'

if not isinstance(EXP_NAME_VEC, list):
    EXP_NAME_VEC = [EXP_NAME_VEC]

if __name__ == "__main__":
    start_time = time.time()
    train_experiments(
        EXP_NAME_VEC,
        DATA_PATH,
        MODEL_PATH,
        METRICS_PATH,
        PREDICTIONS_PATH,
        overwrite_model=ow_model,
        overwrite_predictions=ow_preds,
        )
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

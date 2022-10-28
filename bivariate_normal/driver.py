import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from run_experiments import run_experiments

#==============================================================================

if __name__ == "__main__":
    EXP_NAME_LIST = ("bivariate_normal_101_EPCP24",)

    DATA_PATH = "../data/"
    MODEL_PATH = "saved_models/"

    tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    start_time = time.time()
    run_experiments(
        EXP_NAME_LIST,
        DATA_PATH,
        MODEL_PATH,
        overwrite_model=False,
        verbose=2,
        interval=10
    )
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

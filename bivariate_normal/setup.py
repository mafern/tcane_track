"""One time setup of internal project paths."""
import os.path

__author__ = "Randal J. Barnes and Elizabeth A. Barnes"
__date__   = "30 September 2022"

DIRECTORIES = [
    "figures",
    "figures/Model_diagnostics",
    "model_metrics",
    "saved_models",
    "saved_metrics",
    "saved_predictions",
]


def setup_directory(target):
    if not os.path.isdir(target):
        os.mkdir(target)
        print(f"directory {target} has been created.")
    else:
        print(f"directory {target} already exists.")


def setup_directories():
    for target in DIRECTORIES:
        setup_directory(target)


if __name__ == "__main__":
    setup_directories()

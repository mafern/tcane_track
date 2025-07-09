"""Build the split and scaled training and validation hurricane data
arrays for the bivariate normal model.

Functions
---------
build_data(data_path, settings, verbose=0)

"""
import copy
import numpy as np
import pandas as pd
import pprint

import toolbox

__author__ = "Elizabeth A. Barnes, Randal J Barnes, and Martin A. Fernandez"
__version__ = "5 March 2024"


def build_data(data_path, settings, verbose=0):
    """Build the training, validation, and testing tensors for the bivariate
    normal model.

    Arguments
    ---------
    data_path : str
        The input filepath, not including the file name.

    settings : dict
        The parameters defining the current experiment.

    verbose : int
        0 -> silent
        1 -> description only
        2 -> description and y statistics

    Returns
    -------
    data_summary : dict
        A descriptive dictionary of the data.

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

    x_test : numpy.ndarray
        The test split of the x data.
        shape = [n_test, n_features].

    label_test : numpy.ndarray
        The test split of the predictands.
        shape = [n_test, 2].

    x_valtest : numpy.ndarray
        The union of the test and validation splits of the x data.
        shape = [n_val+n_test, n_features].

    label_valtest : numpy.ndarray
        The union of the test and validation splits of the predictands.
        shape = [n_val+n_test, 2].

    df_train : pandas dataframe
        A pandas dataframe containing training records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the training
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In
        particular, the rows of df_train align with the rows of x_train
        and label_train.

    df_val : pandas dataframe
        A pandas dataframe containing validation records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the validation
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In particular,
        the rows of df_val align with the rows of x_val and label_val.

    df_test : pandas dataframe
        A pandas dataframe containing testing records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the testing
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In particular,
        the rows of df_test align with the rows of x_test and label_test.

    df_valtest : pandas dataframe
        A pandas dataframe containing union of the validation and testing
        records.  The dataframe contains all columns from the original
        file. However, the dataframe contains only rows from the union
        of the validation and testing data sets that satisfy the specified
        basin and leadtime requirements, and were not eliminated due to
        missing values. The dataframe has the shuffled order of the rows.
        In particular, the rows of df_valtest align with the rows of
        x_valtest and label_valtest.

    """
    if settings["uncertainty_type"] not in [
        "bivariate_normal",
        "centered_bivariate_normal",
    ]:
        raise NotImplementedError

    # Setup for the selected target.
    # PREDICTAND_X < 0 means that besttrack was 50 km west of the consensus.
    y_names = [
        "PREDICTAND_X",
        "PREDICTAND_Y",
    ]
    missing = -9999

    # Setup for the selected features
    if settings["x_names"] is None:
        x_names = [
            "VMXC",
            "NCT",
            "AVDX",  # AVDX = -50, then the GFS forecast was 50 km west of the consensus longitude.
            "AVDY",
            "EMDX",
            "EMDY",
            "EGDX",
            "EGDY",
            "HWDX",
            "HWDY",
            "SPDX",
            "SPDY",
            "LONC",  # in degrees EAST
            "LATC",
            "VMAX0",
            "DV12",
            "SSTN",
            "SHDC",
            "DTL",
            "FHOUR"
        ]
    else:
        x_names = settings["x_names"]

    # The predicted local conditional distribution parameters are:
    # [ mu_u, mu_v, sigma_u, sigma_v, rho ].
    n_parameters = 5

    # Get the data from the specified file and filter out the unwanted rows.
    datafile_path = data_path + settings["filename"]
    df_raw = pd.read_table(datafile_path, sep="\s+")

    # PREDICTAND_X : The distance east (km) of the best track position from the
    # NHC or CPHC official track forecast (best track - official).
    df_raw["PREDICTAND_X"] = df_raw[settings["predictand_x"]]

    # PREDICTAND_Y : The distance north (km) of the best track position from the
    # NHC or CPHC official track forecast (best track - official).
    df_raw["PREDICTAND_Y"] = df_raw[settings["predictand_y"]]

    df = df_raw[
        (df_raw["ATCFID"].str.contains(settings["basin"]))
        & (np.isin(df_raw["FHOUR"], np.arange(12, 120+12, 12)))
        ]

    # Drop missing values only when they occur in used data columns
    used_columns = x_names+[settings['predictand_x'], settings['predictand_y']]
    good_rows = np.where(np.sum(df[used_columns] == missing, axis=1) == 0)[0]
    df = df.iloc[good_rows]
    df = df.reset_index(drop=True)

    # Shuffle the rows in the df Dataframe, using the numpy rng.
    # rng = np.random.default_rng(settings['rng_seed'])
    df = df.sample(frac=1, random_state=settings["rng_seed"])
    df = df.reset_index(drop=True)

    # ---------------------------------
    # Training/Validation/Testing Split

    # Split out the testing data.
    if settings["test_condition"] is None:
        # These will be reset below.
        x_test = None
        y_test = None
        df_test = None
    else:
        years = settings["years_test"]
        if verbose != 0:
            print("years" + str(years) + " withheld for testing")
        index = df.index[df["YEAR"].isin(years)]
        df_test = df.iloc[index]
        x_test = df_test[x_names].to_numpy()
        y_test = np.squeeze(df_test[y_names].to_numpy())
        df_test = df_test.reset_index(drop=True)

        df = df.drop(index)
        df = df.reset_index(drop=True)

    # Check that there is data for training.
    if np.shape(df)[0] == 0:
        return (
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
            np.empty((0, 1)),
        )

    # Split out the validation data.
    if settings["val_condition"] == "random":
        if settings["n_val"] <= 1:
            n_val = int(settings["n_val"]*len(df))
            settings["n_val"] = n_val
        index = np.arange(0, settings["n_val"])
        if len(index) < 100:
            raise Warning("Are you sure you want n_val < 100?")
    elif settings["val_condition"] == "years":
        if verbose != 0:
            print("years" + str(settings["n_val"]) + " withheld for validation")
        index = df.index[df["YEAR"].isin(settings["n_val"])]
    else:
        raise NotImplementedError

    df_val = df.iloc[index]
    x_val = df_val[x_names].to_numpy()
    y_val = np.squeeze(df_val[y_names].to_numpy())
    df_val = df_val.reset_index(drop=True)

    df = df.drop(index)
    df = df.reset_index(drop=True)

    if settings["test_condition"] is None:
        x_test = copy.deepcopy(x_val)
        y_test = copy.deepcopy(y_val)
        df_test = df_val.copy()

    # Subsample training if desired.
    if settings["n_train"] == "max":
        df_train = df.copy()
    else:
        df_train = df.iloc[:settings["n_train"]]

    x_train = df_train[x_names].to_numpy()
    y_train = np.squeeze(df_train[y_names].to_numpy())
    df_train = df_train.reset_index(drop=True)

    # ------------------------------------------------------
    # Create 'label' y arrays.

    label_train = np.zeros((len(y_train), 2))
    label_val = np.zeros((len(y_val), 2))
    label_test = np.zeros((len(y_test), 2))

    label_train[:, 0:2] = y_train
    label_val[:, 0:2] = y_val
    label_test[:, 0:2] = y_test

    # Set the dtype of label for consistency.
    label_train = label_train.astype("float32")
    label_val = label_val.astype("float32")
    label_test = label_test.astype("float32")

    # Create combined valtest set.
    x_valtest = np.concatenate((x_val, x_test), axis=0)
    label_valtest = np.concatenate((label_val, label_test), axis=0)
    df_valtest = pd.concat([df_val, df_test])

    # Make a descriptive dictionary.
    data_summary = {
        "datafile_path": datafile_path,
        "x_train_shape": tuple(x_train.shape),
        "x_val_shape": tuple(x_val.shape),
        "x_test_shape": tuple(x_test.shape),
        "x_valtest_shape": tuple(x_valtest.shape),
        "label_train_shape": tuple(label_train.shape),
        "label_val_shape": tuple(label_val.shape),
        "label_test_shape": tuple(label_test.shape),
        "label_valtest_shape": tuple(label_valtest.shape),
        "x_names": x_names,
        "y_names": y_names,
    }

    # Report the results.
    if verbose >= 1:
        pprint.pprint(data_summary, width=80)

    if verbose >= 2:
        toolbox.print_summary_statistics(
            {
                "y_train (OFDX [km])": label_train[:, 0],
                "y_val   (OFDX [km])": label_val[:, 0],
                "y_test  (OFDX [km])": label_test[:, 0],
                },
            sigfigs=1,
        )
        toolbox.print_summary_statistics(
            {
                "y_train (OFDY [km])": label_train[:, 1],
                "y_val   (OFDY [km])": label_val[:, 1],
                "y_test  (OFDY [km])": label_test[:, 1],
                },
            sigfigs=1,
        )

    return (
        data_summary,
        x_train,
        label_train,
        x_val,
        label_val,
        x_test,
        label_test,
        x_valtest,
        label_valtest,
        df_train,
        df_val,
        df_test,
        df_valtest,
    )


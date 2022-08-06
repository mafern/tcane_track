"""Build the split and scaled training and validation hurricane data arrays.

Functions
---------
build_hurricane_data(data_path, settings, verbose=0)

"""
import pprint

import numpy as np
import pandas as pd
import copy
import toolbox

__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "05 August 2022"


def build_hurricane_data(data_path, settings, verbose=0):
    """Build the training, validation, and testing tensors
    for the bivariate normal model.

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

    onehot_train : numpy.ndarray
        The training split of the y data. The first column holds ODBX,
        the second column holds ODBY. The remaining three columns are
        filled with zeros.
        shape = [n_train, 5].

    x_val : numpy.ndarray
        The validation split of the x data.
        shape = [n_val, n_features].

    onehot_val : numpy.ndarray
        The validation split of the y data. The first column holds ODBX,
        the second column holds ODBY. The remaining three columns are
        filled with zeros.
        shape = [n_val, 5].

    x_test : numpy.ndarray
        The test split of the x data.
        shape = [n_val, n_features].

    onehot_test : numpy.ndarray
        The test split of the y data. The first column holds ODBX,
        the second column holds ODBY. The remaining three columns are
        filled with zeros.
        shape = [n_val, 5].

    x_valtest : numpy.ndarray
        The union of the test and validation splits of the x data.
        shape = [n_val+n_test, n_features].

    onehot_valtest : numpy.ndarray
        The union of the test and validation splits of the y data.
        The first column holds ODBX, the second column holds ODBY.
        The remaining three columns are filled with zeros.
        shape = [n_val+n_test, 5].

    df_train : pandas dataframe
        A pandas dataframe containing training records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the training
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In
        particular, the rows of df_train align with the rows of x_train
        and onehot_train.

    df_val : pandas dataframe
        A pandas dataframe containing validation records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the validation
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In particular,
        the rows of df_val align with the rows of x_val and onehot_val.

    df_test : pandas dataframe
        A pandas dataframe containing testing records.  The
        dataframe contains all columns from the original file.
        However, the dataframe contains only rows from the testing
        data set that satisfy the specified basin and leadtime
        requirements, and were not eliminated due to missing values.
        The dataframe has the shuffled order of the rows.  In particular,
        the rows of df_test align with the rows of x_test and onehot_test.

    df_valtest : pandas dataframe
        A pandas dataframe containing union of the validation and testing
        records.  The dataframe contains all columns from the original
        file. However, the dataframe contains only rows from the union
        of the validation and testing data sets that satisfy the specified
        basin and leadtime requirements, and were not eliminated due to
        missing values. The dataframe has the shuffled order of the rows.
        In particular, the rows of df_valtest align with the rows of
        x_valtest and onehot_valtest.

    Notes
    -----
    * No scaling or normalization is applied during data extraction.

    """
    if settings["uncertainty_type"] != 'bivariate_normal':
        raise NotImplementedError
    
    # Setup for the selected target.
    if settings["x_names"] is None:
        x_names = [
            "NCT",
            "VMAX0",
            "AVDX",
            "EMDX",
            "EGDX",
            "HWDX",
            "AVDY",
            "EMDY",
            "EGDY",
            "HWDY",
            "LONC",
            "LATC",
            "VMXC",
            "DV12",
            "SLAT",
            "SHDC",
            "SSTN",
            "DTL",
            # "DSDV",
            # "LGDV",
            # "HWDV",
            # "AVDV",
        ]        
    else:
        x_names = settings["x_names"]
        
    y_names = ["OBDX", "OBDY"]
    missing = -9999

    n_parameters = 5

    # Get the data from the specified file and filter out the unwanted rows.
    datafile_path = data_path + settings["filename"]
    df_raw = pd.read_table(datafile_path, sep="\s+")
    df_raw = df_raw.rename(columns={"Date": "year"})

    df = df_raw[
        (df_raw["ATCF"].str.contains(settings["basin"]))
        & (df_raw["ftime(hr)"] == settings["leadtime"])
    ]

    if missing is not None:
        df = df.drop(df.index[df[y_names[0]] == missing])
        df = df.drop(df.index[df[y_names[1]] == missing])

    # Shuffle the rows in the df Dataframe, using the numpy rng.
    # rng = np.random.default_rng(settings['rng_seed'])
    df = df.sample(frac=1, random_state=settings["rng_seed"])
    df = df.reset_index(drop=True)

    # ------------------------------------------------------
    # Training/Validation/Testing Split

    # Split out the validation data
    if settings["test_condition"] is None:
        pass
    else:
        years = settings["years_test"]
        if verbose != 0:
            print("years" + str(years) + " withheld for testing")
        index = df.index[df["year"].isin(years)]
        df_test = df.iloc[index]
        x_test = df_test[x_names].to_numpy()
        y_test = np.squeeze(df_test[y_names].to_numpy())
        df_test = df_test.reset_index(drop=True)

        df = df.drop(index)
        df = df.reset_index(drop=True)

    # Split out the validation data
    if settings["val_condition"] == "random":
        index = np.arange(0, settings["n_val"])
        if len(index) < 100:
            raise Warning("Are you sure you want n_val < 100?")

    elif settings["val_condition"] == "years":
        if verbose != 0:
            print("years" + str(settings["n_val"]) + " withheld for testing")
        index = df.index[df["year"].isin(settings["n_val"])]

    df_val = df.iloc[index]
    x_val = df_val[x_names].to_numpy()
    y_val = np.squeeze(df_val[y_names].to_numpy())
    df_val = df_val.reset_index(drop=True)

    df = df.drop(index)
    df = df.reset_index(drop=True)

    if settings["test_condition"] is None:
        df_test = df_val.copy()
        x_test = copy.deepcopy(x_val)
        y_test = copy.deepcopy(y_val)

    # Subsample training if desired
    if settings["n_train"] == "max":
        df_train = df.copy()
    else:
        df_train = df.iloc[: settings["n_train"]]

    x_train = df_train[x_names].to_numpy()
    y_train = np.squeeze(df_train[y_names].to_numpy())
    df_train = df_train.reset_index(drop=True)

    # ------------------------------------------------------
    # Create 'onehot' y arrays. The OBDX and OBDY values
    # go in the first two column. The remaining columns
    # are filled with zeros -- i.e. dummy columns.  These
    # dummy columns are required by TensorFlow; the number
    # of columns must equal the number of outputs.
    onehot_train = np.zeros((len(y_train), n_parameters))
    onehot_val = np.zeros((len(y_val), n_parameters))
    onehot_test = np.zeros((len(y_test), n_parameters))

    onehot_train[:, 0:2] = y_train
    onehot_val[:, 0:2] = y_val
    onehot_test[:, 0:2] = y_test

    # Set the dtype of onehot for consistency.
    onehot_train = onehot_train.astype("float32")
    onehot_val = onehot_val.astype("float32")
    onehot_test = onehot_test.astype("float32")

    # Create valtest set.
    x_valtest = np.concatenate((x_val, x_test), axis=0)
    onehot_valtest = np.concatenate((onehot_val, onehot_test), axis=0)
    df_valtest = pd.concat([df_val, df_test])

    # Make a descriptive dictionary.
    data_summary = {
        "datafile_path": datafile_path,
        "x_train_shape": tuple(x_train.shape),
        "x_val_shape": tuple(x_val.shape),
        "x_test_shape": tuple(x_test.shape),
        "x_valtest_shape": tuple(x_valtest.shape),
        "onehot_train_shape": tuple(onehot_train.shape),
        "onehot_val_shape": tuple(onehot_val.shape),
        "onehot_test_shape": tuple(onehot_test.shape),
        "onehot_valtest_shape": tuple(onehot_valtest.shape),
        "x_names": x_names,
        "y_names": y_names,
    }

    # Report the results.
    if verbose >= 1:
        pprint.pprint(data_summary, width=80)

    if verbose >= 2:
        toolbox.print_summary_statistics(
            {
                "y_train (OBDX [km])": onehot_train[:, 0],
                "y_val   (OBDX [km])": onehot_val[:, 0],
                "y_test  (OBDX [km])": onehot_test[:, 0],
            },
            sigfigs=1,
        )
        toolbox.print_summary_statistics(
            {
                "y_train (OBDY [km])": onehot_train[:, 1],
                "y_val   (OBDY [km])": onehot_val[:, 1],
                "y_test  (OBDY [km])": onehot_test[:, 1],
            },
            sigfigs=1,
        )

    return (
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
    )

# Create a new experiment

import numpy as np
import experiment_settings

exps = experiment_settings.Experiments()
print("\ncurrently defined experiments:", *exps.get_exp_list_short, "\n")

default_dict = {"expname": "", "basin": "", "leadtimes": 0, "x_names": [], "predictand": "OFD", "uncertainty": "centered_bivariate_normal", "hiddens": [5, 5], "dropout": [0., 0., 0.], "ridge": [0.0, 0.0], "learning": 0.0001, "batch": 64, "seed_list": [], "act_fun": "relu", "n_epochs": 25_000, "patience": 250, "test_condition": "leave-one-out", "years_test": [2023,], "val_condition": "random", "n_val": 200, "n_train": "max", "loss_function": "compute_bivariate_normal_nll", "metrics": {"CustomMAE": "custom_mae", "InterquartileCapture": "interquartile_capture", "SignTest": "sign_test"}}

description_dict = {
    # required options to set
    "expname": "required input \n  name of the experiment (how it will be saved and called)",
    "basin": "required input \n  ocean basin, either 'AL' (Atlantic) or 'EP' (Eastern/Central Pacific), 'both' to generate both basins",
    "leadtimes": "required input \n  multiples of 12 up to 120 hours, as a list, or 0 (not as a list) to generate separate experiments for each leadtime",
    "seed_list": "required input \n  seeds, as a list, to use for training/testing runs",
    # train/validate/test split options
    "test_condition": "default is 'leave-one-out' \n  'years' tests on the years in years_test, 'leave-one-out' runs through all years (overrides years_test), None does not test the network",
    "years_test": "default is [2023,] \n  which years to use for testings, passing None runs through all years",
    "val_condition": "default is 'random' \n  whether to use random samples for validation 'random', or a year 'years'",
    "n_val": "default is 200 \n  number of validation samples to use, if <= 1, uses a fraction of the training samples",
    "n_train": "default is 'max' \n  number of training samples to use, passing 'max' uses full remaining data set (after training, validation split)",
    # model options -- inputs, outputs, shash model
    "x_names": "default is ['VMXC', 'NCT', 'AVDX', 'AVDY', 'EMDX', 'EMDY', 'EGDX', 'EGDY', 'HWDX', 'HWDY', 'SPDX', 'SPDY', 'LONC', 'LATC', 'VMAX0', 'DV12', 'SLAT', 'SSTN', 'SHDC', 'DTL'] \n  features to use -- entering x_names that are already in default_x_names removes them, otherwise they are added",
    "predictand": "default is 'OFD' \n  the label, either 'OFD' (official forecast error) or 'OBD' (consensus error)",
    "uncertainty": "default is 'centered_bivariate_normal' \n  whether to center the bivariate normal ('centered_bivariate_normal') or allow it to fit ('bivariate_normal')",
    # architecture options
    "hiddens": "default is [5, 5] \n  number of nodes in each layer (length of list determines number of layers)",
    "dropout": "default is [0., 0., 0.] \n  fraction for dropout in each layer",
    "ridge": "default is [0.0, 0.0] \n  ridge parameter for each layer",
    "learning": "default is 0.0001 \n  learning rate, batch: batch size",
    "batch": "default is 64 \n  batch size for training",
    "act_fun": "default is 'relu' \n  activation function, other options available through tensorflow",
    "n_epochs": "default is 25_000 \n  maximum number of training epochs, patience: number of unimproved training epochs before returning best epoch",
    "patience": "default is 250 \n  number of unimproved training epochs before early stopping is triggered",
    "loss_function": "default is 'compute_bivariate_normal_nll' \n  the loss function to use. See custom_loss.py for options",
    "metrics": "default is {'CustomMAE': 'custom_mae', 'InterquartileCapture': 'interquartile_capture', 'SignTest': 'sign_test'} \n  metrics to use for validation, should be a dictionary of {function: name}. See custom_metrics.py for options"
}

assert_dict = {"basin": ["AL", "al", "EP", "ep", "both"], "leadtimes": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], "predictand": ["OFD", "OBD"], "uncertainty": ["centered_bivariate_normal", "bivariate_normal"], "test_condition": ["leave-one-out", "years"], "val_condition": ["random", "years"]}

architecture_keys = ["hiddens", "dropout", "ridge", "learning", "batch", "act_fun", "n_epochs", "patience", "loss_function", "metrics"]
model_keys = ["x_names", "predictand", "uncertainty"]
split_keys = ["test_condition", "years_test", "val_condition", "n_val", "n_train"]
basic_keys = ["expname", "basin", "leadtimes", "seed_list"]

for key in basic_keys:
    print('-----------------------------------------------------')
    print(key+': ', description_dict[key])
    if type(default_dict[key]) is str:
        default_dict[key] = input("choose "+key+": ")
    else:
        default_dict[key] = eval(input("choose "+key+": "))
    if key in assert_dict.keys():
        assert np.isin(default_dict[key], assert_dict[key]), "invalid input, use one of the following " + str(assert_dict[key])
    print('selected:', default_dict[key])

split = eval(input(f"\nadjust the data splitting {split_keys} settings? (True or False): "))
print("\n")
if split:
    for key in split_keys:
        print('-----------------------------------------------------')
        print(key+': ', description_dict[key])
        if type(default_dict[key]) is str:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = setting
        else:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = eval(setting)
        if key in assert_dict.keys():
            assert np.isin(default_dict[key], assert_dict[key]), "invalid input, use one of the following " + str(assert_dict[key])
        print('selected:', default_dict[key])

model = eval(input(f"\nadjust the model {model_keys} settings? (True or False): "))
print("\n")
if model:
    for key in model_keys:
        print('-----------------------------------------------------')
        print(key+': ', description_dict[key])
        if key == "x_names":
            print("available input variables:", *exps.avail_x_names)
        if type(default_dict[key]) is str:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = setting
        else:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = eval(setting)
        if key in assert_dict.keys():
            assert np.isin(default_dict[key], assert_dict[key]), "invalid input, use one of the following " + str(assert_dict[key])
        print('selected:', default_dict[key])

architecture = eval(input(f"\nadjust the network architecture {architecture_keys} settings? (True or False): "))
print("\n")
if architecture:
    for key in architecture_keys:
        print('-----------------------------------------------------')
        print(key+': ', description_dict[key])
        if type(default_dict[key]) is str:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = setting
        else:
            setting = input("choose "+key+": ")
            if setting != '':
                default_dict[key] = eval(setting)
        if key in assert_dict.keys():
            assert np.isin(default_dict[key], assert_dict[key]), "invalid input, use one of the following " + str(assert_dict[key])
        print('selected:', default_dict[key])

if default_dict['basin'] == 'both':
    new_exp_AL = exps.make_exp_dictionary(default_dict['expname'], 'AL', leadtimes=default_dict['leadtimes'], x_names=default_dict['x_names'], predictand=default_dict['predictand'], uncertainty=default_dict['uncertainty'], hiddens=default_dict['hiddens'], dropout=default_dict['dropout'], ridge=default_dict['ridge'], learning=default_dict['learning'], batch=default_dict['batch'], seed_list=default_dict['seed_list'], rng_seed=None, act_fun=default_dict['act_fun'], n_epochs=default_dict['n_epochs'], patience=default_dict['patience'], test_condition=default_dict['test_condition'], years_test=default_dict['years_test'], val_condition=default_dict['val_condition'], n_val=default_dict['n_val'], n_train=default_dict['n_train'], loss_function=default_dict['loss_function'], metrics=default_dict['metrics'])
    exps.add_experiment(new_exp_AL)

    new_exp_EP = exps.make_exp_dictionary(default_dict['expname'], 'EP', leadtimes=default_dict['leadtimes'], x_names=default_dict['x_names'], predictand=default_dict['predictand'], uncertainty=default_dict['uncertainty'], hiddens=default_dict['hiddens'], dropout=default_dict['dropout'], ridge=default_dict['ridge'], learning=default_dict['learning'], batch=default_dict['batch'], seed_list=default_dict['seed_list'], rng_seed=None, act_fun=default_dict['act_fun'], n_epochs=default_dict['n_epochs'], patience=default_dict['patience'], test_condition=default_dict['test_condition'], years_test=default_dict['years_test'], val_condition=default_dict['val_condition'], n_val=default_dict['n_val'], n_train=default_dict['n_train'], loss_function=default_dict['loss_function'], metrics=default_dict['metrics'])
    exps.add_experiment(new_exp_EP)

else:
    new_exp = exps.make_exp_dictionary(default_dict['expname'], default_dict['basin'].upper(), leadtimes=default_dict['leadtimes'], x_names=default_dict['x_names'], predictand=default_dict['predictand'], uncertainty=default_dict['uncertainty'], hiddens=default_dict['hiddens'], dropout=default_dict['dropout'], ridge=default_dict['ridge'], learning=default_dict['learning'], batch=default_dict['batch'], seed_list=default_dict['seed_list'], rng_seed=None, act_fun=default_dict['act_fun'], n_epochs=default_dict['n_epochs'], patience=default_dict['patience'], test_condition=default_dict['test_condition'], years_test=default_dict['years_test'], val_condition=default_dict['val_condition'], n_val=default_dict['n_val'], n_train=default_dict['n_train'], loss_function=default_dict['loss_function'], metrics=default_dict['metrics'])
    exps.add_experiment(new_exp)

print("experiment", default_dict['expname'], 'added')

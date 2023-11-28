"""Experimental settings class"""

import json
import numpy as np
import os
import sys
import train_experiments as texp


# helper to be consistent with existing code
def get_settings(experiment_name):
    experiments = Experiments().experiments
    return experiments[experiment_name]


class Experiments():
    """
    Class to manage experiments (settings), including generating new experiments and running experiments.
    """
    def __init__(self, filename="experiments.json", default_x_names=["VMXC", "NCT", "AVDX", "AVDY", "EMDX", "EMDY", "EGDX", "EGDY", "HWDX", "HWDY", "SPDX", "SPDY", "LONC", "LATC", "VMAX0", "DV12", "SLAT", "SSTN", "SHDC", "DTL"]):
        # load the dictionary and keys
        try:
            self.filename = filename
            self.default_x_names = default_x_names
            self.experiments = json.load(open(self.filename, 'r'))
            self.keys = list(self.experiments.keys())
        except ValueError:
            print("No experiment file with name: ", self.filename)

    def get_experiment(self, exp_name):
        "get the experiment dictionary entry with exp_name"
        return self.experiments[exp_name]

    def add_experiment(self, new_exp):
        "add an experiment (dictionary) or experiments to the save file"
        # add to the dictionary, resave
        experiments = self.experiments.copy()
        for exp_name in new_exp:
            experiments[exp_name] = new_exp[exp_name]
        with open(self.filename, "w") as exp_file:
            exp_file.write(json.dumps(experiments))
        # reload the experiments
        self.experiments = experiments
        self.keys = list(self.experiments.keys())

    def delete_experiment(self, exp_name):
        "delete an experiment from the main file, does not delete from any backups"
        self.experiments.pop(exp_name)
        with open(self.filename, "w") as exp_file:
            exp_file.write(json.dumps(self.experiments))
        # reload the exp name list
        self.keys = list(self.experiments.keys())

    def save_backup(self, filename="experiments-backup.json"):
        "save a backup file!"
        with open(filename, "w") as exp_file:
            exp_file.write(json.dumps(self.experiments))

    def restore_from_backup(self, backupfile, *, confirm=0):
        "Overwrite the main saved file with the dictionary in the backup file"
        if confirm == 1: # just to make sure
            backup = json.load(open(backupfile, 'r'))
            with open(self.filename, "w") as exp_file:
                exp_file.write(json.dumps(backup))
            self.experiments = json.load(open(self.filename, 'r'))
        else:
            print("must pass confirm=1 to overwrite main file with backup")

    def make_exp_dictionary(self, expname, basin, *, leadtimes=0, x_names=[], predictand="OFD", uncertainty="centered_bivariate_normal", hiddens=[5, 5], dropout=[0., 0., 0.], ridge=[0.0, 0.0], learning=0.0001, batch=64, seed_list=[123], rng_seed=None, act_fun="relu", n_epochs=25_000, patience=250, test_condition="leave-one-out", years_test=[2022,], val_condition="random", n_val=200, n_train="max", loss_function="compute_bivariate_normal_nll", metrics={"CustomMAE": "custom_mae", "InterquartileCapture": "interquartile_capture", "SignTest": "sign_test"}):
        """
        Make an experiment dictionary with the following settings:
                - expname: name of the experiment (how it will be saved and called)
                - basin: ocean basin, either 'AL' (Atlantic) or 'EP' (Eastern/Central Pacific)
                - leadtime: multiples of 12 up to 120 hours, or 0 to generate separate experiments for each leadtime
                - x_names: features to use -- entering x_names that are already in default_x_names removes them, otherwise they are added
                - predictand: the label, either 'OFD' (official forecast error) or 'OBD' (consensus error)
                - uncertainty: whether to center the bivariate normal ("centered_bivariate_normal") or allow it to fit ("bivariate_normal")
                - hiddens: number of nodes in each layer (length of list determines number of layers)
                - dropout: fraction for dropout in each layer
                - ridge: ridge parameter for each layer
                - learning: learning rate
                - batch: batch size
                - seed_list: seeds to use for training/testing runs
                - rng_seed: leave as None
                - act_fun: activation function, 'relu' as default. Other options available through tensorflow
                - n_epochs: maximum number of training epochs
                - patience: number of unimproved training epochs before returning best epoch
                - test_condition: 'years' tests on the years in years_test, 'leave-one-out' runs through all years (overrides years_test), None does not test the network
                - years_test: which years to use for testings, passing None runs through all years
                - val_condition: whether to use random samples for validation 'random', or a year 'years'
                - n_val: number of validation samples to use, if <= 1, uses a fraction of the training samples
                - n_train: number of training samples to use, passing 'max' uses full remaining data set (after training, validation split)
                - loss_function: the loss function to use. See custom_loss.py for options
                - metrics: metrics to use for validation, should be a dictionary of {function: name}. See custom_metrics.py for options
                """
        # if x_names entry already in default, remove it, else add it
        default_x_names = self.default_x_names.copy()
        for name in x_names:
            if name in default_x_names:
                print("Removing "+name+" from the input parameters")
                default_x_names.remove(name)
            else:
                default_x_names.append(name)
        x_names = default_x_names
        print("using features: ", x_names)

        if leadtimes == 0: leadtimes = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
        if not isinstance(leadtimes, list):
            leadtimes = [leadtimes]
        dictionary = {}
        for leadtime in list(leadtimes):
            # check if the proposed experiment name already exists
            assert expname+'_'+predictand+'_'+basin+str(leadtime) not in self.keys, "experiment with that name already exists"
            dictionary[expname+'_'+predictand+'_'+basin+str(leadtime)] = {"filename": "nnfit_vlist_03Nov2023.dat", "uncertainty_type": uncertainty, "leadtime": leadtime, "basin": basin, "hiddens": hiddens, "dropout_rate": dropout, "ridge_param": ridge, "learning_rate": learning, "batch_size": batch, "rng_seed_list": seed_list, "rng_seed": rng_seed, "act_fun": act_fun, "n_epochs": n_epochs, "patience": patience, "test_condition": test_condition, "years_test": years_test, "val_condition": val_condition, "n_val": n_val, "n_train": n_train, "x_names": x_names, "loss_function": loss_function, "predictand_x": predictand+'X', "predictand_y": predictand+'Y', "metrics": metrics,}
        return dictionary

    # function wrapping train_experiments
    def run_experiments(self, experiments, *, model_path='saved_models/', metric_path='saved_metrics/', data_path='data/', prediction_path='saved_predictions/', overwrite_predictions=False, overwrite_model=False):
        if not isinstance(experiments, list):
            experiments = [experiments]
        for expname in experiments:
            texp.train_experiments([expname], data_path, model_path, metric_path, prediction_path, overwrite_predictions=overwrite_predictions, overwrite_model=overwrite_model)

    @property
    def get_exp_list(self):
        "return a list of the experiment names"
        return self.keys

    @property
    def get_exp_list_short(self):
        "return a list of the experiment names"
        unique_exps, unique_inds = np.unique([self.keys[i][::-1].split('_', maxsplit=1)[1][::-1] for i in range(len(self.keys))], return_index=True)
        return unique_exps[np.argsort(unique_inds)]

    @property
    def avail_x_names(self):
        avail_x_names = ['YEAR', 'MMDDHH', 'FHOUR', 'VMAX0', 'NCI', 'VMAXN', 'DSDV', 'LGDV', 'HWDV', 'AVDV', 'VMXC', 'DV12', 'SLAT', 'SSTN', 'SHDC', 'DTL', 'D200', 'T200', 'RHMD', 'SPDX', 'SPDY', 'LON0', 'LAT0', 'NCT', 'LONN', 'LATN', 'AVDX', 'EMDX', 'EGDX', 'HWDX', 'LONC', 'AVDY', 'EMDY', 'EGDY', 'HWDY', 'LATC']
        return avail_x_names

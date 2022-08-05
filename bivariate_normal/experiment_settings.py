"""Hurricane prediction experimental settings

uncertainty_type: "shash3", "bnn", "mcdrop", "reg"
val_condition   : "random", "years"  

"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__   = "29 July 2022"


def get_settings(experiment_name):
    experiments = {   
        #--------------------------------------------------
        "bivariate_normal_101_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bivariate_normal',  
            "leadtime": 48,
            "basin": "EP|CP",
            "undersample": False,
            "hiddens": [5, 5],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed_list": [123, 234, 345],
            "rng_seed": None,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 250,
            "test_condition": "leave-one-out",
            "years_test": None,
            "val_condition": "random",
            "n_val": 200,
            "n_train": "max",
        },
        "bivariate_normal_102_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bivariate_normal',  
            "leadtime": 48,
            "basin": "EP|CP",
            "undersample": False,
            "hiddens": [15, 10],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed_list": [123, 234, 345],
            "rng_seed": None,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 250,
            "test_condition": "leave-one-out",
            "years_test": None,
            "val_condition": "random",
            "n_val": 200,
            "n_train": "max",
        },        
        
    }

    return experiments[experiment_name]
"""Save the transfer blueprint for the Fortran evaluation engine.

Functions
---------
save_transfer_blueprint(data_summary, fit_summary, model, model_path,
    model_name, settings, version)

Notes
-----
* A model is a sequence of three distinct sets of components: the input
layer, the hidden layers, and the output channels.

-- The input layer normalizes the input. The input layer is fully-connected to
the first hidden layer.

-- The hidden layers are a sequence of one or more fully-connected (dense)
layers. The hidden layers may be different sizes, i.e. have a different number
of neurons. Each hidden layer has a matrix of weights, a vector of biases,
and an activation function.

-- There is one output channel for each scalar output (e.g. each of the five
parameters of the bivariate normal distribution). Each output channel has one
neuron, which is fully-connected to the last hidden layer. The output from the
neuron is transforned and then rescaled.

"""
import numpy as np
import json
import toolbox

__author__ = "Randal J. Barnes and Elizabeth A. Barnes"
__date__ = "28 October 2022"


def save_transfer_blueprint(
    data_summary,
    fit_summary,
    model,
    model_path,
    model_name,
    settings,
    version,
):
    """Save the model, weights, history, and metadata.

    Arguments
    ---------
    data_summary : dict

    fit_summary : dict

    model : tensorflow.keras.models.Model

    model_path : str
        Path to the folder for saved models, which is used to store the
            *_model,
            *_weights.h5,
            *_history.pickle, and
            *_metadata.json
        files for a run.

    model_name : str
        The unique model name to distinguish one run form another. This name
        is the initial component of each saved file and model folder.

    settings : dict
        Dictionary of experiment settings for the run.

    version : str
        Version of the train_intensity notebook.

    Returns
    -------
    None

    """
    blueprint = {}

    # General traits
    blueprint["n_input"] = len(data_summary["x_names"])
    blueprint["n_hidden"] = len(settings["hiddens"])
    blueprint["n_output"] = 5

    blueprint["input_names"] = data_summary["x_names"]
    blueprint["output_names"] = ["mu_u", "mu_v", "sigma_u", "sigma_v", "rho"]

    # InputLayerTraits
    input_traits = {
        "n_input": model.get_layer("normalization").adapt_mean.shape[0],
        "mean": model.get_layer("normalization").adapt_mean.numpy(),
        "std": np.sqrt(model.get_layer("normalization").adapt_variance.numpy()),
    }
    blueprint["input_traits"] = input_traits

    # HiddenLayerTraits
    hidden_traits = []

    for i in range(len(settings["hiddens"])):
        if i == 0:
            layer_name = "dense"
        else:
            layer_name = "dense" + f"_{i}"

        weights = model.get_layer(layer_name).get_weights()[0]
        bias = model.get_layer(layer_name).get_weights()[1]
        hidden_traits.append(
            {
                "n_input_connections": weights.shape[0],
                "n_output_connections": weights.shape[1],
                "weights": weights,
                "bias": bias,
                "activation": settings["act_fun"].upper(),
            }
        )
    blueprint["hidden_traits"] = hidden_traits

    # OutputLayerTraits
    output_traits = []

    OUTPUT_CHANNEL_TRAITS = [
        ("mu_U_unit", "LINEAR", "mu_u_unit"),
        ("mu_V_unit", "LINEAR", "mu_v_unit"),
        ("alpha_unit", "SOFTPLUS", "sigma_u_unit"),
        ("beta_unit", "SOFTPLUS", "sigma_v_unit"),
        ("gamma_unit", "TANH", None),
    ]

    for traits in OUTPUT_CHANNEL_TRAITS:
        weights = model.get_layer(traits[0]).get_weights()[0].squeeze()
        bias = model.get_layer(traits[0]).get_weights()[1][0]

        if traits[2] is not None:
            mean = model.get_layer(traits[2]).offset
            std = model.get_layer(traits[2]).scale
        else:
            mean = 0.0
            std = 1.0

        output_traits.append(
            {
                "n_input_connections": weights.shape[0],
                "weights": weights,
                "bias": bias,
                "transformation": traits[1],
                "mean": mean,
                "std": std,
            }
        )
    blueprint["output_traits"] = output_traits

    with open(model_path + model_name + "_blueprint.json", "w") as handle:
        json.dump(blueprint, handle, indent="   ", cls=toolbox.NumpyEncoder)

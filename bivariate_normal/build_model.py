"""Build the fully-connected network architecture.

Classes
---------
Exponentiate(keras.layers.Layer)

Functions
---------
make_model(settings, x_train, onehot_train, model_compile)
build_bivariate_normal_model(hiddens, input_shape, output_shape, ridge_penalty, act_fun, rng_seed)

"""
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from custom_loss import compute_shash_NLL, compute_NLL
from custom_metrics import CustomMAE, InterquartileCapture, SignTest

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "01 August 2022"


class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)


def make_model(settings, x_train, onehot_train, model_compile=False):
    model = build_bivariate_normal_model(
        x_train,
        onehot_train,
        hiddens=settings["hiddens"],
        output_shape=onehot_train.shape[1],
        ridge_penalty=settings["ridge_param"],
        act_fun=settings["act_fun"],
        rng_seed=settings["rng_seed"],
    )

    if model_compile == True:
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=settings["learning_rate"],
            ),
            loss=compute_shash_NLL,
            metrics=[
                CustomMAE(name="custom_mae"),
                InterquartileCapture(name="interquartile_capture"),
                SignTest(name="sign_test"),
            ],
        )

    return model


def build_bivariate_normal_model(
    x_train,
    onehot_train,
    hiddens,
    ridge_penalty=[
        0.0,
    ],
    act_fun="relu",
    rng_seed=999,
):
    """Build the fully-connected bivariate normal network architecture with
    internal scaling.

    Arguments
    ---------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : numpy.ndarray
        The training split of the scaled y data is in the first column.
        The remaining columns are filled with zeros. The number of columns
        equal the number of distribution parameters.
        shape = [n_train, n_parameters].

    hiddens : list (integers)
        Numeric list containing the number of neurons for each layer.

    ridge_penalty : float, default=0.0
        The L2 regularization penalty for the first layer.

    act_fun : function, default="relu"
        The activation function to use on the deep hidden layers.

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----
    * The first layer of the network model normalizes the x input
        values automatically. We must normalize the y target values
        manually.
        
    * We have two target variates in this model: OBDX and OBDY. There
        are too many x's and y's wandering through our notation to keep
        everything straight.  To clarify this mess, we define 
        
            u = OBDX, with statistics u_avg and u_std
            v = OBDY, with statistics v_avg and v_std
        
        Then we define the normalized versions as
        
            U = (u - u_avg)/u_std
            V = (v - v_avg)/v_std
        
    * The conditional bivariate normal distribution has five parameters:
    
            ev_u, ev_v, cov_uu, cov_vv, and cov_uv.
        
    * The network predicts the parameters of the normalized variates:
    
            ev_U, ev_V, log(cov_UU), log(cov_VV), and cov_UV.
        
        The "log" terms are necessary to guarantee positive variances.
        
    * To recover the parameters of the conditional bivariate normal 
        distribution we use:
        
            ev_u = ev_U * u_std + u_avg
            ev_v = ev_V * v_std + v_avg
        
            cov_uu = exp(log(cov_UU)) * u_std * u_std
            cov_vv = exp(log(cov_VV)) * v_std * v_std
            cov_uv = cov_UV * u_std * v_std
    
    """
    # set inputs
    if len(hiddens) != len(ridge_penalty):
        ridge_penalty = np.ones(np.shape(hiddens)) * ridge_penalty

    # The avg and std for feature normalization are computed from x_train.
    # Using the .adapt method, these are set once and do not change, but
    # the constants travel with the model.
    inputs = tf.keras.Input(shape=x_train.shape[1:])

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    x = normalizer(inputs)

    # Initialize the hidden layers.
    for ilayer, layer_size in enumerate(hiddens):
        x = tf.keras.layers.Dense(
            units=layer_size,
            activation=act_fun,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(
                l1=0.00, l2=ridge_penalty[ilayer]
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                seed=rng_seed + ilayer
            ),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=rng_seed + ilayer
            ),
        )(x)

    # Compute the mean and standard deviation of the training target 
    # data. These are used to normalize the data and then to rescale
    # the parameters.
    u_avg = np.mean(onehot_train[:, 0])
    u_std = np.std(onehot_train[:, 0])

    v_avg = np.mean(onehot_train[:, 1])
    v_std = np.std(onehot_train[:, 1])

    # Units to predict the conditional expect value of u.
    ev_U_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="ev_U_unit",
    )(x)

    ev_u_unit = tf.keras.layers.Rescaling(
        scale=u_std,
        offset=u_avg,
        name="ev_u_unit",
    )(ev_U_unit)

    # Units to predict the conditional expect value of v.    
    ev_V_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="ev_V_unit",
    )(x)

    ev_v_unit = tf.keras.layers.Rescaling(
        scale=v_std,
        offset=v_avg,
        name="ev_v_unit",
    )(ev_V_unit)
    
    # Units to predict the conditional variance of u.
    log_cov_UU_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_cov_UU_unit",
    )(x)

    cov_UU_unit = Exponentiate(
        name="cov_UU_unit",
    )(log_cov_UU_unit)
    
    cov_uu_unit = tf.keras.layers.Rescaling(
        scale=std_u*std_u,
        offset=0.0,
        name="cov_uu_unit",
    )(cov_UU_unit)

    # Units to predict the conditional variance of v.
    log_cov_VV_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_cov_VV_unit",
    )(x)

    cov_VV_unit = Exponentiate(
        name="cov_VV_unit",
    )(log_cov_VV_unit)
    
    cov_vv_unit = tf.keras.layers.Rescaling(
        scale=std_v*std_v,
        offset=0.0,
        name="cov_vv_unit",
    )(cov_VV_unit)
    
    # Units to predict the conditional covariance of u and v.
    cov_UV_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="cov_UV_unit",
    )(x)
   
    cov_uv_unit = tf.keras.layers.Rescaling(
        scale=std_u*std_v,
        offset=0.0,
        name="cov_uv_unit",
    )(cov_UV_unit)

    
    output_layer = tf.keras.layers.concatenate(
        [ev_u_unit, ev_v_unit, cov_uu_unit, cov_vv_unit, cov_uv_unit], axis=1
    )

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model
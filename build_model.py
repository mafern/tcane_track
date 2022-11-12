"""Build the fully-connected network architecture.

Classes
---------
class Softplus(keras.layers.Layer)
class Tanh(keras.layers.Layer)

Functions
---------
make_model(settings, x_train, label_train, model_compile)

build_bivariate_normal_model(hiddens, input_shape, output_shape,
    ridge_penalty, act_fun, rng_seed)

"""
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow_probability as tfp

from custom_metrics import CustomMAE, InterquartileCapture, SignTest
from custom_loss import compute_bivariate_normal_nll

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "12 November 2022"


class Softplus(keras.layers.Layer):
    """Custom layer to map the alpha {-inf : +inf} onto sigma_U {0 : +inf}
    and beta {-inf : inf} onto sigma_V {0 : +inf} inline."""

    def __init__(self, **kwargs):
        super(Softplus, self).__init__(**kwargs)
        self.softplus = tfp.bijectors.Softplus()

    def call(self, inputs):
        return self.softplus.forward(inputs)


class Tanh(keras.layers.Layer):
    """Custom layer to map the gamma {-inf : +inf} onto rho {-1 : +1} inline."""

    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)
        self.tanh = tfp.bijectors.Tanh()

    def call(self, inputs):
        return self.tanh.forward(inputs)


def make_model(settings, x_train, label_train, model_compile=False):
    if settings["uncertainty_type"] == "bivariate_normal":
        iscentered = False
    elif settings["uncertainty_type"] == "centered_bivariate_normal":
        iscentered = True
    else:
        raise NotImplementedError

    model = build_bivariate_normal_model(
        x_train,
        label_train,
        hiddens=settings["hiddens"],
        ridge_penalty=settings["ridge_param"],
        act_fun=settings["act_fun"],
        rng_seed=settings["rng_seed"],
        iscentered=iscentered,
    )

    if model_compile:
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=settings["learning_rate"],
            ),
            loss=compute_bivariate_normal_nll,
            metrics=[
                CustomMAE(name="custom_mae"),
                InterquartileCapture(name="interquartile_capture"),
                SignTest(name="sign_test"),
            ],
        )

    return model


def build_bivariate_normal_model(
    x_train,
    label_train,
    hiddens,
    ridge_penalty=0.0,
    act_fun="relu",
    rng_seed=999,
    iscentered=False
):
    """Build the fully-connected bivariate normal network architecture with
    internal scaling.

    Arguments
    ---------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    label_train : numpy.ndarray
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

    rng_seed : int
        Base random number seed for keras.

    iscentered : boolean, default=False
        If True, a centered bivariate normal distribution (mu_u = mu_v = 0) is
        used, otherwise mu_u and mu_v are free to be trained.

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----
    * We have two target variates in this model: OBDX and OBDY. To simplify
        this discussion, we introduce the following notation.

            u = OBDX
            v = OBDY

        We model u and v as realizations of bivariate normal random variates.

    * The conditional bivariate normal distribution is most commonly defined
        by five parameters: two means, two standard deviations, and a
        correlation. We denote these by

            mu_u, mu_v, sigma_u, sigma_v, and rho,

        where mu_u and mu_v are unconstrained, but sigma_u > 0, sigma_v > 0,
        and -1 < rho < 1.

        The conditional mean vector is given by

            [ mu_u ]
            [ mu_v ]

        Note: if iscentered is True, mu_u and mu_v are fixed at 0.0 and
        designated trainable = False. The conditional variance/covariance
        matrix is given by

            [ sigma_u^2,           rho*sigma_u*sigma_v ]
            [ rho*sigma_u*sigma_v, sigma_v^2           ]

    * The first layer of the network model normalizes the x input
        values automatically using a TensorFlow adaptive normalizer

            tf.keras.layers.Normalization()

        This means that the network inputs are the physical dimensioned
        values, and the internal normalizing constants travel with
        the model.

    * We do not explicitly normalize the two target variates. Rather,
        we implicitly normalize the target variates by scaling the
        network output as follows.

        We compute the averages and standard deviations for u and v
        of the training data. We denote these

            u => u_avg and u_std
            v => v_avg and v_std

        Then we define, but do not explicitly compute, the normalized
        versions of u and v as

            U = (u - u_avg)/u_std
            V = (v - v_avg)/v_std

        Since U and V are simple affine (linear) transformations of u
        and v, U and V also follow a bivariate normal distribution.

        Internally, the network predicts the five parameters of the
        conditional bivariate normal distribution of U and V:

            mu_U, mu_V, sigma_U, sigma_V, and rho.

        We then rescale these parameters in the output layer using
        tf.keras.layers.Rescaling layers; that is

            mu_u = mu_U * u_std + u_avg
            mu_v = mu_V * v_std + v_avg

            sigma_u = sigma_U * u_std
            sigma_v = sigma_V * v_std

        and rho is dimensionless, so it does not need to be rescaled.

        The scaling parameters u_avg, u_std, v_avg, and v_std travel
        with the model as part of the output layer.

    * The parameters of the conditional bivariate normal distribution
        for U and V must also satisfy sigma_U > 0, sigma_V > 0, and
        -1 < rho < 1. We use standard TensorFLow tricks to guarantee
        that we meet these constraints.

        We have the network predict five unconstrained outputs:

            mu_U, mu_V, alpha, beta, and gamma.

        We then compute sigma_U and sigma_V using

            tfp.bijectors.Softplus()

        that is

            sigma_U = log(1 + exp(alpha))
            sigma_V = log(1 + exp(beta))

        The Softplus maps {-infinity : infinity} onto {0 : infinity}.

        We compute rho using

            tfp.bijectors.Tanh()

        The Tanh bijector maps {-infinity : infinity} onto {-1 : 1}.

        Note, mu_U and mu_V are unconstrained, so they do not require
        special treatment.

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
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty[ilayer]),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + ilayer),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=rng_seed + ilayer
            ),
        )(x)

    # Compute the mean and standard deviation of the training target
    # data. These are used to implicitly normalize the target variates
    # by rescaling the parameters. (See the notes above.)
    u_avg = np.mean(label_train[:, 0])
    u_std = np.std(label_train[:, 0])

    v_avg = np.mean(label_train[:, 1])
    v_std = np.std(label_train[:, 1])

    # Fix mu_u = mu_v = 0 if is iscentered.
    if iscentered is True:
        mu_trainable = False
        u_scale = 0.0
        u_offset = 0.0
        v_scale = 0.0
        v_offset = 0.0
    else:
        mu_trainable = True
        u_scale = u_std
        u_offset = u_avg
        v_scale = v_std
        v_offset = v_avg

    # Units to predict the conditional expect value of u.
    mu_u_raw_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_u_raw_unit",
        trainable=mu_trainable,
    )(x)

    mu_u_unit = tf.keras.layers.Rescaling(
        scale=u_scale,
        offset=u_offset,
        name="mu_u_unit",
    )(mu_u_raw_unit)

    # Units to predict the conditional expect value of v.
    mu_v_raw_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_v_raw_unit",
        trainable=mu_trainable,
    )(x)

    mu_v_unit = tf.keras.layers.Rescaling(
        scale=v_scale,
        offset=v_offset,
        name="mu_v_unit",
    )(mu_v_raw_unit)

    # Units to predict the conditional standard deviation of u.
    sigma_u_raw_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="sigma_u_raw_unit",
    )(x)

    sigma_u_prescale_unit = Softplus(
        name="sigma_u_prescale_unit",
    )(sigma_u_raw_unit)

    sigma_u_unit = tf.keras.layers.Rescaling(
        scale=u_std,
        offset=0.0,
        name="sigma_u_unit",
    )(sigma_u_prescale_unit)

    # Units to predict the conditional standard deviation of v.
    sigma_v_raw_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="sigma_v_raw_unit",
    )(x)

    sigma_v_prescale_unit = Softplus(
        name="sigma_v_prescale_unit",
    )(sigma_v_raw_unit)

    sigma_v_unit = tf.keras.layers.Rescaling(
        scale=v_std,
        offset=0.0,
        name="sigma_v_unit",
    )(sigma_v_prescale_unit)

    # Units to predict the conditional correlation of u and v.
    rho_raw_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="rho_raw_unit",
    )(x)

    rho_unit = Tanh(
        name="rho_unit",
    )(rho_raw_unit)

    # Stitch everything together.
    output_layer = tf.keras.layers.concatenate(
        [mu_u_unit, mu_v_unit, sigma_u_unit, sigma_v_unit, rho_unit], axis=1
    )

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

"""Build the fully-connected network architecture.

Classes
---------
class Softplus(keras.layers.Layer)
class Tanh(keras.layers.Layer)

Functions
---------
make_model(settings, x_train, onehot_train, model_compile)

build_bivariate_normal_model(hiddens, input_shape, output_shape,
    ridge_penalty, act_fun, rng_seed)

build_centered_bivariate_normal_model(hiddens, input_shape,
    output_shape, ridge_penalty, act_fun, rng_seed)

"""
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from custom_loss import compute_bivariate_normal_nll
from custom_loss import compute_centered_bivariate_normal_nll

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "24 August 2022"


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


def make_model(settings, x_train, onehot_train, model_compile=False):
    if settings["uncertainty_type"] == "bivariate_normal":
        model = build_bivariate_normal_model(
            x_train,
            onehot_train,
            hiddens=settings["hiddens"],
            ridge_penalty=settings["ridge_param"],
            act_fun=settings["act_fun"],
            rng_seed=settings["rng_seed"],
        )

        if model_compile:
            model.compile(
                optimizer=optimizers.Adam(
                    learning_rate=settings["learning_rate"],
                ),
                loss=compute_bivariate_normal_nll,
            )

    elif settings["uncertainty_type"] == "centered_bivariate_normal":
        model = build_centered_bivariate_normal_model(
            x_train,
            onehot_train,
            hiddens=settings["hiddens"],
            ridge_penalty=settings["ridge_param"],
            act_fun=settings["act_fun"],
            rng_seed=settings["rng_seed"],
        )

        if model_compile:
            model.compile(
                optimizer=optimizers.Adam(
                    learning_rate=settings["learning_rate"],
                ),
                loss=compute_centered_bivariate_normal_nll,
            )

    else:
        raise NotImplementedError

    return model


def build_bivariate_normal_model(
        x_train,
        onehot_train,
        hiddens,
        ridge_penalty=0.0,
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

    rng_seed : int
        Base random number seed for keras.

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

        and the conditional variance/covariance matrix is given by

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
    u_avg = np.mean(onehot_train[:, 0])
    u_std = np.std(onehot_train[:, 0])

    v_avg = np.mean(onehot_train[:, 1])
    v_std = np.std(onehot_train[:, 1])

    # Units to predict the conditional expect value of u.
    mu_U_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_U_unit",
    )(x)

    mu_u_unit = tf.keras.layers.Rescaling(
        scale=u_std,
        offset=u_avg,
        name="mu_u_unit",
    )(mu_U_unit)

    # Units to predict the conditional expect value of v.
    mu_V_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_V_unit",
    )(x)

    mu_v_unit = tf.keras.layers.Rescaling(
        scale=v_std,
        offset=v_avg,
        name="mu_v_unit",
    )(mu_V_unit)

    # Units to predict the conditional standard deviation of u.
    alpha_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="alpha_unit",
    )(x)

    sigma_U_unit = Softplus(
        name="sigma_U_unit",
    )(alpha_unit)

    sigma_u_unit = tf.keras.layers.Rescaling(
        scale=u_std,
        offset=0.0,
        name="sigma_u_unit",
    )(sigma_U_unit)

    # Units to predict the conditional standard deviation of v.
    beta_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="beta_unit",
    )(x)

    sigma_V_unit = Softplus(
        name="sigma_V_unit",
    )(beta_unit)

    sigma_v_unit = tf.keras.layers.Rescaling(
        scale=v_std,
        offset=0.0,
        name="sigma_v_unit",
    )(sigma_V_unit)

    # Units to predict the conditional correlation of u and v.
    gamma_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="gamma_unit",
    )(x)

    rho_unit = Tanh(
        name="rho_unit",
    )(gamma_unit)

    # Stitch everything together.
    output_layer = tf.keras.layers.concatenate(
        [mu_u_unit, mu_v_unit, sigma_u_unit, sigma_v_unit, rho_unit], axis=1
    )

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model


def build_centered_bivariate_normal_model(
        x_train,
        onehot_train,
        hiddens,
        ridge_penalty=0.0,
        act_fun="relu",
        rng_seed=999,
):
    """Build the fully-connected centered (mu_u = mu_v = 0) bivariate
    normal network architecture with internal scaling.

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

    rng_seed : int
        Base random number seed for keras.

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----
    * The conditional mean parameters, [mu_u, mu_v], are fixed at 0.
        This is accomplished by setting the Rescaling scale and offset
        parameters to 0 for both the mu_u_unit and the mu_v_unit.

    * See the notes for the build_bivariate_normal_model function for
        more details. The two versions of this function are almost
        identical.  The only differences are the details discussed in
        the preceding note.
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

    # Compute the standard deviation of the training target data.
    # These are used to implicitly normalize the target variates
    # by rescaling the parameters. (See the notes above.)
    u_std = np.std(onehot_train[:, 0])
    v_std = np.std(onehot_train[:, 1])

    # Units to fake-predict the conditional expect value of u.
    mu_U_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_U_unit",
        trainable=False,
    )(x)

    mu_u_unit = tf.keras.layers.Rescaling(
        scale=0.0,
        offset=0.0,
        name="mu_u_unit",
    )(mu_U_unit)

    # Units to fake-predict the conditional expect value of v.
    mu_V_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed + 100),
        name="mu_V_unit",
        trainable=False,
    )(x)

    mu_v_unit = tf.keras.layers.Rescaling(
        scale=0.0,
        offset=0.0,
        name="mu_v_unit",
    )(mu_V_unit)

    # Units to predict the conditional standard deviation of u.
    alpha_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="alpha_unit",
    )(x)

    sigma_U_unit = Softplus(
        name="sigma_U_unit",
    )(alpha_unit)

    sigma_u_unit = tf.keras.layers.Rescaling(
        scale=u_std,
        offset=0.0,
        name="sigma_u_unit",
    )(sigma_U_unit)

    # Units to predict the conditional standard deviation of v.
    beta_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="beta_unit",
    )(x)

    sigma_V_unit = Softplus(
        name="sigma_V_unit",
    )(beta_unit)

    sigma_v_unit = tf.keras.layers.Rescaling(
        scale=v_std,
        offset=0.0,
        name="sigma_v_unit",
    )(sigma_V_unit)

    # Units to predict the conditional correlation of u and v.
    gamma_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="gamma_unit",
    )(x)

    rho_unit = Tanh(
        name="rho_unit",
    )(gamma_unit)

    # Stitch everything together.
    output_layer = tf.keras.layers.concatenate(
        [mu_u_unit, mu_v_unit, sigma_u_unit, sigma_v_unit, rho_unit], axis=1
    )

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

"""Custom loss functions using on negative log-likelihood.

Functions
---------
compute_bivariate_normal_NLL(y_true, param)

"""
import tensorflow as tf
import tensorflow_probability as tfp

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "09 August 2022"


def compute_bivariate_normal_NLL(y_true, param):
    """Negative log-likelihood loss using the bivariate normal distribution.

    Arguments
    ---------
    y_true : tensor
        The ground truth values.  The first column holds ODBX, the second
        column holds ODBY. The remaining three columns are filled with zeros,
        as required by TensorFlow.
        shape = [batch_size, 5]

    param :
        The predicted local conditional distribution parameters:

            [ sigma_u, sigma_v, rho ]

        where u = OBDX, and v = OBDY. Note, mu_u and mu_v = 0.
        shape = [batch_size, 5]

    Returns
    -------
    loss : tensor, shape = [1, 1]
        The average negative log-likelihood of the batch using the predicted
        conditional distribution parameters.

    Notes
    -----
    * The conditional variance/covariance matrix is given by

            [ sigma_u^2,           rho*sigma_u*sigma_v ]
            [ rho*sigma_u*sigma_v, sigma_v^2           ]

    * TensorFlow does not want users to pass the full covariance matrix.

            WARNING:tensorflow: MultivariateNormalFullCovariance
            is deprecated and will be removed after 2019-12-01.

        TensorFlow wants users to pass the lower triangular matrix
        resulting from the Cholesky decomposition of the full covariance
        matrix.

            `MultivariateNormalFullCovariance` is deprecated, use
            `MultivariateNormalTriL(
                loc=loc,
                scale_tril=tf.linalg.cholesky(covariance_matrix)
            )`
            instead.

        Rather than building the full covariance matrix and then calling
        tf.linalg.cholesky, we will compute the lower triangular Cholesky
        decompsition matrix directly.

    * Since we are working with a bivariate normal distribution, the
        covariance matrix is only (2 x 2), and the analytic expressions
        are relatively simple.

        Let the lower triangular Cholesky decompsition matrix L be
        given by

            [ c, 0 ]
            [ b, a ]

        This naming convention is in keeping with the clockwise spiral
        approach used by tfp.bijectors.FillTriangular.  The Cholesky
        decomposition is defined by

            L * L' = covariance matrix

        so we write

            [ c, 0 ] [ c, b ] = [ c^2, cb        ]
            [ b, a ] [ 0, a ]   [ cb,  b^2 + a^2 ]

            = [ sigma_u^2,           rho*sigma_u*sigma_v ]
              [ rho*sigma_u*sigma_v, sigma_v^2           ]

        Equating like terms we find:

            c = sigma_u

            cb = rho*sigma_u*sigma_v
            --> b = sigma_v * rho

            b^2 + a^2 = sigma_v^2
            --> a = sqrt(sigma_v^2 - b^2) = sigma_v * sqrt(1 - rho^2)

    """
    b = tfp.bijectors.FillTriangular(upper=False)
    mvn = tfp.distributions.MultivariateNormalTriL(
        loc=tf.zeros_like(param[:, 0:2]),
        scale_tril=b.forward(
            tf.stack(
                (
                    param[:, 1] * tf.math.sqrt(1.0 - tf.math.square(param[:, 2])),
                    param[:, 1] * param[:, 2],
                    param[:, 0],
                ),
                axis=1,
            )
        ),
    )

    loss = -mvn.log_prob(y_true[:, 0:2])
    return tf.reduce_mean(loss, axis=-1)

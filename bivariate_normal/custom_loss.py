"""Custom loss functions for nonlinear, heterogeneous, parametric regression
    using the sinh-arcsinh normal distribution.

Functions
---------
compute_NLL(y_true, param)

"""
import tensorflow as tf

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "01 August 2022"


def compute_NLL(y_true, param):
    """Negative log-likelihood loss using the bivariate normal distribution.

    Arguments
    ---------
    y_true : tensor
        The ground truth values.  The first column holds ODBX, the second 
        column holds ODBY. The remaining three columns are filled with zeros.
        shape = [batch_size, 5]

    param :
        The predicted local conditional distribution parameters:
        
            ev_u, ev_v, cov_uu, cov_vv, cov_uv
            
        where u = OBDX, and v = OBDy.
        shape = [batch_size, 5]

    Returns
    -------
    loss : tensor, shape = [1, 1]
        The average negative log-likelihood of the batch using the predicted
        conditional distribution parameters.

    Notes
    -----

    """
    mu_u = param[:, 0]
    mu_v = param[:, 1]

    sigma_u = param[:, 2]
    sigma_v = param[:, 3]
    rho = param[:, 4]

    dist = shash_tfp.Shash(mu, sigma, gamma, tau)
    loss = -dist.log_prob(y_true[:, 0])
    return tf.reduce_mean(loss, axis=-1)

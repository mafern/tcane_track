"""Tools for the Mahalanobis cdf using a bivariate normal.

Functions
---------
plot_cdf(mu_u, mu_v, sigma_u, sigma_v, rho)
compute_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, y)

Notes
-----
* The Mahalanobis cdf is defined in the "Cumulative distribution
    function" section of [1].
    
* The Mahalanobis cdf is the probability that a sample lies inside
    the ellipse determined by its Mahalanobis distance.

References
----------
[1] Wikipedia contributors. (2022, July 20). Multivariate normal 
    distribution. In Wikipedia, The Free Encyclopedia. Retrieved 
    19:53, August 5, 2022.

[2] Wikipedia contributors. (2022, June 21). Mahalanobis distance.
    In Wikipedia, The Free Encyclopedia. Retrieved 19:55, August 5, 2022.

[3] Michael Bensimhoun. (2009, June). N-dimension Cumulative Function
    and Other Useful Facts About Gaussians and Normal Densities.
    https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf

"""
import matplotlib.pyplot as plt
import numpy as np
import palettable

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "06 August 2022"



COLOR_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.mpl_colors
THETA = np.linspace(0, 2 * np.pi, 1000)


def plot_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, besttrack_u=None, besttrack_v=None, colors=COLOR_DEFAULT):
    """Plot the Mahalanobis cdf.

    Plot the eliptical contours of the Mahalanobis cdf for a bivariate
    normal distribution. The nine contour levels are the distribution's
    deciles: probability of capture = {0.10, 0.20, ..., 0.90}.

    Plot the Consensus track as a circle at (0, 0). If given, plot the
    Besttrack as a square.

    Arguments
    ---------
    mu_u : float
        mean of u.

    mu_v : float
        mean of v.

    sigma_u : float, u > 0.
        standard deviation of u.

    sigma_v : float, v > 0.
        standard deviation of v.

    rho : float, -1 < rho < 1.
        correlation between u and v.

    besttrack_u : float or None
        the u-coordinate of the truth.

    besttrack_v : float or None
        the v-coordinate of the truth.

    Returns
    -------
    None

    Notes
    -----
    * The equations for x and y come from the bottom of Page 2 in [3].

    """
    for i, p in enumerate(np.arange(9, 0, -1) / 10.0):
        r = np.sqrt(-2.0 * (np.log(1 - p)))
        x = r * sigma_u * np.cos(THETA) + mu_u
        y = (
            r * sigma_v * (rho * np.cos(THETA) + np.sqrt(1 - rho * rho) * np.sin(THETA))
            + mu_v
        )
        plt.fill(x, y, color=colors[i])

    # plot consensus and true label
    plt.plot(0, 0, "o", color="k", markersize=5, markerfacecolor="None", label="Consensus")

    if besttrack_u is not None and besttrack_v is not None:
        plt.plot(
            besttrack_u,
            besttrack_v,
            "s",
            color="k",
            markersize=6,
            label="BestTrack",
        )

    plt.axis("equal")


def compute_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, u, v):
    """Compute the Mahalanobis cdf for [u, v] using a bivariate normal
    distribution.

    Arguments
    ---------
    mu_u : float
        mean of u.

    mu_v : float
        mean of v.

    sigma_u : float, u > 0.
        standard deviation of u.

    sigma_v : float, v > 0.
        standard deviation of v.

    rho : float, -1 < rho < 1.
        correlation between u and v.

    Notes
    -----
    * The equations for the Mahalanobis distance, r_sqr, comes from
        the bottom of Page 2 in [3].

    * The Mahalanobis distance, r_sqr, follows the chi-squared distribution
        with 2 degrees of freedom.

    * The equation for the returned cdf comes for the "Normal Distribution"
        section of [2], and the middle of Page 4 of [3].

    """
    U = (u - mu_u) / sigma_u
    V = (v - mu_v) / sigma_v
    r_sqr = 1.0 / (1.0 - rho * rho) * (U * U - 2 * rho * U * V + V * V)
    return 1.0 - np.exp(-r_sqr / 2.0)

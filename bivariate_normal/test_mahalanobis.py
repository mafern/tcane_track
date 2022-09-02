import numpy as np
import mahalanobis

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "02 September 2022"


def test_compute_cdf():
    """Test the theoretical mahalanobi cdf.

    The test compares the probability integral transform (PIT)
    histogram from a large set of simulated bivariate realizations
    to an anticipated uniform distribution.  The comparison uses
    the Mahalanobis cdf.

    Notes
    -----
    * The test fails is any of the bin counts deviate from the expected
        number by more than 4 standard deviations. The probability that
        this happens by random chance is less than 0.00005.
    """

    N_REALIZATIONS = 100000
    N_BINS = 10

    mu_u = 1.0
    mu_v = 1.0
    sigma_u = 2.0
    sigma_v = 1.0
    rho = 0.0

    mean = np.array([mu_u, mu_v])
    cov = np.array([[sigma_u ** 2, rho * sigma_v * sigma_u], [rho * sigma_v * sigma_u, sigma_v ** 2]])
    z = np.random.multivariate_normal(mean, cov, size=N_REALIZATIONS)

    F = mahalanobis.compute_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, z[:, 0], z[:, 1])

    bins = np.linspace(0, 1, N_BINS + 1)
    counts = np.histogram(F, bins)[0]

    expected_counts = float(N_REALIZATIONS) / float(N_BINS)
    var_counts = float(N_REALIZATIONS) * 1.0 / float(N_BINS) * (1.0 - 1.0 / float(N_BINS))
    std_counts = np.sqrt(var_counts)

    assert np.all(np.isclose(counts, expected_counts, atol=4.0 * std_counts))

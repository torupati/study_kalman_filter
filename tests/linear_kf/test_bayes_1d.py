import numpy as np

from linear_kf.bayes_1d.bayes_gaussian_1d import gauss_pdf, gaussian_posterior


def test_posterior_example_prior_2m_obs_3m():
    mu1, var1 = gaussian_posterior(2.0, 1.0, 3.0, 0.25)
    assert np.isclose(mu1, 2.8)
    assert np.isclose(var1, 0.2)


def test_posterior_matches_kalman_update():
    mu0, var0, z, r = -1.3, 0.7, 2.4, 1.9
    mu1, var1 = gaussian_posterior(mu0, var0, z, r)
    K = var0 / (var0 + r)
    assert np.isclose(mu1, mu0 + K * (z - mu0))
    assert np.isclose(var1, (1.0 - K) * var0)


def test_posterior_matches_normalized_product_on_grid():
    mu0, std0, z, std_r = 2.0, 1.0, 3.0, 0.5
    x = np.linspace(-6.0, 10.0, 20001)
    prod = gauss_pdf(x, mu0, std0) * gauss_pdf(x, z, std_r)
    prod /= np.trapezoid(prod, x)
    mu1, var1 = gaussian_posterior(mu0, std0 ** 2, z, std_r ** 2)
    assert np.allclose(prod, gauss_pdf(x, mu1, np.sqrt(var1)), atol=1e-8)

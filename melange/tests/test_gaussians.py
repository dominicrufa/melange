"""
test melange.gaussians
"""
from jax import random
from melange.gaussians import *
from scipy.stats import multivariate_normal
import tqdm
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

def run_multivariate_gaussian_logp(key, size):
    """
    i had to manually build a mulitvariate gaussian log_pdf calculator because it doesnt live anywhere
    in jax; this is a test to make sure it works (in a validation against the scipy package)
    """
    mukey, covkey, genkey  = random.split(key, 3)
    mu = random.uniform(mukey, shape = [size])

    #make a random positive semidefinite matrix to serve as the cov matrix
    A =  random.normal(covkey, shape=[size, size])
    cov = jnp.dot(A, A.T)

    random_number = random.multivariate_normal(key = genkey, mean = mu, cov = cov)
    logp = multivariate_gaussian_logp(random_number, mu, cov)

    #now validate
    valid_logp = multivariate_normal(mean = mu, cov = cov).logpdf(random_number)

    assert jnp.isclose(logp, valid_logp, atol=1e-6), f"check unsuccessful; difference = {abs(logp - valid_logp)}"

def test_multivariate_gaussian_logp(key = random.PRNGKey(0), size = 4, num_iters = 10):
    """
    make an assertion for a 4D multivariage gaussian logpdf
    """
    import tqdm
    for i in tqdm.trange(num_iters):
        key, newkey = random.split(key)
        run_multivariate_gaussian_logp(newkey, size)

def test_Gaussian_utils(dim=5):
    """
    this is a (trivial) test that will make sure that the manually-written (diagonal) multivariate normal logpdf calculation is close to the scipy version's
    """
    #make test vars
    for i in tqdm.trange(100):
        x = np.random.randn(dim)
        mean = np.random.randn(dim)
        cov = np.diag(np.abs(np.random.randn(dim)))

        mvn_logp = multivariate_normal.logpdf(x, mean, cov)
        manual_logp = unnormalized_Normal_logp(x, mean, jnp.diag(cov)) - Normal_logZ(mean, jnp.diag(cov))

        assert np.isclose(unnormalized_Normal_logp(x, mean, jnp.diag(cov)) - Normal_logZ(mean, jnp.diag(cov)), mvn_logp)

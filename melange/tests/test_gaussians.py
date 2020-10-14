"""
test melange.gaussians
"""
from jax import random
from melange.gaussians import *
from scipy.stats import multivariate_normal

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

def test_1D_multivariate_gaussian_logp(key = random.PRNGKey(0), size = 1, num_iters = 10):
    """
    make an assertion for a 1D multivariage gaussian logpdf
    """
    import tqdm
    for i in tqdm.trange(num_iters):
        key, newkey = random.split(key)
        run_multivariate_gaussian_logp(newkey, size)

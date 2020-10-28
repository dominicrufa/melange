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

def test_multivariate_gaussian_logp(key = random.PRNGKey(0), size = 4, num_iters = 10):
    """
    make an assertion for a 4D multivariage gaussian logpdf
    """
    import tqdm
    for i in tqdm.trange(num_iters):
        key, newkey = random.split(key)
        run_multivariate_gaussian_logp(newkey, size)

def test_twisted_gmm(dim=1, rand_key = random.PRNGKey(4), mixtures=5, atol=1e-4):
    """
    simple test that will assert weights, mus, and covs are unchanged upon twisting with zero functions;
    also asserts that the mixture weights sum to one upon a random twist.

    TODO : 
        1. there is something lossy in higher dimensions (probably a float32 issue) (fix the square mahalanobis function)
        2. fix the offset exp normalization to avoid underflow
        3. make covariance matrix assertions
    """
    from jax.lax import map
    A = jnp.diag(jnp.zeros(dim))
    b = jnp.zeros(dim)

    rand_mixkey, rand_mukey, rand_covkey, run_key = random.split(rand_key, 4)
    rand_mixs = random.normal(rand_mixkey, shape=(mixtures,))
    rand_mixs = rand_mixs - min(rand_mixs)
    rand_normal_mixs = rand_mixs / rand_mixs.sum()
    rand_mus = random.normal(rand_mukey, shape=(mixtures, dim))
    rand_covs = random.normal(rand_covkey, shape=(mixtures, dim, dim))
    rand_covs = map(lambda x: jnp.dot(x, x.transpose()), rand_covs)
    #print(rand_covs)

    log_normalizer, log_alpha_tildes, sigma_tildes, (twisted_mus, sigma_tildes) = get_twisted_gmm(rand_normal_mixs, rand_mus, rand_covs, A, b)
    #print(log_normalizer)
    assert jnp.isclose(jnp.exp(log_alpha_tildes).sum(), 1.), f"{jnp.exp(log_alpha_tildes).sum()}"
    #print(twisted_mus.shape)
    assert jnp.allclose(twisted_mus, rand_mus, atol)
    assert jnp.allclose(sigma_tildes, rand_covs, atol)

    idx, sam = sample_gmm(run_key, jnp.exp(log_alpha_tildes), twisted_mus, sigma_tildes)

"""
test melange.csmc
"""
from melange.gaussians import *
from melange.csmc import *
import tqdm
from jax import random
import numpy as np

def test_null_twisted_log_normalizer(dim=5):
    """
    assert that a gaussian with zero twisting parameters gives the same log normalizing constant and a 0-valued twisting log constant
    """
    x = np.random.randn(dim)
    mean = np.random.randn(dim)
    cov = np.abs(np.random.randn(dim))
    A, b = np.zeros(dim)[np.newaxis, ...], np.zeros(dim)[np.newaxis, ...]
    twisted_mu, twisted_cov, _ = twisted_normal_params(mean, cov, A[0], b[0])
    twisted_log_konstant = twist_log_constant(mean, cov, A[0], b[0])

    untwisted_logZ = Normal_logZ(mean, cov)
    twisted_logZ = Normal_logZ(twisted_mu, twisted_cov)
    assert np.isclose(untwisted_logZ, twisted_logZ + twisted_log_konstant)

def test_null_multiple_twists(dim=5, num_twists=3):
    """
    for several twists of a gaussian (with zero-valued twists), assert that the logpdf is the same as the untwisted gaussian
    """
    from melange.csmc import do_param_twist

    x = np.random.randn(dim)
    mean = np.random.randn(dim)
    cov = np.abs(np.random.randn(dim))
    As, bs = np.zeros((num_twists, dim)), np.zeros((num_twists, dim))

    untwisted_logZ = Normal_logZ(mean, cov)

    twisted_mu, twisted_cov, log_normalizer = do_param_twist(mean, cov, As, bs, True)

    assert np.allclose(twisted_mu, mean) #check there is no twisted parameter
    assert np.allclose(twisted_cov, cov) #check there is no twisted parameters
    assert np.isclose(untwisted_logZ, log_normalizer) #the log normalizing constant must be unchanged

def test_multiple_twists(dim=1, num_twists=3, integration_range=(-20, 20), num_point_ops=20000):
    """
    for several random twists of a base gaussian, assert an interconsistent logpdf
    """
    from jax.lax import map

    for i in tqdm.trange(10):
        #define the params
        x = np.random.randn(dim)
        mean = np.random.randn(dim)
        cov = np.abs(np.random.randn(dim)) + np.ones(dim)*10
        As, bs = np.abs(np.random.randn(num_twists, dim)), np.random.randn(num_twists, dim)

        #get the untwisted logZ, logpdf, and the twisting value
        untwisted_logZ = Normal_logZ(mean, cov)
        untwisted_logpdf = unnormalized_Normal_logp(x, mean, cov)
        log_twist = log_psi_twist(x, As, bs) #manually compute the twisting value
        manual_unnorm_logp_twist = untwisted_logpdf + log_twist #add the twist value to the untwisted unnormalized logp

        #do the parameter twist
        twisted_mu, twisted_cov, log_normalizer = do_param_twist(mean, cov, As, bs, True)
        twisted_logZ = Normal_logZ(twisted_mu, twisted_cov)
        twisted_logpdf = unnormalized_Normal_logp(x, twisted_mu, twisted_cov) + log_normalizer - twisted_logZ

        #make sure the manual and automatic one are equal
        assert np.isclose(manual_unnorm_logp_twist, twisted_logpdf)

        #now, we have to do the hard part and try to compute the twisted normalization constant...

        unnorm_logp = lambda x: jnp.exp(unnormalized_Normal_logp(x, mean, cov) + log_psi_twist(x, As, bs))
        vals = map(unnorm_logp, jnp.linspace(integration_range[0], integration_range[1], num_point_ops)[..., jnp.newaxis])
        dx = (integration_range[1] - integration_range[0]) / num_point_ops
        man_logZ = jnp.log(dx*vals.sum())
        assert np.isclose(man_logZ, log_normalizer, atol=1e-2)

def test_twisted_Gaussian(dim=5):
    """test to make sure that we can appropriately twist a gaussian"""

    for i in tqdm.trange(10): #this is slower, so we'll only do 10
        x = np.random.randn(dim)
        mean = np.random.randn(dim)
        cov = np.abs(np.random.randn(dim))
        A, b = np.abs(np.random.randn(dim))[np.newaxis, ...], np.random.randn(dim)[np.newaxis, ...]

        #manually compte the unnormalized twisted logpdf
        unnormalized_untwisted_logp = unnormalized_Normal_logp(x, mean, cov)
        log_twist = log_psi_twist(x, A, b)
        init_unnorm_twisted_logp = unnormalized_untwisted_logp + log_twist

        #automatically compute unnormalized logp
        twisted_mu, twisted_cov, _ = twisted_normal_params(mean, cov, A[0], b[0])
        twisted_log_konstant = twist_log_constant(mean, cov, A[0], b[0])

        auto_compute_unnorm_logp = unnormalized_Normal_logp(x, twisted_mu, twisted_cov) + twisted_log_konstant
        assert np.isclose(init_unnorm_twisted_logp, auto_compute_unnorm_logp)

def test_twisted_gmm(dim=5, mixtures=3):
    #make mixtures
    unnormalized_mixtures = np.abs(np.random.randn(mixtures))
    normalized_mixtures = unnormalized_mixtures/unnormalized_mixtures.sum()
    assert np.isclose(normalized_mixtures.sum(), 1.)

    #make randoms.
    means = np.random.randn(mixtures, dim)
    covs = np.abs(np.random.randn(mixtures, dim))
    A, b = np.abs(np.random.randn(dim)), np.random.randn(dim)
    A0, b0 = np.zeros(dim), np.zeros(dim)

    #compute the twisted_gaussians
    full_log_normalizer, log_normalized_twisted_mixtures, (twisted_mus, sigma_tildes) = get_twisted_gmm(normalized_mixtures, means, covs, A, b)
    null_full_log_normalizer, null_log_normalized_twisted_mixtures, (null_twisted_mus, null_sigma_tildes) = get_twisted_gmm(normalized_mixtures, means, covs, A0, b0)

    #assertions for null
    assert np.allclose(null_twisted_mus, means)
    assert np.allclose(null_sigma_tildes, covs)

    #assert twisted mixtures are normalized
    assert np.isclose(jnp.exp(log_normalized_twisted_mixtures).sum(), 1.), f"{log_normalized_twisted_mixtures}"

    #sample the non-null twisted gmm
    sample_gmm(random.PRNGKey(np.random.randint(1, 1000)), jnp.exp(log_normalized_twisted_mixtures), twisted_mus, sigma_tildes)

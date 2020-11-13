"""
test melange.csmc
"""
from melange.gaussians import *
from melange.csmc import *
import tqdm
from jax import random
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

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
        twisted_logpdf = unnormalized_Normal_logp(x, twisted_mu, twisted_cov) + log_normalizer - twisted_logZ + untwisted_logZ

        #make sure the manual and automatic one are equal
        assert np.isclose(manual_unnorm_logp_twist, twisted_logpdf)

        #now, we have to do the hard part and try to compute the twisted normalization constant...
        unnorm_logp = lambda x: jnp.exp(unnormalized_Normal_logp(x, mean, cov) + log_psi_twist(x, As, bs))
        vals = map(unnorm_logp, jnp.linspace(integration_range[0], integration_range[1], num_point_ops)[..., jnp.newaxis])
        dx = (integration_range[1] - integration_range[0]) / num_point_ops
        man_logZ = jnp.log(dx*vals.sum())
        assert np.isclose(man_logZ, log_normalizer + untwisted_logZ, atol=1e-2)

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

def test_null_do_param_twist(dim=3, num_twists=5, scale=1e3):
    """
    test the do_param_twist function with a null (all zeros) twisting potential. This asserts that the twisted mus, covs are unchanged.
    This also asserts that the log normalizer is zero.
    """
    import tqdm
    for i in tqdm.trange(10):
        base_mu = jnp.array(np.random.randn(dim))*scale
        base_cov = jnp.array(np.abs(np.random.randn(dim)))*scale

        As = jnp.zeros((num_twists, dim))
        bs = jnp.zeros((num_twists, dim))


        twisted_mu, twisted_cov, log_normalizer = do_param_twist(base_mu, base_cov, As, bs, get_log_normalizer=True)
        assert np.allclose(twisted_mu, base_mu)
        assert np.allclose(twisted_cov, base_cov)

        assert np.isclose(log_normalizer, 0.), f"{log_normalizer}"

def test_uncontrolled_csmc(dim=1):
    """
    test a vanilla implementation of uncontrolled sequential monte carlo of a static model with an Euler Maruyama forward/backward kernel.
    the cSMC twisted weights should be the same as the uncontrolled SMC weights
    """
    from melange.tests.utils import get_nondefault_potential_initializer, checker_function
    from melange.smc import generate_trajs, SIS_logW
    from melange.smc_objects import StaticULA

    N,T,Dx = 10000, 1000, dim #define some smc parameters
    potential, (mu, cov), dG = get_nondefault_potential_initializer(Dx) #get the potential and the starting distribution parameters
    cov = cov[0] #rework the covariance matrix so that it is a vector (squashed diagonal)
    dt=1e-2 #timestep

    dummy_A = lambda x, param: jnp.zeros(Dx) #create a dummy A function
    dummy_b = lambda x, param: jnp.zeros(Dx) #create adummy b function

    base_smc_obj = StaticULA(N, potential, potential, potential) #define a static ULA object so we can pull the _compute_log_work_function
    base_twisted_logW_fn = base_smc_obj.log_weights_fn() #pull out the log weight function of the uncontrolled SMC

    smc_obj= StaticULAControlledSMC(N,
                                potential=potential,
                                forward_potential=potential,
                                backward_potential=potential,
                                A_fn = dummy_A, #A_fn=lambda x, p: jnp.zeros((Dx, Dx)),
                                b_fn = dummy_b, #b_fn = lambda x, p: jnp.zeros(Dx),
                                A_params_len=1,
                                b_params_len=1,
                                Dx=1,
                                T=T
                               ) #create a controlled smc class

    prop, logw, (init_Xs, init_logws) = smc_obj.get_fns() #pull the necessary functions from it
    potential_params = jnp.linspace(0,1,T)[..., jnp.newaxis] #define the potential parameters

    prop_params = {
                   'potential_params': potential_params,
                   'forward_potential_params': potential_params,
                   'backward_potential_params': potential_params[1:],
                   'dt': dt
                   } #potential parameters dictionary goes here

    init_params = {'mus': jnp.array([mu]),
                   'covs': jnp.array([cov]),
                   'mixture_weights': jnp.array([1.])} #define the initial params

    model_params=None #there are no model parameters
    y=jnp.arange(T) #the y data aren't important. just know that there are the same number of them as there are data
    rs = random.PRNGKey(5) #define a random number

    #generate trajectories and assert that (with sufficiently long annealing), the initial and final distributions are appropriately placed
    Xs = generate_trajs(prop_params, model_params, y, rs, init_params, init_Xs, prop)
    assert np.isclose(Xs[0,:,:].var(), 2., atol=2e-1)
    assert np.isclose(Xs[-1,:,:].var(), 1., atol=1e-1)

    #compute weights (there is a problem here because twisting should give me constant offsets that i dont want)
    vtwist_fn = smc_obj.vtwist_fn
    untwisted_logW_fn = smc_obj.log_weights_fn()
    twisted_mus, twisted_covs, logK_Zs, (As, bs) = vtwist_fn(Xs[0],
                                                             smc_obj.forward_potential,
                                                             prop_params['dt'],
                                                             prop_params['forward_potential_params'][0],
                                                             smc_obj.A_fn,
                                                             smc_obj.b_fn,
                                                             smc_obj.A_params_cache[:smc_obj.twisting_iteration],
                                                             smc_obj.b_params_cache[:smc_obj.twisting_iteration],
                                                             True)

    # assert all of the logK_Zs are close to zero
    assert np.allclose(logK_Zs, 0.)

    # assert all twists are zeros
    assert np.allclose(As.flatten(), 0.)
    assert np.allclose(bs.flatten(), 0.)

    #_just_ do the twisting potential and make sure it is zero
    log_psi_ts = smc_obj.vlog_psi_twist(Xs[0], As, bs)
    assert np.allclose(log_psi_ts, 0.)

    #compute twisted log weights:
    logWs = SIS_logW(Xs, prop_params, model_params, y, init_params, logw, init_logws)
    assert np.isclose(logsumexp(logWs[-1,:]) - jnp.log(N), dG, atol=1e-1)

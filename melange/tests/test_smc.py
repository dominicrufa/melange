"""
test smc and smc_objects
"""
from jax import random, grad
from jax.scipy.special import logsumexp
from jax import numpy as jnp
from jax import jit

def test_ULA_vSMC_lower_bound():
    """
    test the unadjusted langevin algorithm vSMC lower bound in the near-ais regime (one-dimension) with full resampling _and_ adaptive resampling (nESS threshold of 0.5). the True ELBO must be within 0.1 log units
    """
    from melange.smc_objects import StaticULA
    from melange.smc import vsmc_lower_bound, adapt_vsmc_lower_bound, SIS, og_vsmc_lower_bound, SIS_logZ
    from melange.tests.utils import get_nondefault_potential_initializer
    from melange.tests.utils import checker_function
    from melange.reporters import vSMCReporter

    T = 100
    N=248
    Dx=1
    potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
    potential=potential
    forward_potential = potential
    backward_potential = potential

    #make smc object
    smc_obj = StaticULA(T, N, Dx, potential, forward_potential, backward_potential)


    #make propagation parameters
    potential_params = jnp.linspace(0,1,T)[..., jnp.newaxis]
    forward_potential_params = potential_params
    backward_potential_params = potential_params[1:]
    forward_dts = 1e-2*jnp.ones(T)
    backward_dts=forward_dts[1:]

    prop_params = (potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts)
    model_params=None
    y = jnp.arange(T)
    init_params = (mu, cov, False)
    rs = random.PRNGKey(0)

    reporter = vSMCReporter(T, N, Dx, save_Xs=False)

    #check the logZ
    logZ = vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params)
    adapt_logZ = adapt_vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params, nESS_threshold=0.0)
    SIS_logZ = SIS(prop_params, model_params, y, smc_obj, rs, init_params, reporter)
    og_logZ = og_vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params)
    #ref_logZ = SIS_logZ(reporter.X, prop_params, model_params, y, smc_obj, init_params)

    print(f"true dG: {dG}")
    print(f"logZ: {logZ}; adapt_logZ: {adapt_logZ}")
    print(f"SIS logZ: {SIS_logZ}; og_logZ: {og_logZ}")

    tolerance=0.1
    assert checker_function(logZ - dG, tolerance)
    assert checker_function(adapt_logZ - dG, tolerance)
    assert checker_function(SIS_logZ - dG, tolerance)
    return reporter

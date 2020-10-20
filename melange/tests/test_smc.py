"""
test smc and smc_objects
"""
from jax import random, grad
from jax import numpy as jnp

def test_ULA_vSMC_lower_bound():
    """
    test the unadjusted langevin algorithm vSMC lower bound in the near-ais regime (one-dimension) with full resampling _and_ adaptive resampling (nESS threshold of 0.5). the True ELBO must be within 0.05 log units
    """
    from melange.smc_objects import StaticULA
    from melange.smc import vsmc_lower_bound
    from melange.tests.utils import get_nondefault_potential_initializer
    from melange.tests.utils import checker_function
    from melange.reporters import vSMCReporter

    T = 10
    N=100
    Dx=1
    potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
    potential=potential
    forward_potential = potential
    backward_potential = potential

    #make smc object
    smc_obj = StaticULA(T, N, Dx, potential, forward_potential, backward_potential)

    #make a reporter
    reporter_obj = vSMCReporter(T, N, Dx, save_Xs=True)

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

    #check the logZ
    logZ = vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params, verbose=False, adapt_resamp=False, reporter=reporter_obj)
    adapt_logZ = vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params, verbose=False, adapt_resamp=0.5)
    checker_function(logZ - dG, 0.05)
    checker_function(logZ - dG, 0.05)

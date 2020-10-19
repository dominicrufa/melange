"""
test smc and smc_objects
"""
from jax import random
from jax import numpy as jnp

def test_ULA_vSMC_lower_bound():
    """
    test the unadjusted langevin algorithm vSMC lower bound in the near-ais regine
    """
    from melange.smc_objects import StaticULA
    from melange.smc import vsmc_lower_bound
    from melange.tests.utils import get_nondefault_potential_initializer
    from melange.tests.utils import checker_function

    T = 10
    N=400
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

    logZ = vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params, verbose=False, adapt_resamp=False)
    checker_function(logZ - dG, 0.01)

"""
test smc and smc_objects
"""
from jax import random, grad, jit
from jax.scipy.special import logsumexp
from jax import numpy as jnp
from jax import jit

def test_ULA_SIS():
    """
    test the unadjusted langevin algorithm in SMC and SIS to ensure close agreement with an analytical \log Z_T;
    we propagate 1000 particles 10 steps in R^1 with a nondefault_potential and use a tolerance of 0.05 log units
    """
    from melange.tests.utils import get_nondefault_potential_initializer, checker_function
    from melange.smc import vSIS_lower_bound, vSMC_lower_bound
    from melange.smc_objects import StaticULA

    N,T,Dx = 1000, 10, 1
    pot, (mu, cov), dG = get_nondefault_potential_initializer(1)
    smc_obj= StaticULA(N, potential=pot, forward_potential=pot, backward_potential=pot)
    prop, logw, inits = smc_obj.get_fns()
    potential_params = jnp.linspace(0,1,T)[..., jnp.newaxis]
    prop_params = {'potential_params': potential_params,
                   'forward_potential_params': potential_params,
                   'backward_potential_params': potential_params[1:],
                   'dt': 1e-2
                   }
    init_params = {'mu': mu, 'cov': cov}
    model_params=None
    y=jnp.zeros(T)
    rs = random.PRNGKey(10)

    jSIS = jit(vSIS_lower_bound, static_argnums=(5,6,7))
    jSMC = jit(vSMC_lower_bound, static_argnums=(5,6,7))


    SIS_logZs, SMC_logZs = [], []

    for i in range(100):
        rs, run_rs, run_aux_rs = random.split(rs, 3)
        SIS_logZ =jSIS(prop_params, model_params, y, run_rs, init_params, prop, logw, inits)
        SMC_logZ = jSMC(prop_params, model_params, y, run_aux_rs, init_params, prop, logw, inits)
        SIS_logZs.append(SIS_logZ)
        SMC_logZs.append(SMC_logZ)

    tolerance=0.05
    assert checker_function(jnp.array(SIS_logZs).mean() - dG, tolerance)
    assert checker_function(jnp.array(SMC_logZs).mean() - dG, tolerance)

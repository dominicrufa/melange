"""
variational Sequential Monte Carlo utilities
"""

from jax.scipy.special import logsumexp
from jax.lax import scan, stop_gradient
from jax import grad, vmap
import jax.numpy as jnp

def compute_log_weights(trajectories,
                        ipotential,
                        ipotential_parameters,
                        iforward_potential,
                        iforward_potential_parameters,
                        iforward_dts,
                        ibackward_potential,
                        ibackward_potential_parameters,
                        ibackward_dts):
    """
    arguments
        trajectories : jnp.array(T,M,N)
            trajectories of length T, M particles, each of dimension N
        ipotential : function
            target potential function
        ipotential_parameters : jnp.array(T,Q)
            sequence of potential parameters of the target distributions
        iforward_potential : function
            proposal invariant distribution of forward kernels
        iforward_potential_parameters : jnp.array(T,R)
            proposal invariant distribution parameters of forward kernels
        iforward_dts : jnp.array(T)
            sequence of timesteps
        ibackward_potential : function
            proposal invariant distribution of backward kernels
        ibackward_potential_parameters : jnp.array(T-1,S)
            proposal invariant distribution parameters of backward kernels
        ibackward_dts : jnp.array(T-1)
            sequence of timesteps
    returns
        out : jnp.array(T, M)
            log unnormalized weight matrix
    """
    from melange.miscellaneous import compute_log_pdf_ratio
    from melange.propagators import log_Euler_Maruyma_kernel

    ref = jnp.zeros(3)

    T,N,dim = trajectories.shape
    vcompute_log_pdf_ratio = vmap(compute_log_pdf_ratio, in_axes=(None, None, None, 0,0))
    vEuler_Maruyama_kernel = vmap(log_Euler_Maruyma_kernel, in_axes=(0, 0, None, None, None))
    batched_forward_potential = vmap(iforward_potential, in_axes=(0,None)) #for initial proposal
    batched_potential = vmap(ipotential, in_axes=(0,None)) #for initial weight calculation

    def weight_scanner(prev_log_normalized_weights, t):
        xs_tm1, xs_t = trajectories[t-1], trajectories[t]
        forward_dt = iforward_dts[t]
        backward_dt = ibackward_dts[t-1]
        potential_logps = vcompute_log_pdf_ratio(ipotential,
                                             ipotential_parameters[t-1],
                                             ipotential_parameters[t],
                                             xs_tm1,
                                             xs_t)
        logL = vEuler_Maruyama_kernel(xs_t, xs_tm1, ibackward_potential, ibackward_potential_parameters[t-1], backward_dt[t-1])
        logK = vEuler_Maruyama_kernel(xs_tm1, xs_t, iforward_potential, iforward_potential_parameters[t], forward_dt)
        kernel_logps = logL - logK
        log_weights = prev_log_normalized_weights + jnp.log(N) + potential_logps + kernel_logps
        return stop_gradient(log_weights - logsumexp(log_weights)), log_weights

    ts = jnp.arange(1,T)
    #init_log_weights = -batched_potential(trajectories[0], potential_parameters[0]) + batched_forward_potential(trajectories[0], forward_potential_parameters[0])
    init_log_weights = jnp.zeros(N)
    init_log_norm_weights = stop_gradient(init_log_weights - logsumexp(init_log_weights))
    _, log_weight_matrix = scan(weight_scanner, init_log_norm_weights, ts)
    return jnp.vstack((init_log_weights, log_weight_matrix))


def mod_log_partition_ratio(log_weight_matrix):
    """
    compute log(Z_1/Z_0) from an unnormalized log weight matrix

    arguments
        log_weight_matrix : jnp.array(M,N)
        M, N are the number of iterations and number of particles, respectively
    """
    T, N = log_weight_matrix.shape

    def logZ_scanner(carrier, unnorm_log_weights):
        inc_logZ_T = logsumexp(unnorm_log_weights) - jnp.log(N)
        return None, inc_logZ_T

    _, logZ_incs = scan(logZ_scanner, None, log_weight_matrix)
    return logZ_incs.sum()

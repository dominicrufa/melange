"""
miscellaneous utilities
"""
from jax import numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)

def compute_log_pdf_ratio(potential_function, parameter_tm1, parameter_t, x_tm1, x_t):
    """
    return log( \gamma_{t}(x_t) / \gamma_{tm1}(x_{tm1}))

    arguments
        potential_function : function
            potential function (takes args x and parameters)
        parameter_tm1 : jnp.array()
            second argument of potential that parameterizes the potential (at previous iteration)
        parameter_t : jnp.array()
            second argument of the potential that parameterizes the potential (at the current iteration)
        x_tm1 : jnp.array(N)
            positions (or latent variables) at previous iteration
        x_t : jnp.array(N)
            positions (or latent variables) at the current iteration

    returns
        out : float
            log( \gamma_{t}(x_t) / \gamma_{tm1}(x_{tm1}))
    """
    return potential_function(x_tm1, parameter_tm1) - potential_function(x_t, parameter_t)

def calculate_SIS_nESS(log_weight_matrix):
    """
    calculate the normalized effective sample size give a log weight matrix

    arguments
        log_weight_matrix : jnp.array(T,N)
            log unnormalized weights of N particles over T iterations
    returns
        out : float
            normalized Effective sample size
    """
    from jax.scipy.special import logsumexp
    full_log_weights = log_weight_matrix.sum(0)
    num_particles= log_weight_matrix.shape[1]
    normalized_weights = jnp.exp(full_log_weights - logsumexp(full_log_weights))
    return 1./((normalized_weights**2).sum()*num_particles)

def calculate_SIS_log_partition_ratio(log_weight_matrix):
    """
    calculate log (Z_1 / Z_0) (i.e. -\Delta F) given an unnormalized log weight matrix

    WARNING : this calculation is _only_ valid

    arguments
        log_weight_matrix
    """
    from jax.scipy.special import logsumexp
    num_particles= log_weight_matrix.shape[1]
    out= logsumexp(log_weight_matrix.sum(0)) - jnp.log(num_particles)
    return out

def log_twisted_psi_t(Xp, Xc, A_fn, b_fn, A_params, b_params):
    """
    compute the log twisting psi value

    arguments
        Xp : jnp.array(N)
            previous positions
        Xc : jnp.array(N)
            current positions
        A_fn : function
            matrix twisting function
        b_fn : function
            vector twisting function
        A_params : jnp.array(Q)
            params as argument 1 to A_fn
        b_params : jnp.array(R)
            params as argument 1 to b_fn

    returns
        log_psi : float
            log of twist function
    """
    term1 = -jnp.dot(jnp.dot(Xc, A_fn(Xp, A_params)), Xc)
    term2 = -jnp.dot(Xc, b_fn(Xp, b_params))
    return term1 + term2

def log_twisted_psi0(X0, A0, b0):
    """
    compute the log twisting psi value at t=0

    arguments
        X0 : jnp.array(N)
            starting positions
        A0 : jnp.array(N,N)
            twisting matrix
        b0 : jnp.array(N)
            twisting vector
    returns
        out : float
            log_twisted_psi0
    """
    return -jnp.dot(jnp.dot(X0,A0),X0) - jnp.dot(X0, b0)

def exp_normalize(logWs):
    """
    compute an array of weights given an array of log weights
    """
    b = logWs.max()
    y = jnp.exp(logWs - b)
    return y / y.sum()

"""
miscellaneous utilities
"""
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

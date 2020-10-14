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

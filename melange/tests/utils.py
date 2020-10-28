"""
miscellaneous utilities
"""
from jax import numpy as jnp
import numpy as np
import jax
from jax.lax import scan

def default_potential(pos, parameter): #define the potential
    """
    default potential anneals between a 1d gaussian centered at 0 and a variance of 0.5 to a 1d gaussian centered at 0 and a variance of (still) 0.5;
    the free energy is identically 0.

    arguments
        pos : jnp.array(N)
            position 1D array
        parameter : jnp.array(1)
            parameter from [0] to [1]; (lives inside an array) for batching purposes

    returns
        float : value of reduced potential energy
    """
    mod_parameter = parameter[0]
    x0 = jnp.ones(pos.shape)
    return (1. - mod_parameter)*jnp.dot(pos, pos) + mod_parameter*jnp.dot(pos - x0, pos - x0)

def nondefault_gaussian_trap_potential(pos, parameter):
    """
    nondefault potential that will anneal between two gaussians (at lambda=0,1)
    at lambda=0: gaussian has a variance of 2
    at lambda=1: gaussian has a variance of 1
    logZ_T is approx -0.3465 (please check this again)

    arguments
        pos : jnp.array(N)
            position 1D array
        parameter : jnp.array(1)
            parameter from [0] to [1]; (lives inside an array) for batching purposes

    returns
        float : value of reduced potential energy

    """
    mod_parameter = parameter[0]
    dim = pos.shape[0]
    k0, k1 = 0.5 * jnp.diag(jnp.ones(dim)), jnp.diag(jnp.ones(dim))
    return 0.5*((1. - mod_parameter) * jnp.dot(jnp.dot(pos, k0), pos) + mod_parameter*jnp.dot(jnp.dot(pos, k1), pos))

def get_default_potential_initializer(dimension):
    """
    helper function to return a tuple of defaults

    arguments
        dimension : int
            dimension of the latent variable

    returns
        default_potential
            mean (at lambda=0)
        cov (at lambda=1)
            free_energy (-logZ_T)
    """
    potential = default_potential
    mu, cov = jnp.zeros(dimension), jnp.diag(jnp.ones(dimension))*0.5
    dG = 0.
    return potential, (mu, cov), dG

def get_nondefault_potential_initializer(dimension):
    """
    helper function to return a tuple of nondefaults

    arguments
        dimension : int
            dimension of the latent variable

    returns
        default_potential
        mean (at lambda=0)
        cov (at lambda=1)
        free_energy (-logZ_T)
    """
    potential = nondefault_gaussian_trap_potential
    mu, cov = jnp.zeros(dimension), jnp.diag(jnp.ones(dimension))*2.
    dG = -0.3465
    return potential, (mu, cov), dG

def checker_function(value, tolerance):
    if abs(value) < tolerance:
        return True
    else:
        return False

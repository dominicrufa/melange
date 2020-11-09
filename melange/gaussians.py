"""
some Gaussian utilities
"""
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
from jax import vmap

def square_mahalanobis(u,v,VI):
    """
    compute the square mahalonobis distance of two vectors u, v w.r.t an inverse matrix (VI)
    delta = u - v
    computed as dot( dot(delta, VI), delta )

    arguments
        u : jnp.array(N)
        v : jnp.array(N)
        VI : jnp.array(N,N)

    returns
        out : float
            the square of the mahalonobis distance
    """
    delta = u - v
    out = jnp.dot(jnp.dot(delta, VI), delta)
    return out

def multivariate_gaussian_logp(x, mu, cov):
    """
    compute the log probability of a multivariate gaussian given x, mu, and a covariance matrix
    arguments
        x : jnp.array(N)
            position in latent space
        mu : jnp.array(N)
            mean of Gaussian
        cov : jnp.array(N,N)
            covariance of gaussian
    returns
        logp : float
            log probability of N(x | mu, cov)
    """
    from jax.lax_linalg import cholesky, triangular_solve
    n = mu.shape[0]
    L = cholesky(cov)
    y = triangular_solve(L, x - mu, lower=True, transpose_a=True)
    return -1./2. * jnp.einsum('...i,...i->...', y, y) - n/2.*jnp.log(2*jnp.pi) - jnp.log(L.diagonal()).sum()

def sample_gmm(key, weights, mus, covs):
    """
    sample a gaussian mixture model

    arguments
        key : random.PRNGKey
            key to sample
        weights : jnp.array(N)
            mixture weights
        mus : jnp.array(J,N)
            mean vectors
        covs : jnp.array(J, N, N)
            covariance vectors

    returns
        mixture_id : int
            mixture index
        out : jnp.array(N)
            sampled value
    """
    mix_key, normal_key = random.split(key)
    mixture_idx = random.choice(mix_key, len(weights), p=weights)
    return mixture_idx, random.multivariate_normal(key = normal_key, mean = mus[mixture_idx], cov = jnp.diag(covs[mixture_idx]))

"""
Normal-twiting utilities
"""
def Normal_logZ(mu, cov): #tested
    """
    compute the log normalizing constant of a normal distribution given a mean vector and a covariance *vector

    arguments
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector

    returns
        logZ : float
            log normalization constant
    """
    dim = len(mu)
    logZ = 0.5*dim*jnp.log(2.*jnp.pi) + 0.5 * jnp.log(cov).sum()
    return logZ

vNormal_logZ = vmap(Normal_logZ, in_axes=(0, 0))

def twist_log_constant(mu, cov, A, b):
    """
    compute the auxiliary twisting log constant

    arguments
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector
        A : jnp.array(Dx)
            A twisting vector
        b : jnp.array(Dx)
            b twisting vector

    returns
        out : float
            twisting log constant
    """
    term1 = -(mu/(2.*cov)).dot(mu)
    l_r_terms = mu/cov - b
    inner = l_r_terms / (2./cov + 4*A)
    return term1 + inner.dot(l_r_terms)

def unnormalized_Normal_logp(x, mu, cov): #tested
    """
    compute an unnormalized gaussian logp

    arguments
        x : jnp.array(Dx)
            position
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector

    returns
        out : float
            unnormalized gaussian logp
    """
    delta = x-mu
    return -0.5*(delta/cov).dot(delta)

"""
some Gaussian utilities
"""
from jax import numpy as jnp
from jax import scipy as jsp

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
    inv_cov = jsp.linalg.inv(cov)
    k = x.shape[0]
    logp = -0.5 * square_mahalanobis(x,mu, inv_cov) - 0.5 * (k * jnp.log(2.*jnp.pi) + jnp.log(jsp.linalg.det(cov)))
    return logp

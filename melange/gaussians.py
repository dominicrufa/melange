"""
some Gaussian utilities
"""
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random

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

def get_twisted_gmm(mix_weights, mus, covs, A, b):
    """
    twist a gaussian mixture model with matrix A and vector b

    arguments
        mix_weights : jnp.array(J)
            mixture weights (normalized)
        mus : jnp.array(J, N)
            mean vectors
        covs : jnp.array(J, N, N)
            covariance matrices
        A : jnp.array(N,N)
            twisting matrix
        b : jnp.array(N)
            twisting vector

    returns
        log_normalizer : float
            log normalization constant of GMM
        log_alpha_tildes : jnp.array(N)
            twisted mixture weights
        sigma_tildes : jnp.array(J, N, N)
            twisted covariance matrices
        tup : tuple
            twisted_mus : jnp.array(J, N)
                twisted mean vectors
            sigma_tildes : above
    """
    from jax.lax import map
    from jax.scipy.special import logsumexp
    from melange.miscellaneous import exp_normalize
    num_mixtures, dim = mus.shape
    sigma_tildes = map(lambda x: jnp.linalg.inv(jnp.linalg.inv(x) + 2.*A), covs)
    log_zetas = jnp.array([-0.5 * jnp.linalg.det(cov) + 0.5 * jnp.linalg.det(sigma_tilde) + 0.5 * square_mahalanobis(jnp.matmul(jnp.linalg.inv(cov), mu), b, sigma_tilde) - 0.5 * square_mahalanobis(mu, jnp.zeros(dim), jnp.linalg.inv(cov)) for mu, cov, sigma_tilde in zip(mus, covs, sigma_tildes)])
    log_alpha_zetas = jnp.log(mix_weights) + log_zetas
    log_normalizer = logsumexp(log_alpha_zetas)
    log_alpha_tildes = log_alpha_zetas - log_normalizer

    twisted_mus = jnp.array([jnp.matmul(sigma_tilde, jnp.matmul(jnp.linalg.inv(sigma), mu) - b) for sigma_tilde, sigma, mu in zip(sigma_tildes, covs, mus)])
    return log_normalizer, log_alpha_tildes, sigma_tildes, (twisted_mus, sigma_tildes)

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
    assert jnp.isclose(weights.sum(), 1.)
    mixture_idx = random.choice(mix_key, len(weights), p=weights)
    return mixture_idx, random.multivariate_normal(key = normal_key, mean = mus[mixture_idx], cov = covs[mixture_idx])

def logK_ints(b, f, theta, dt):
    """
    we have to be able to compute K_t(\hat{\psi}_t^i)(x_{t-1});
    for ULA kernels with a driving potental of the form \psi_t = exp{-x_t^T*A_t(x_{t-1})*x_t^T - x_t^T*b_t(x_{t-1})},
    we get K_t(\hat{\psi}_t^i)(x_{t-1})

    arguments
        b : jnp.array(N)
            twisting vector
        f : jnp.array(N)
            push vector
        theta : jnp.array(N,N)
            twist matrix
        dt : float
            time increment

    return
        logK_int : float
            twisted kernel integral
    """
    logK_int = 0.5*jnp.linalg.det(theta) + (0.5/dt)*square_mahalanobis(f, dt*b, theta) - (0.5/dt)*jnp.dot(f, f)
    return logK_int

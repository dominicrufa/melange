"""
smc managers
"""
from jax import random
from jax import numpy as jnp
from jax.lax import stop_gradient, fori_loop
from jax.scipy.special import logsumexp
import numpy as np

def resampling(w, rs):
    """
    Stratified resampling with "stop_gradient" to ensure autograd
    takes no derivatives through it.
    """
    N = w.shape[0]
    bins = jnp.cumsum(w)
    ind = jnp.arange(N)
    u = (ind  + random.uniform(rs, shape=[N]))/N

    return stop_gradient(jnp.digitize(u, bins))

def vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, init_params, verbose=False, adapt_resamp=False, reporter=None):
    """
    Estimate the VSMC lower bound. Amenable to (biased) reparameterization
    gradients.
    .. math::
        ELBO(\theta,\lambda) =
        \mathbb{E}_{\phi}\left[\nabla_\lambda \log \hat p(y_{1:T}) \right]

    Requires an SMC object with 2 member functions:
    -- sim_prop(t, x_{t-1}, y, prop_params, model_params, rs)
    -- log_weights(t, x_t, x_{t-1}, y, prop_params, model_params)
    """

    #handle adaptive resampling
    if type(adapt_resamp) == float:
        adapt_resamp=True
        ESS_threshold=adapt_resamp
    else:
        ESS_threshold=None

    #handle reporter
    report = True if reporter is not None else False

    # Extract constants
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    # Initialize SMC
    rs, init_rs = random.split(rs)
    X, logW = smc_obj.initialize(prop_params, init_rs, init_params)
    max_logW = jnp.max(logW)
    Xp = X
    W = jnp.exp(logW - logsumexp(logW))
    logZ = 0.
    ESS = 1./jnp.sum(W**2)/N

    if report:
        reporter.report(0, (X, logZ, ESS))

    for t in jnp.arange(1,T):
        # Resampling
        if adapt_resamp:
            if ESS < ESS_threshold:
                rs, resample_rs = random.split(rs)
                ancestors = resampling(W, resample_rs)
                Xp = X[ancestors]
                logZ = logZ + max_logW + jnp.log(jnp.sum(W)) - jnp.log(N)
                logW = jnp.zeros(N)
            else:
                Xp = X
        else:
            if t > 0:
                rs, resample_rs = random.split(rs)
                ancestors = resampling(W, resample_rs)
                Xp = X[ancestors]
            else:
                Xp = X

        # Propagation
        rs, prop_rs = random.split(rs)
        X = smc_obj.sim_prop(t, Xp, y, prop_params, model_params, prop_rs)

        # Weighting
        if adapt_resamp:
            logW = logW + smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        else:
            logW = smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)

        max_logW = jnp.max(logW)
        W = jnp.exp(logW-max_logW)
        if adapt_resamp:
            if t == T-1:
                logZ = logZ + max_logW + jnp.log(jnp.sum(W)) - jnp.log(N)
        else:
            logZ = logZ + max_logW + jnp.log(jnp.sum(W)) - jnp.log(N)
        W = W / jnp.sum(W)
        ESS = 1./jnp.sum(W**2)/N

        if report:
            reporter.report(t, (X, logZ, ESS))

    return logZ

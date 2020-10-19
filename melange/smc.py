"""
smc managers
"""
from jax import random
from jax import numpy as jnp
from jax.lax import stop_gradient

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

def vsmc_lower_bound(prop_params, model_params, y, smc_obj, rs, verbose=False, adapt_resamp=False):
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
    # Extract constants
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    # Initialize SMC
    X = np.zeros((N,Dx))
    Xp = np.zeros((N,Dx))
    logW = np.zeros(N)
    W = np.exp(logW)
    W /= np.sum(W)
    logZ = 0.
    ESS = 1./np.sum(W**2)/N

    for t in range(T):
        # Resampling
        if adapt_resamp:
            if ESS < 0.5:
                ancestors = resampling(W, rs)
                Xp = X[ancestors]
                logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
                logW = np.zeros(N)
            else:
                Xp = X
        else:
            if t > 0:
                ancestors = resampling(W, rs)
                Xp = X[ancestors]
            else:
                Xp = X

        # Propagation
        X = smc_obj.sim_prop(t, Xp, y, prop_params, model_params, rs)

        # Weighting
        if adapt_resamp:
            logW = logW + smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        else:
            logW = smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        max_logW = np.max(logW)
        W = np.exp(logW-max_logW)
        if adapt_resamp:
            if t == T-1:
                logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
        else:
            logZ = logZ + max_logW + np.log(np.sum(W)) - np.log(N)
        W /= np.sum(W)
        ESS = 1./np.sum(W**2)/N
    if verbose:
        print('ESS: '+str(ESS))
    return logZ

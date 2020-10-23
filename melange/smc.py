"""
smc managers
"""
from jax import random, jit
from jax import numpy as jnp
from jax.lax import stop_gradient, scan, cond
from jax.scipy.special import logsumexp
import numpy as np
from melange.reporters import BaseSMCReporter
import jax

def resampling(w, rs):
    """
    Stratified resampling with "stop_gradient" to ensure autograd
    takes no derivatives through it.

    arguments
        w : jnp.array(N)
            particle weights
        rs : jax.random.PRNGKey
            random key
    """
    N = w.shape[0]
    bins = jnp.cumsum(w)
    ind = jnp.arange(N)
    u = (ind  + random.uniform(rs, shape=[N]))/N

    return stop_gradient(jnp.digitize(u, bins))

def nESS(logW):
    """
    compute a normalized effective sample size

    arguments
        logW : jnp.array(N)
            log particle weights
    """
    N = len(logW)
    return 1./jnp.sum(jnp.exp(logW - logsumexp(logW))**2)/N


def SMC(prop_params, model_params, y, rs, init_params, prop_fn, logW_fn, init_fns):
    """
    conduct SMC (with stratified resampling) and return logZ, Xs

    arguments
        prop_params : pytree
            propagation parameters passed to the prop_fn
        model_params : pytree
            model parameters passed to the prop_fn
        y : jnp.array(T, R)
            data of sequence T, dimension R
        rs : jax.random.PRNGKey
            random key
        init_params : pytree
            initialization params passed to init_fns[0]
        prop_fn : function
            propagation function
        logW_fn : function
            compute the log weights of particles
        init_fns : tuple
            init_Xs_fn : function
                initialize Xs
            init_logW_fn : function
                initialize logW

    returns
        out_logZ : float
            log normalization
        latents : jnp.array(T,N,Dx)
            latent variable trajectories
    """
    T = y.shape[0]

    init_Xs_fn, init_logW_fn = init_fns #separate the X initilaizer and logW initializer

    #initialize
    rs, init_rs = random.split(rs) #split the rs
    X0 = init_Xs_fn(prop_params, init_rs, init_params) #initialize N Xs with the random seed and the propagation parameters
    potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params #split the prop params
    init_logWs = init_logW_fn(X0, prop_params) #initialize the logWs
    logZ = 0. #initialize the logZ
    X = X0 #set X to the init positions
    N = len(init_logWs)

    def scanner(carry, t): #build a scanner
        X, logW, logZ, rs = carry #separate the carrier
        rs, resample_rs = random.split(rs) #split the randoms to resample

        # resample
        W = jnp.exp(logW - logsumexp(logW)) #define the works
        ancestors = resampling(W, resample_rs) #take ancestors and resample
        Xp = X[ancestors] #resample

        # Propagate
        rs, prop_rs = random.split(rs) # split the randoms to propagate
        X = prop_fn(t, Xp, y, prop_params, model_params, prop_rs) #propagate with the resampled particles

        # weighting
        logW = logW_fn(t, Xp, X, y, prop_params, model_params) #define the log weights

        # Update logZ, ESS
        logZ = logZ + logsumexp(logW) - jnp.log(N) #update the logZ

        return (X, logW, logZ, rs), X #return positions, logW, logZ, randoms, and the Xs to collect

    (out_X, out_logW, out_logZ, rs), out_Xs = scan(scanner, (X, init_logWs, logZ, rs), jnp.arange(1,T)) #run scan
    return out_logZ, jnp.vstack((X0[jnp.newaxis, ...], out_Xs)) # return the final logZ, and the stacked positions

def vSMC_lower_bound(prop_params, model_params, y,  rs, init_params, prop_fn, logW_fn, init_fns):
    """
    conduct SMC (with stratified resampling) and return logZ
    """
    logZ, Xs = SMC(prop_params, model_params, y,  rs, init_params, prop_fn, logW_fn, init_fns)
    return logZ

def SIS(prop_params, model_params, y,  rs, init_params, prop_fn, logW_fn, init_fns):
    """
    conduct sequential importance sampling (no resampling) and return Xs, logWs
    """
    T = y.shape[0]

    init_Xs_fn, init_logW_fn = init_fns

    #make trajs
    Xs = generate_trajs(prop_params, model_params, y, rs, init_params, init_Xs_fn, prop_fn)

    #make cum weight matrix
    logWs = SIS_logW(Xs, prop_params, model_params, y, logW_fn, init_logW_fn)

    return Xs, logWs

def vSIS_lower_bound(prop_params, model_params, y,  rs, init_params, prop_fn, logW_fn, init_fns):
    """
    conduct sequential importance sampling (no resampling) and return logZ
    """
    T = y.shape[0]

    init_Xs_fn, init_logW_fn = init_fns

    #make trajs
    Xs = generate_trajs(prop_params, model_params, y, rs, init_params, init_Xs_fn, prop_fn)

    #make cum weight matrix
    logWs = SIS_logW(Xs, prop_params, model_params, y, logW_fn, init_logW_fn)
    N = logWs.shape[1]

    return logsumexp(logWs[-1,:]) - jnp.log(N)

def generate_trajs(prop_params, model_params, y, rs, init_params, init_X_fn, prop_fn):
    """
    generate trajectories of length T, N
    """
    T = y.shape[0]

    #initialize
    rs, init_rs = random.split(rs)
    X0 = init_X_fn(prop_params, init_rs, init_params)

    def scanner(carry, t):
        Xp, rs = carry
        rs, run_rs = random.split(rs)
        X = prop_fn(t, Xp, y, prop_params, model_params, run_rs)
        return (X, rs), X

    _, Xs = scan(scanner, (X0, rs), jnp.arange(1,T))
    return jnp.vstack((X0[jnp.newaxis, ...], Xs))

def SIS_logW(Xs, prop_params, model_params, y, logW_fn, init_logW_fn):
    """compute the logW of an ensemble of SIS trajectories"""
    T,N,Dx = Xs.shape
    potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params

    init_logWs = init_logW_fn(Xs[0], prop_params)
    def scanner(none, t):
        Xp, Xc = Xs[t-1], Xs[t]
        return None, logW_fn(t, Xp, Xc, y, prop_params, model_params)

    _, logW_matrix = scan(scanner, None, jnp.arange(1,T))
    return jnp.cumsum(jnp.vstack((init_logWs, logW_matrix)), axis=0)

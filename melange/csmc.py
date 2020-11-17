"""
smc objects and utilities for controlled SMC
"""
from melange.smc_objects import StaticULA
from melange.gaussians import *
import jax.numpy as jnp
from functools import partial
from jax import random
from jax import grad, vmap, jit
from jax.lax import map, scan, cond
from jax.scipy.special import logsumexp
from jax.config import config; config.update("jax_enable_x64", True)
import logging

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("csmc")
_logger.setLevel(logging.INFO)

"""twisting utilities"""
def twisted_normal_params(mu, cov, A, b, get_log_twist_constant=True): #tested
    """
    given a gaussian distribution with mu and cov, multiply this kernel by e^{-x^T A(x_tm1) x - x^T b(x_tm1)},
    get the resultant mu, cov, and if specified, the log integral of the forward kernel

    arguments
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector
        A : jnp.array(Dx)
            A twisting vector
        b : jnp.array(Dx)
            b twisting vector
        get_log_twist_constant : bool, default True
            whether to compute the log twisting constant values

    returns
        twisted_mu : jnp.array(Dx)
            twisted mean
        twisted_cov : jnp.array(Dx)
            twisted covariance
        log_twist_constant : float
            log twisting constant of twists
    """
    twisted_mu = ( mu/cov - b) / ( 1/cov + 2*A )
    twisted_cov = 1 / ( 1/cov + 2*A )
    log_twist_constant = cond(get_log_twist_constant,
                lambda x: twist_log_constant(*x),
                lambda x: 0.,
                (mu, cov, A, b)
               )
    return (twisted_mu, twisted_cov, log_twist_constant)

vtwisted_normal_params = vmap(twisted_normal_params, in_axes=(0, 0, None, None, None))

def base_fk_params(X, potential, dt, potential_parameters):
    """
    generate base forward kernel parameters with an Euler Maruyama proposal

    arguments
        x : jnp.array(Dx)
            position
        potential : function
            potential function
        dt : float
            time increment
        potential_parameters : jnp.array(R)
            arg 1 for potential function

    returns
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector
    """
    from melange.propagators import EL_mu_sigma
    mu, cov = EL_mu_sigma(X, potential, dt, potential_parameters)
    return mu, jnp.diag(cov)

def twist_scanner_fn(carry, x):
    """auxiliary twisting scanner function"""
    start_mu, start_cov, init_log_twist_constant, get_log_twist_constant = carry
    A = x[0,:]
    b = x[1,:]
    new_mu, new_cov, log_twist_constant = twisted_normal_params(start_mu, start_cov, A, b, get_log_twist_constant)
    return (new_mu, new_cov, init_log_twist_constant + log_twist_constant, get_log_twist_constant), None

def do_param_twist(base_mu, base_cov, As, bs, get_log_normalizer):
    """
    get the twisted gaussian params and the log normalizer

    arguments
        base_mu : jnp.array(Dx)
            mean vector
        base_cov : jnp.array(Dx)
            covariance vector
        As : jnp.array(Q, Dx)
            array of A twisting vectors
        bs : jnp.array(Q, Dx)
            array of b twisting vectors
        get_log_normalizer : bool
            whether to get a log normalizing constant for a twisted forward kernel

    return
        twisted_mu : jnp.array(Dx)
            twisted mean
        twisted_cov : jnp.array(Dx)
            twisted covariance vector
        logZ : float
            forward kernel log normalizer
    """
    A_b_merger = jnp.stack((As, bs), axis=1) #combine A and b parameters to push through the
    init = (base_mu, base_cov, 0., get_log_normalizer) #wrap an initial tuple for the scanner
    (twisted_mu, twisted_cov, log_twist_constant, out_bool), _ = scan(twist_scanner_fn, init, A_b_merger) #run the scan

    #compute a normalized forward kernel of the base gaussian:
    base_logZ = Normal_logZ(base_mu, base_cov)
    logZ = cond(get_log_normalizer, lambda x: Normal_logZ(*x) - base_logZ, lambda x: 0., (twisted_mu, twisted_cov)) # compute the logZ of the kernel if specified
    return twisted_mu, twisted_cov, logZ + log_twist_constant

def gaussians_twist(Xp, potential, dt, potential_params, A_fn, b_fn, A_params, b_params, get_log_normalizer):
    """
    conduct a gaussian twist to return a twisted mean, covariance vector, normalizing constant, As, and bs

    arguments
        Xp : jnp.array(Dx)
            previous x values
        potential : function
            potential function
        dt : float
            time increment
        potential_parameters : jnp.array(R)
            arg 1 for potential function
        A_fn : function
            A twisting function
        b_fn : function
            b twisting function
        A_params : jnp.array(U, V)
            U vector of parameters to twist; each is of dim V
        b_params : jnp.array(U, W)
            U vector of parameters to twist; each is of dim W
        get_log_normalizer : bool
            whether to compute the normalization constant of the forward kernel
    return
        twisted_mu : jnp.array(Dx)
            twisted mean
        twisted_cov : jnp.array(Dx)
            twisted covariance vector
        logZ : float
            forward kernel log normalizer
        As : jnp.array(Q, Dx)
            array of A twisting vectors
        bs : jnp.array(Q, Dx)
            array of b twisting vectors
    """
    partial_A_fn, partial_b_fn = partial(A_fn, Xp), partial(b_fn, Xp)
    As = map(partial_A_fn, A_params)
    bs = map(partial_b_fn, b_params)
    start_mu, start_cov = base_fk_params(Xp, potential, dt, potential_params)
    twisted_mu, twisted_cov, logZ = do_param_twist(start_mu, start_cov, As, bs, get_log_normalizer)
    return twisted_mu, twisted_cov, logZ, (As, bs)

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
        A : jnp.array(N)
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
    twisted_mus, twisted_covs, log_twist_constants = vtwisted_normal_params(mus, covs, A, b, True)
    untwisted_logZs = vNormal_logZ(mus, covs)
    twisted_logZs = vNormal_logZ(twisted_mus, twisted_covs)
    log_mix_normalizers = twisted_logZs + log_twist_constants - untwisted_logZs

    log_mixture_twists = jnp.log(mix_weights) + log_mix_normalizers
    full_log_normalizer = logsumexp(log_mixture_twists)
    log_normalized_twisted_mixtures = log_mixture_twists - full_log_normalizer

    return full_log_normalizer, log_normalized_twisted_mixtures, (twisted_mus, twisted_covs)


"""
gaussian twisting potentials

log_psi_twist_A and log_psi_twist_b are
"""
def log_psi_twist_A(x, As):
    """
    compute the log twisted potential of the A values
    arguments
        x : jnp.array(Dx)
            position
        As : jnp.array(Q, Dx)
            Q covariance twisting matrices

    returns
        out : float
            negative square mahalonobis distance
    """
    return -(x*(As.sum(axis=0))).dot(x)

def log_psi_twist_b(x, bs):
    """
    compute the log twisted potential of the b values

    arguments
        x : jnp.array(Dx)
            position
        bs : jnp.array(Q, Dx)
            Q mean twisting vector

    returns
        out : float
            twisted b potential
    """
    return -x.dot(bs.sum(axis=0))

def log_psi_twist(x, As, bs):
    """
    compute the full log twist

    arguments
        x : jnp.array(Dx)
            position
        As : jnp.array(Q, Dx)
            Q mean twisting matrices (diagonal)
        bs : jnp.array(Q, Dx)
            Q mean twisting vector

    return
        out : float
            full twist
    """
    return log_psi_twist_A(x, As) + log_psi_twist_b(x, bs)

"""
define a vlog_psi_twist to vectorize all of the log_psi_twist inputs

    arguments
        xs : jnp.array(N, Dx)
            vectorized positions
        N_As : jnp.array(N, Q, Dx)
            vectorized covariance twists
        N_bs : jnp.array(N, Q, Dx)
            vectorized twisting vectors
    return
        out : jnp.array(N)
"""
vlog_psi_twist = vmap(log_psi_twist, in_axes=(0,0,0))

#this util is not used directly
vlog_psi_twist_util = vmap(log_psi_twist, in_axes=(0,None, None))

def vlog_psi_twist_single(xs, A, b):
    """
    do a twist on N different positions with a single A and b vector

    arguments
        xs : jnp.array(N, Dx)
            vectorized positions
        A : jnp.array(Dx)
            vectorized covariance twists
        b : jnp.array(Dx)
            vectorized twisting vectors
    return
        out : jnp.array(N)

    """
    return vlog_psi_twist_util(xs, A[jnp.newaxis, ...], b[jnp.newaxis, ...])

#########################################
#####CSMC Classes########################
#########################################

class StaticULAControlledSMC(StaticULA):
    """
    controlled SMC handler for Static model wth an Unadjusted Langevin Algorithm

    prop_params is a dict containing: ['potential_params', 'forward_potential_params', 'backward_potential_params', 'dt'];
    init_params is a dict containing: ['mixture_weights', 'mus', 'covs', 'A0', 'b0']

    NOTE: in the future, i will generalize this to State Space models,as well
    NOTE: while we do handle the generalized multivariate case here, it is worth ensuring that covariances will be passed as 1D vectors (only diagonal elements)
    """
    def __init__(self,
                 *args, # N, potential, forward_potential, backward_potential
                 A_fn,
                 b_fn,
                 A_params_len,
                 b_params_len,
                 Dx,
                 T,
                 max_twists = 10,
                 **kwargs):
        _logger.debug(f"instantiating super")
        super().__init__(*args, **kwargs) #init the og args

        _logger.debug(f"getting x dimension")
        self.Dx = Dx

        _logger.debug(f"getting the terminal time")
        self.T = T

        #define the twisting functions
        _logger.debug(f"define A, b functions")
        self.A_fn = A_fn
        self.b_fn = b_fn

        #vmap the A, b functions; this is a special utility to compute As, bs of a single twist for N particle positions
        self.vA_fn = vmap(A_fn, in_axes=(0,None))
        self.vb_fn = vmap(b_fn, in_axes=(0,None))

        #define twisting iterations
        _logger.debug(f"define twisting parameters and parameter caches")
        self.twisting_iteration = 1
        self.max_twists = max_twists
        self.A_params_cache = jnp.zeros((max_twists, T, A_params_len))
        self.b_params_cache = jnp.zeros((max_twists, T, A_params_len))
        self.A0, self.b0 = jnp.zeros(self.Dx), jnp.zeros(self.Dx)

        #assert parameter lengths are the same
        assert A_params_len == b_params_len, f"the A and b parameter lengths must be the same"
        self.A_params_len = A_params_len
        self.b_params_len = b_params_len

        #define the psi twisting function
        _logger.debug(f"define psi twisting functions")
        self.vlog_psi_twist = vmap(log_psi_twist, in_axes=(0,0,0))
        self.twist_fn = gaussians_twist
        self.vtwist_fn = vmap(self.twist_fn, in_axes=(0, None, None, None, None, None, None, None, None)) #args (Xp, potential, dt, potential_params, A_fn, b_fn, A_params, b_params, get_log_normalizer)


    def sim_prop_fn(self):
        _logger.debug(f"generating simulation propagation function")
        self.vpropagator = vmap(gaussian_proposal, in_axes=(0,0,0))

        def prop(t, Xp, y, prop_params, model_params, rs):
            N, Dx = Xp.shape #extract the number of particles and the dimension of x
            folder_rs = random.split(rs, num=N+1) #create a list of random keys
            new_rs, runner_rs = folder_rs[0], folder_rs[1:] #separate the random keys

            twisted_mus, twisted_covs, _, _ = self.vtwist_fn(Xp,
                                                          self.forward_potential,
                                                          prop_params['dt'],
                                                          prop_params['forward_potential_params'][t],
                                                          self.A_fn,
                                                          self.b_fn,
                                                          self.A_params_cache[:self.twisting_iteration,t],
                                                          self.b_params_cache[:self.twisting_iteration,t],
                                                          False) # twisted_mu, twisted_cov, logZ, (As, bs)


            Xs = self.vpropagator(runner_rs, twisted_mus, twisted_covs)
            return Xs
        return prop

    def log_weights_fn(self):
        _logger.debug(f"generating the log weight calculator")
        #define vmapped twisting functions and logK_integral functions
        static_ula_logW_fn = super().log_weights_fn() #get the super static ula log_weights_fn

        def log_weights(t, Xp, Xc, y, prop_params, model_params):
            """
            arguments
                prop_params : tuple
                    potential_params : jnp.array(T,Q)
                    forward_potential_params : jnp.array(T,R)
                    backward_potential_params : jnp.array(T-1,S)
                    forward_dts : jnp.array(T)
                    backward_dts : jnp.array(T-1)

            NOTE : need to implement a conditional here. specifically, if t == T, then we do _not_ need to compute the logK_Zs. we need only the untwisted incremental weight and the
                twisting potential
            """
            # compute untwisted logWs
            logWs = static_ula_logW_fn(t, Xp, Xc, y, prop_params, model_params)

            #compute twisting logZ forward kernel and psi_twist
            compute_logK_Z = cond(t == self.T, lambda x: False, lambda x: True, None)
            _, _, logK_Zs, (As, bs) = self.vtwist_fn(Xp,
                                                 self.forward_potential,
                                                 prop_params['dt'],
                                                 prop_params['forward_potential_params'][t],
                                                 self.A_fn,
                                                 self.b_fn,
                                                 self.A_params_cache[:self.twisting_iteration,t],
                                                 self.b_params_cache[:self.twisting_iteration,t],
                                                 compute_logK_Z)

            #compute twisting functions
            log_psi_ts = self.vlog_psi_twist(Xc, As, bs)

            #build modifier
            return logWs + logK_Zs - log_psi_ts
        return log_weights

    def initialize_Xs_fn(self):
        """
        initialize Xs
        """
        _logger.debug(f"generating SMC variable initializer function")
        from melange.miscellaneous import exp_normalize
        self.vsample_gmm = vmap(sample_gmm, in_axes = (0, None, None, None)) # args: (key, weights, mus, covs)


        def init_xs(prop_params, rs, init_params):
            #resolve random keys
            folder_rs = random.split(rs, self.N+1)
            rs, run_rs = folder_rs[0], folder_rs[1:]

            # conduct twisted initialization full_log_normalizer, log_normalized_twisted_mixtures, (twisted_mus, twisted_covs)
            _, log_normalized_twisted_mixtures, (twisted_mus, twisted_covs) = get_twisted_gmm(init_params['mixture_weights'],
                                                                                  init_params['mus'],
                                                                                  init_params['covs'],
                                                                                  self.A0,
                                                                                  self.b0) #compute twisting parameters

            _, Xs = self.vsample_gmm(run_rs, exp_normalize(log_normalized_twisted_mixtures), twisted_mus, twisted_covs)
            return Xs
        return init_xs

    def initialize_logW_fn(self, **kwargs):
        _logger.debug(f"generating SMC log weight initializer function")

        def init_logWs(X, init_params, prop_params):
            log_normalizer, _, _ = get_twisted_gmm(init_params['mixture_weights'],
                                                   init_params['mus'],
                                                   init_params['covs'],
                                                   self.A0,
                                                   self.b0) #compute twisting parameters

            # compute twisted weights
            log_psi0s = log_normalizer - vlog_psi_twist_single(X, self.A0, self.b0)

            return log_psi0s
        return init_logWs

    def get_ADP_fn(self):
        """
        get the functions needed to do approximate dynamic programming
        # TODO : add constraints for the twisted matrices (have to be positive definite)
        """
        _logger.debug(f"generating approximate dynamic programming functions")
        from jax.ops import index, index_add, index_update
        import tqdm
        import numpy as np

        def sum_square_diffs(Xcs, As, bs, V_bars):
            """
            compute the sum of square differences between the potential (-vlog_psi_twist) and the precomputed V_bars

            arguments
                Xcs : jnp.array(N, Dx)
                    current positions
                As : jnp.array(N, Dx)
                    A twists
                bs : jnp.array(N, Dx)
                    b twists
                V_bars : jnp.array(N)
                    previously-computed potentials

            returns
                out : float
                    sum of square differences

            NOTE : vlog_psi_twist takes As, bs of dim (N, Q, Dx), but the As, bs have dim (N, Dx);
                    since we are summing over Q=1, we have to expand the dimension of the As, bs at axis 1
            """
            return (
                    (-self.vlog_psi_twist(Xcs,
                                          jnp.expand_dims(As, axis=1),
                                          jnp.expand_dims(bs, axis=1)
                                          )
                        - V_bars)**2.).sum()

        def loss(Xps, Xcs, A_params, b_params, V_bars):
            """
            the loss function is the sum_square_diffs explicitly parameterized by A_params, b_params

            arguments
                Xps : jnp.array(N, Dx)
                    previous positions
                Xcs : jnp.array(N, Dx)
                    current positions
                A_params : jnp.array(Q)
                    parameters to pass to the A twisting function
                b_params : jnp.array(Q)
                    parameters to pass to the b twisting function
                V_bars : jnp.array(N)
                    previously-computed potentials

            returns
                out : float
                    sum of square differences
            """
            As, bs = self.vA_fn(Xps, A_params), self.vb_fn(Xps, b_params)
            return sum_square_diffs(Xcs, As, bs, V_bars)

        def scipy_loss(A_b_params, Xps, Xcs, V_bars):
            """
            rewrite the loss function s.t. it is amenable to scipy.optimize.minimize library

            arguments
                A_b_params : jnp.array(Q + R)
                    flattened A, b parameter inputs where Q is the dimension of A params and R is dimension of R params

                see above functions for redundant arguments

            returns
                out : float
                    loss
            """
            A_params, b_params = A_b_params[:self.A_params_len], A_b_params[self.A_params_len:]
            return loss(Xps, Xcs, A_params, b_params, V_bars)

        def t0_scipy_loss(A0_b0, Xcs, V_bars):
            """
            the scipy loss function for the t=0 twisting iteration is slightly different since we need not compute a tensor of As, bs.
            the twisting A0, b0 arrays are specified _exclusively_ as x-independent and parameter-independent arrays

            arguments
                A0_b0 : jnp.array(2*Dx)
                    flattened A0, b0 input parameters

                see above functions for redundant arguments

            returns
                out : float
                    loss
            """
            A0, b0 = A0_b0[:self.Dx], A0_b0[self.Dx:]
            return ((-vlog_psi_twist_single(Xcs, A0, b0) - V_bars)**2).sum()

        def get_callback_fn(Xps, Xcs, V_bars, t0=False):
            """
            utility to retrieve a callback function to pass to the scipy.optimize.minimize function for the purpose of logging outputs

            arguments
                see above functions for redundant arguments

            returns
                loss_logger : list()
                    empty list to log the losses
                callback : fn
                    callback function that is supposed to be passed to the scipy.optimize.minimize function

            """
            loss_logger = []
            if t0:
                partial_scipy_loss = partial(t0_scipy_loss, Xcs=Xcs, V_bars=V_bars)
            else:
                partial_scipy_loss = partial(scipy_loss, Xps=Xps, Xcs=Xcs, V_bars=V_bars)

            def callback(xk):
                val=partial_scipy_loss(xk)
                loss_logger.append(val)
            return loss_logger, callback

        #jax grad wrapper
        scipy_loss_grad = grad(scipy_loss, argnums=(0,))
        t0_scipy_loss_grad = grad(t0_scipy_loss, argnums=(0,))

        #to numpy wrapper
        # def np_scipy_loss(A_b_params, Xps, Xcs, V_bars): return float(scipy_loss(A_b_params, Xps, Xcs, V_bars))
        # def np_t0_scipy_loss(A0_b0, Xcs, V_bars): return float(t0_scipy_loss(A0_b0, Xcs, V_bars))
        # def np_scipy_loss_grad(A_b_params, Xps, Xcs, V_bars): return np.array(scipy_loss_grad(A_b_params, Xps, Xcs, V_bars), dtype=np.float64)
        # def np_t0_scipy_loss_grad(A0_b0, Xcs, V_bars): return np.array(t0_scipy_loss_grad(A0_b0, Xcs, V_bars), dtype=np.float64)

        def np_scipy_loss(*args): return float(scipy_loss(*args))
        def np_t0_scipy_loss(*args): return float(t0_scipy_loss(*args))
        def np_scipy_loss_grad(*args): return np.array(scipy_loss_grad(*args), dtype=np.float64)
        def np_t0_scipy_loss_grad(*args): return np.array(t0_scipy_loss_grad(*args), dtype=np.float64)


        def ADP(Xs,
                logWs,
                prop_params,
                minimizer_params = {
                                    'method': 'L-BFGS-B',
                                    'options': {'disp':True}
                                    },
                verbose=False
                ):
            """
            conduct ADP

            ISSUE : there is a discrepancy between the initial value of the loss function (scipy_loss) when computed inside/outside the scipy's minimize function.

            WARNING : this function CANNOT be jit'd
            """
            from scipy.optimize import minimize #import the optimizer

            #internal consistencies
            T, N = logWs.shape #get the number of steps and particles
            assert T == self.T
            assert N == self.N

            #initialize arrays as nan
            out_A_params = np.empty((T, self.A_params_len)); out_A_params[:] = np.nan
            out_b_params = np.empty((T, self.b_params_len)); out_b_params[:] = np.nan
            loss_trace=[] #create a loss trace if we are verbose

            logK_int_twist = jnp.zeros(N) #initialize the logK_twist_T+1

            _logger.debug(f"iterating backward...")
            for t in tqdm.tqdm(jnp.arange(1,T)[::-1]): #do ADP backward
                Vbar_t = -logWs[t] - logK_int_twist #define the Vbar_ts

                # TODO : do we have to flatten an array as first argument to minimizer
                runner_args = (Xs[t-1], Xs[t], Vbar_t) #other arguments passed to minimizer
                if verbose:
                    #generally, the loss loggers do not have consistent number of minimization steps at each iteration
                    #so it is just an empty list
                    loss_logger, callback_fn = get_callback_fn(*runner_args)
                else:
                    callback_fn=None

                out_containter = minimize(np_scipy_loss, #at time t, we use the generic np scipy loss function
                                          x0 = jnp.zeros((2, self.A_params_len)).flatten(), #initial parameters
                                          args = runner_args, #other args
                                          jac = np_scipy_loss_grad,
                                          callback=callback_fn,
                                          **minimizer_params)

                if verbose:
                    _logger.debug(f"minimizer message: {out_containter['message']}")
                    loss_trace.append(np.array(loss_logger))
                assert out_containter['success'], f"iteration {t} failed with message {out_containter['message']}" #TODO : make this fail safe

                #extract the output parameters
                reshaped_out_x = out_containter['x'].reshape(2,self.A_params_len) #check to make sure this is reshaped properly
                new_A_params, new_b_params = reshaped_out_x[0,:], reshaped_out_x[1,:]

                #record the parameters
                out_A_params[t,:] = new_A_params
                out_b_params[t,:] = new_b_params

                #compute the logK twisting integrals for the next optimization step
                _, _, logK_int_twist, _ = self.vtwist_fn(Xs[t-1],
                                                     self.forward_potential,
                                                     prop_params['dt'],
                                                     prop_params['forward_potential_params'][t],
                                                     self.A_fn,
                                                     self.b_fn,
                                                     jnp.vstack((self.A_params_cache[:self.twisting_iteration,t], jnp.asarray(new_A_params))),
                                                     jnp.vstack((self.b_params_cache[:self.twisting_iteration,t], jnp.asarray(new_b_params))),
                                                     True)

            #now to do the 0th time twist
            _logger.debug(f"conducting t=0 twist...")
            Vbar0 = -logWs[0] - logK_int_twist #compute the Vbars
            # TODO : do we have to flatten an array as first argument to minimizer

            runner_args = (None, Xs[0], Vbar0) #other arguments passed to minimizer
            if verbose:
                loss_logger, callback_fn = get_callback_fn(*runner_args, t0=True)
            else:
                callback_fn=None

            out_containter = minimize(np_t0_scipy_loss,
                                      x0 = jnp.zeros((2, self.Dx)).flatten(),
                                      args = runner_args[1:], #we cannot pass 'None'
                                      jac = np_t0_scipy_loss_grad,
                                      callback=callback_fn,
                                      **minimizer_params)
            if verbose:
                _logger.debug(f"minimizer message: {out_containter['message']}")
                loss_trace.append(np.array(loss_logger))
            assert out_containter['success'], f"iteration 0 failed with message {out_containter['message']}" #TODO : make this fail safe
            reshaped_out_x = out_containter['x'].reshape(2,self.Dx)
            new_A0, new_b0 = reshaped_out_x[0,:], reshaped_out_x[1,:]

            _logger.debug(f"finished optimizations; terminating...")

            # package the new parameters
            output = {
                      'out_A_params': out_A_params[1:],
                      'out_b_params': out_b_params[1:],
                      'out_A0': new_A0,
                      'out_b0': new_b0,
                      'loss_trace': loss_trace[::-1]
                      }

            return output

        out_fns = [sum_square_diffs, loss, scipy_loss, t0_scipy_loss, get_callback_fn, scipy_loss_grad, t0_scipy_loss_grad, np_scipy_loss, np_t0_scipy_loss, np_scipy_loss_grad, np_t0_scipy_loss_grad, ADP]
        output_fn_dict = {fn.__name__: fn for fn in out_fns}
        return output_fn_dict

    def update_cSMC(self,
                    new_A_params,
                    new_b_params,
                    new_A0,
                    new_b0):
        """
        update the instance with the new twisting parameters and initial parameters
        """
        _logger.debug(f"updating cSMC with new parameters")
        #make sure that the input parameters are compatible with those of the class
        assert new_A_params.shape == (self.T, self.A_params_len)
        assert new_b_params.shape == (self.T, self.b_params_len)
        assert new_A0.shape == (self.Dx,)
        assert new_b0.shape == (self.Dx,)

        #catalogue the parameters
        catalogue_index = self.twisting_iteration-1
        _logger.debug(f"cacheing parameters at index {catalogue_index}")
        self.A_params_cache[catalogue_index] = new_A_params
        self.b_params_cache[catalogue_index] = new_b_params
        self.A0 = new_A0
        self.b0 = new_b0

        #then update the twisting iteration
        self.twisting_iteration += 1

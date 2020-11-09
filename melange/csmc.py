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
    from melange import EL_mu_sigma
    mu, cov = EL_mu_sigma(x, potential, dt, parameters)
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
    A_b_merger = jnp.stack((As, bs), axis=1)
    init = (base_mu, base_cov, 0., get_log_normalizer)
    (twisted_mu, twisted_cov, log_twist_constant, out_bool), _ = scan(twist_scanner_fn, init, A_b_merger)
    logZ = cond(get_log_normalizer, lambda x: Normal_logZ(*x), lambda x: 0., (twisted_mu, twisted_cov))
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


"""twisting potentials"""
log_psi_twist_A = lambda x, As: -(x*As.sum(axis=0)).dot(x)
log_psi_twist_b = lambda x, bs: -x.dot(bs.sum(axis=0))
log_psi_twist = lambda x, As, bs: log_psi_twist_A(x, As).sum() + log_psi_twist_b(x, bs).sum()



class StaticULAControlledSMC(StaticULA):
    """
    controlled SMC handler for Static model wth an Unadjusted Langevin Algorithm

    prop_params is a dict containing: ['potential_params', 'forward_potential_params', 'backward_potential_params', 'dt', 'A_params', 'b_params'];
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
                 max_twists = 10,
                 **kwargs):
        _logger.debug(f"instantiating super")
        super().__init__(*args, **kwargs) #init the og args

        _logger.debug(f"getting x dimension")
        self.Dx = Dx

        #define the twisting functions
        _logger.debug(f"define A, b functions")
        self.A_fn = A_fn
        self.b_fn = b_fn

        #define twisting iterations
        _logger.debug(f"define twisting parameters and parameter caches")
        self.twisting_iteration = 1
        self.max_twists = max_twists
        self.A_params_cache = jnp.zeros((max_twists, A_params_len))
        self.b_params_cache = jnp.zeros((max_twists, A_params_len))
        self.A0, self.b0 = jnp.zeros(self.Dx), jnp.zeros(self.Dx)
        self.A_params_len = A_params_len
        self.b_params_len = b_params_len

        #define the psi twisting function
        _logger.debug(f"define psi twisting functions")
        self.vlog_psi_twist = vmap(log_psi_twist, in_axes=(0,0,0))
        self.twist_fn = gaussians_twist
        self.vtwist_fn = vmap(self.twist_fn, in_axes=(0, None, None, None, None, None, None, None, None)) #args (Xp, potential, dt, potential_params, A_fn, b_fn, A_params, b_params, get_log_normalizer)


    def sim_prop_fn(self):
        _logger.debug(f"generating simulation propagation function")

        _prop = lambda key, mu, cov: random.multivariate_normal(key, mu, cov)
        self.vpropagator = vmap(_prop, in_axes = (0,0,0))

        def prop(t, Xp, y, prop_params, model_params, rs):
            N, Dx = Xp.shape #extract the number of particles and the dimension of x
            folder_rs = random.split(rs, num=N+1) #create a list of random keys
            new_rs, runner_rs = folder_rs[0], folder_rs[1:] #separate the random keys

            twisted_mus, twisted_covs, _, _, _ = self.vtwist_fn(Xp,
                                                          self.forward_potential,
                                                          prop_params['dt'],
                                                          prop_params['forward_potential_params'],
                                                          self.A_fn,
                                                          self.b_fn,
                                                          self.A_params_cache[:self.twisting_iteration],
                                                          self.b_params_cache[:self.twisting_iteration],
                                                          False)


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
            """
            # compute untwisted logWs
            logWs = static_ula_logW_fn(t, Xp, Xc, y, prop_params, model_params)

            #compute twisting logZ forward kernel and psi_twist
            _, _, logK_Zs, As, bs = self.vtwist_fn(Xp,
                                                 self.forward_potential,
                                                 prop_params['dt'],
                                                 prop_params['forward_potential_params'],
                                                 self.A_fn,
                                                 self.b_fn,
                                                 self.A_params_cache[:self.twisting_iteration],
                                                 self.b_params_cache[:self.twisting_iteration],
                                                 True)

            #compute twisting functions
            log_psi_ts = self.vlog_psi_twist(Xc, As, bs)
            return logWs + logK_Zs - log_psi_ts
        return log_weights

    def initialize_Xs_fn(self):
        """
        initialize Xs
        """
        _logger.debug(f"generating SMC variable initializer function")
        from melange.gaussians import get_twisted_gmm, sample_gmm
        from melange.miscellaneous import exp_normalize
        self.vsample_gmm = vmap(sample_gmm, in_axes = (0, None, None, None)) # args: (key, weights, mus, covs)


        def init_xs(prop_params, rs, init_params):
            #resolve random keys
            folder_rs = random.split(rs, self.N+1)
            rs, run_rs = folder_rs[0], folder_rs[1:]

            # conduct twisted initialization
            _, log_alpha_tildes, _, (twisted_mus, sigma_tildes) = get_twisted_gmm(init_params['mixture_weights'],
                                                                                  init_params['mus'],
                                                                                  init_params['covs'],
                                                                                  self.A0,
                                                                                  self.b0) #compute twisting parameters

            _, Xs = self.vsample_gmm(run_rs, exp_normalize(log_alpha_tildes), twisted_mus, sigma_tildes)
            return Xs
        return init_xs

    def initialize_logW_fn(self, **kwargs):
        _logger.debug(f"generating SMC log weight initializer function")
        from melange.gaussians import get_twisted_gmm

        def init_logWs(X, init_params, prop_params):
            log_normalizer, _, _, _ = get_twisted_gmm(init_params['mixture_weights'],
                                                      init_params['mus'],
                                                      init_params['covs'],
                                                      self.A0,
                                                      self.b0) #compute twisting parameters

            # compute twisted weights
            log_psi0s = log_normalizer - self.vlog_psi0_twist(X,
                                                              self.A0,
                                                              self.b0)

            return log_psi0s
        return init_logWs

    def get_ADP_fn(self):
        """
        get the functions needed to do approximate dynamic programming
        # TODO : add constraints for the twisted matrices (have to be positive definite)
        """
        _logger.debug(f"generating approximate dynamic programming functions")
        from jax.ops import index, index_add, index_update
        from jax.lax import scan
        from jax import value_and_grad
        import tqdm

        (A_fn, b_fn) = (self.A_fn, self.b_fn) if self.twisting_iteration == 0 else (self.dummy_A_fn, self.dummy_b_fn)
        (A_params, b_params) = (jnp.zeros((1,1,1)), jnp.zeros((1,1,1))) if self.twisting_iteration == 0 else (self.A_params_cache[:self.twisting_iteration], self.b_params_cache[:self.twisting_iteration])

        if self.twisting_iteration == 0:
            def naught_A_fn(x, params):
                return self.base_A_fn(x, params[-1])
            def naught_b_fn(x, params):
                return self.base_b_fn(x, params[-1])
            twist_A_fn, twist_b_fn = naught_A_fn, naught_b_fn
        else:
            twist_A_fn, twist_b_fn = self.A_fn, self.b_fn

        A_params_cache = A_params
        b_params_cache = b_params

        opt_A0, opt_b0 = jnp.zeros((self.Dx, self.Dx)),  jnp.zeros(self.Dx)

        def loss_t(param_dict, Xps, Xcs, Vbars):
            """
            define a loss function for the t_th parameters;
            the loss here is w.r.t. the twisted _base_ functions
            """
            _A_params = param_dict['A_params'] # extract A
            _b_params = param_dict['b_params'] # extract b
            return ((-self.vlog_psi_t_twist(Xps, Xcs, self.base_A_fn, self.base_b_fn, _A_params, _b_params) - Vbars)**2).sum()

        def loss_0(param_dict, X0s, Vbars):
            """define a loss function for the 0th parameters"""
            _A0 = param_dict['A0'] # extract A
            _b0 = param_dict['b0'] # extract b
            return ((-self.vlog_psi0_twist(X0s, _A0, _b0) - Vbars)**2).sum() # (X0, A0, b0)

        #define the gradient of the loss functions: always arg 0
        grad_loss_t = value_and_grad(loss_t)
        grad_loss_0 = value_and_grad(loss_0)

        def optimize(loss_and_grad_fn, loss_and_grad_fn_args, optimization_steps, lr_dict):
            """define the optimizer function for the t_th paramters"""
            def scanner(carry_param_dict, t):
                loss, loss_grad = loss_and_grad_fn(carry_param_dict, *loss_and_grad_fn_args[1:])
                for key, val in loss_grad.items():
                    carry_param_dict[key] = carry_param_dict[key] - val*lr_dict[key]
                return carry_param_dict, loss

            out_param_dict, losses = scan(scanner, loss_and_grad_fn_args[0], jnp.arange(optimization_steps))
            return out_param_dict, losses

        def ADP(Xs,
                rs,
                logWs,
                prop_params,
                init_params,
                opt_steps,
                learning_rates_t={'A_params': 1e-5, 'b_params': 1e-5},
                learning_rates_0={'A0': 1e-5, 'b0': 1e-5},
                randomization_scale=1e-2):
            """
            conduct ADP
            """
            T, N = logWs.shape #get the number of steps and particles
            rs, A_rs, b_rs = random.split(rs, 3)

            A_params_to_opt = jnp.zeros((T, self.A_params_len))
            b_params_to_opt = jnp.zeros((T, self.b_params_len))

            #define container
            losses = jnp.zeros((T, opt_steps))

            #define the initial loss:
            init_losses = jnp.zeros(T)

            logK_int_twist = jnp.zeros(N) #initialize the logK_twist_T+1

            print(f"iterating backward...")
            for t in tqdm.tqdm(jnp.arange(1,T)[::-1]): #do ADP backward
                Vbar_t = -logWs[t] - logK_int_twist #define the Vbar_ts
                init_loss = (Vbar_t**2).sum()
                print(f"t: {t}; initial loss: {init_loss}")
                init_losses = index_update(init_losses, index[t], init_loss)
                opt_param_dict, losses_t = optimize(loss_and_grad_fn = grad_loss_t,
                                                  loss_and_grad_fn_args = (
                                                                              {'A_params': A_params_to_opt[t],
                                                                               'b_params': b_params_to_opt[t]},
                                                                              Xs[t-1],
                                                                              Xs[t],
                                                                              Vbar_t
                                                                          ),
                                                  optimization_steps=opt_steps,
                                                  lr_dict = learning_rates_t
                                                 )
                print(f"optimized parameters: {opt_param_dict}")

                A_params_to_opt = index_update(A_params_to_opt, index[t, :], opt_param_dict['A_params']) #update A params
                b_params_to_opt = index_update(b_params_to_opt, index[t, :], opt_param_dict['b_params']) # update b params
                losses = index_update(losses, index[t,:], losses_t) #update losses

                #aggregate the parameters
                _twist_A_params = jnp.vstack((A_params_cache[:,t,:], opt_param_dict['A_params'][jnp.newaxis, ...]))
                _twist_b_params = jnp.vstack((b_params_cache[:,t,:], opt_param_dict['b_params'][jnp.newaxis, ...]))
                assert len(list(_twist_A_params.shape)) == 2
                print(f"twisting A params: {_twist_A_params}")
                print(f"twist b params: {_twist_b_params}")


                #compute the logK twisting integrals for the next optimization step
                _, bs, fs, thetas = self.vdrive_params(Xs[t-1], # get the previous positions
                                                   self.forward_potential, # get the forward potential
                                                   prop_params['dt'], #get the dt parameter
                                                   twist_A_fn, #A function
                                                   twist_b_fn, #b function
                                                   prop_params['forward_potential_params'][t], #forward potential parameters at time t
                                                   _twist_A_params,
                                                   _twist_b_params
                                                 )
                logK_int_twist = self.vlogK_ints(bs, fs, thetas, prop_params['dt']) #calculate the twistin

            #now to do the 0th time twist
            Vbar0 = -logWs[0] - logK_int_twist #compute the Vbars
            init_losses = index_update(init_losses, index[0], (Vbar0**2).sum())
            opt_param_dict, losses_0 = optimize(loss_and_grad_fn = grad_loss_0,
                                                  loss_and_grad_fn_args= (
                                                                          {'A0': opt_A0,
                                                                           'b0': opt_b0},
                                                                          Xs[0],
                                                                          Vbar0
                                                                         ),
                                                  optimization_steps = opt_steps,
                                                  lr_dict = learning_rates_0
                                                 )
            # pull out the optimized parameter
            out_A0 = opt_param_dict['A0']
            out_b0 = opt_param_dict['b0']
            losses = index_update(losses, index[0,:], losses_0) #update losses

            # package the new parameters
            new_params = {'A0': out_A0, 'b0': out_b0, 'A_params': A_params_to_opt, 'b_params': b_params_to_opt}

            return new_params, losses, init_losses

        return (loss_0, loss_t), (grad_loss_0, grad_loss_t), optimize, ADP

    def get_fn_scanner(self):
        def scan_fn(carry, param_stack):
            xs, base_fn = carry
            return (xs, base_fn), base_fn(xs, param_stack)
        return scan_fn

    def update_twist(self, opt_params):
        """update the twist"""
        from jax.ops import index, index_add, index_update
        from jax.lax import scan, map
        from functools import partial


        #pull the optimized parameters
        A_T, A_dim = opt_params['A_params'].shape
        b_T, b_dim = opt_params['b_params'].shape

        assert A_T == b_T
        assert A_dim == self.A_params_len
        assert b_dim == self.b_params_len
        T = A_T

        if self.twisting_iteration == 0:
            self.A_params_cache = jnp.zeros((self.max_twists, T, A_dim))
            self.b_params_cache = jnp.zeros((self.max_twists, T, b_dim))

        # update parameter cache
        self.A_params_cache = index_update(self.A_params_cache, index[self.twisting_iteration, :, :], opt_params['A_params']) #update A params
        self.b_params_cache = index_update(self.b_params_cache, index[self.twisting_iteration, :, :], opt_params['b_params']) #update A params

        # update functions
        run_A_fn = self.base_A_fn
        run_b_fn = self.base_b_fn

        def new_A_fn(x, parameters):
            partial_A = partial(run_A_fn, x)
            outs = map(partial_A, parameters)
            return outs.sum(0)
        def new_b_fn(x, parameters):
            partial_b = partial(run_b_fn, x)
            outs = map(partial_b, parameters)
            return outs.sum(0)

        self.A_fn = new_A_fn
        self.b_fn = new_b_fn

        self.A0 = opt_params['A0']
        self.b0 = opt_params['b0']

        self.twisting_iteration += 1

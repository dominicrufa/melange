"""
smc objects and utilities for controlled SMC
"""
from melange.smc_objects import StaticULA
import jax.numpy as jnp
from jax import random
from jax import grad, vmap, jit


class StaticULAControlledSMC(StaticULA):
    """
    controlled SMC handler for Static model wth an Unadjusted Langevin Algorithm

    prop_params is a dict containing: ['potential_params', 'forward_potential_params', 'backward_potential_params', 'dt', 'A_params', 'b_params'];
    init_params is a dict containing: ['mixture_weights', 'mus', 'covs', 'A0', 'b0']
    """
    def __init__(self, *args, A_fn, b_fn, A_params_len, b_params_len, Dx, max_twists = 10, **kwargs):
        from melange.propagators import driven_Langevin_parameters
        from melange.miscellaneous import log_twisted_psi_t, log_twisted_psi0
        from melange.gaussians import logK_ints
        super().__init__(*args, **kwargs)

        self.Dx = Dx

        #special vmapped utility function to compute driven langevin parameters
        self.vdrive_params = vmap(driven_Langevin_parameters, in_axes = (0,None,None, None, None, None, None, None)) # args: (x, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter)


        #define vmapped twisting functions and logK_integral functions
        self.vlog_psi_t_twist = vmap(log_twisted_psi_t, in_axes = (0,0,None,None,None,None)) # args: (Xp, Xc, A_fn, b_fn, A_params, b_params)
        self.vlogK_ints = vmap(logK_ints, in_axes = (0, 0, 0, None)) # args: (b, f, theta, dt)

        #define the twisting functions
        self.A_fn = A_fn
        self.b_fn = b_fn

        #define the base twisting functions
        self.base_A_fn = A_fn
        self.base_b_fn = b_fn

        #dummy_fns
        self.dummy_A_fn = lambda x, p: jnp.zeros((x.shape[0], x.shape[0]))
        self.dummy_b_fn = lambda x, p : jnp.zeros(x.shape[0])

        #define twisting iterations
        self.twisting_iteration = 0
        self.max_twists = max_twists
        self.A_params_cache = None
        self.b_params_cache = None
        self.A0 = jnp.zeros((self.Dx, self.Dx))
        self.b0 = jnp.zeros(self.Dx)

        #define the twisting function
        self.vlog_psi_t_twist = vmap(log_twisted_psi_t, in_axes = (0,0,None,None,None,None)) # args: (Xp, Xc, A_fn, b_fn, A_params, b_params)
        self.vlogK_ints = vmap(logK_ints, in_axes = (0, 0, 0, None)) # args: (b, f, theta, dt)
        self.vlog_psi0_twist = vmap(log_twisted_psi0, in_axes = (0, None, None)) # args: (X0, A0, b0)

        #dummy
        def dummy_vdrive_params(Xp, *args, **kwargs):
            N, Dx = Xp.shape
            return jnp.zeros((N, Dx, Dx)), jnp.zeros((N, Dx)), jnp.zeros((N, Dx)), jnp.zeros((N, Dx, Dx))
        self.dummy_vdrive_params = dummy_vdrive_params

        #get the shape of the A, b params
        self.A_params_len = A_params_len
        self.b_params_len = b_params_len


    def sim_prop_fn(self):
        from melange.propagators import driven_mu_cov
        from jax.lax import cond
        _prop = lambda key, mu, cov: random.multivariate_normal(key, mu, cov)
        self.vpropagator = vmap(_prop, in_axes = (0,0,0))
        self.vdriven_mu_cov = vmap(driven_mu_cov, in_axes = (0, 0, 0, None)) #args : (b, f, theta, dt)

        (A_fn, b_fn) = (self.A_fn, self.b_fn) if self.twisting_iteration > 0 else (self.dummy_A_fn, self.dummy_b_fn)
        (A_params, b_params) = (jnp.zeros((1,1,1)), jnp.zeros((1,1,1))) if self.twisting_iteration == 0 else (self.A_params_cache[:self.twisting_iteration], self.b_params_cache[:self.twisting_iteration])

        def prop(t, Xp, y, prop_params, model_params, rs):
            N, Dx = Xp.shape
            folder_rs = random.split(rs, num=N+1)
            new_rs, runner_rs = folder_rs[0], folder_rs[1:]

            As, bs, fs, thetas = self.vdrive_params(Xp,
                                                    self.forward_potential,
                                                    prop_params['dt'],
                                                    A_fn,
                                                    b_fn,
                                                    prop_params['potential_params'][t],
                                                    A_params[:, t, :], # A_params
                                                    b_params[:, t, :] #b_params
                                                    )

            mus, covs = self.vdriven_mu_cov(bs, fs, thetas, prop_params['dt'])
            Xs = self.vpropagator(runner_rs, mus, covs)
            return Xs
        return prop

    def log_weights_fn(self):
        #define vmapped twisting functions and logK_integral functions
        static_ula_logW_fn = super().log_weights_fn() #get the super static ula log_weights_fn

        (A_fn, b_fn) = (self.A_fn, self.b_fn) if self.twisting_iteration > 0 else (self.dummy_A_fn, self.dummy_b_fn)
        (A_params, b_params) = (jnp.zeros((1,1,1)), jnp.zeros((1,1,1))) if self.twisting_iteration == 0 else (self.A_params_cache[:self.twisting_iteration], self.b_params_cache[:self.twisting_iteration])

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


            # twisting helpers
            _, bs, fs, thetas = self.vdrive_params(Xp,
                                        self.forward_potential,
                                        prop_params['dt'],
                                        A_fn,
                                        b_fn,
                                        prop_params['forward_potential_params'][t],
                                        A_params[:, t, :], # A_params
                                        b_params[:, t, :] #b_params
                                        )

            #compute logK_ints
            logK_ints = self.vlogK_ints(bs, fs, thetas, prop_params['dt'])

            #compute log psi_t_twist function
            log_psi_ts = self.vlog_psi_t_twist(Xp,
                                               Xc,
                                               A_fn,
                                               b_fn,
                                               A_params[:,t,:],
                                               b_params[:,t,:]
                                               )
            return logWs + logK_ints - log_psi_ts
        return log_weights



    def initialize_Xs_fn(self):
        """
        initialize Xs
        """
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
            mixture_ids, Xs = self.vsample_gmm(run_rs, exp_normalize(log_alpha_tildes), twisted_mus, sigma_tildes)
            return Xs
        return init_xs

    def initialize_logW_fn(self, **kwargs):
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
            A_params_cache = jnp.zeros((1,1,self.A_params_len))
            b_params_cache = jnp.zeros((1,1,self.b_params_len))
        else:
            twist_A_fn, twist_b_fn = self.A_fn, self.b_fn
            A_params_cache = self.A_params_cache
            b_params_cache = self.b_params_cache

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

            new_A_params = jnp.zeros((T, self.A_params_len))
            new_b_params = jnp.zeros((T, self.b_params_len))

            #define container
            losses = jnp.zeros((T, opt_steps))

            #define the initial loss:
            init_losses = jnp.zeros(T)

            logK_int_twist = jnp.zeros(N) #initialize the logK_twist_T+1

            print(f"iterating backward...")
            for t in tqdm.tqdm(jnp.arange(1,T)[::-1]): #do ADP backward
                Vbar_t = -logWs[t] - logK_int_twist #define the Vbar_ts
                init_losses = index_update(init_losses, index[t], (Vbar_t**2).sum())
                opt_param_dict, losses_t = optimize(loss_and_grad_fn = grad_loss_t,
                                                  loss_and_grad_fn_args = (
                                                                              {'A_params': new_A_params[t],
                                                                               'b_params': new_b_params[t]},
                                                                              Xs[t-1],
                                                                              Xs[t],
                                                                              Vbar_t
                                                                          ),
                                                  optimization_steps=opt_steps,
                                                  lr_dict = learning_rates_t
                                                 )

                new_A_params = index_update(new_A_params, index[t, :], opt_param_dict['A_params']) #update A params
                new_b_params = index_update(new_b_params, index[t, :], opt_param_dict['b_params']) # update b params
                losses = index_update(losses, index[t,:], losses_t) #update losses

                #aggregate the parameters
                _twist_A_params = jnp.vstack((A_params_cache[:,t,:], opt_param_dict['A_params'][jnp.newaxis, ...]))
                _twist_b_params = jnp.vstack((b_params_cache[:,t,:], opt_param_dict['b_params'][jnp.newaxis, ...]))
                assert len(list(_twist_A_params.shape)) == 2


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
            new_params = {'A0': out_A0, 'b0': out_b0, 'A_params': new_A_params, 'b_params': new_b_params}

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

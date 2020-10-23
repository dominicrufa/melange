"""
smc objects
"""
from jax import numpy as jnp
from jax import random, vmap, grad
import numpy as np
from jax.lax import stop_gradient

class BaseSMCObject(object):
    """
    base algorithm class for static models
    """
    def __init__(self, N):
        self.N = N

class StaticULA(BaseSMCObject):
    """
    unadjusted langevin algorithm class for static models
    """
    def __init__(self, N, potential, forward_potential, backward_potential):
        from melange.propagators import ULA_move, log_Euler_Maruyma_kernel
        super().__init__(N)

        #define forward/backward_potentials
        self.forward_potential = forward_potential
        self.backward_potential = backward_potential
        self.vforward_potential = vmap(forward_potential, in_axes=(0,None))

        #propagator
        #ULA_move(x, potential, dt, key, potential_parameter)
        self.vpropagator = vmap(ULA_move, in_axes=(0, None, None, 0, None))

        #kernel
        self.vkernel = vmap(log_Euler_Maruyma_kernel, in_axes = (0, 0, None, None, None))
        self.vpotential = vmap(potential, in_axes = (0,None))

    def sim_prop_fn(self):
        """
        return a function that takes a tuple and propagates latent variables in a step
        """
        def prop(t, Xp, y, prop_params, model_params, rs):
            """
            arguments
                prop_params : tuple
                    potential_params : jnp.array(T,Q)
                    forward_potential_params : jnp.array(T,R)
                    backward_potential_params : jnp.array(T-1,S)
                    forward_dts : jnp.array(T)
                    backward_dts : jnp.array(T-1)
            """
            N = Xp.shape[0]
            potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params
            folder_rs = random.split(rs, num=N+1)
            new_rs, runner_rs = folder_rs[0], folder_rs[1:]
            X = self.vpropagator(Xp, self.forward_potential, forward_dts[t], runner_rs, forward_potential_params[t])
            return X
        return prop

    def log_weights_fn(self):
        """
        return a function that takes a tuple and computes log weights of a set of particles
        """
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

            potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params

            potential_diff = self.vpotential(Xp, potential_params[t-1]) - self.vpotential(Xc, potential_params[t])
            logK = self.vkernel(Xp, Xc, self.forward_potential, forward_potential_params[t], forward_dts[t])
            logL = self.vkernel(Xc, Xp, self.backward_potential, backward_potential_params[t-1], backward_dts[t])
            return potential_diff + logL - logK
        return log_weights

    def initialize_Xs_fn(self):
        """
        initialize Xs and log weights

        arguments
            init_params :
                mu
                cov
                gradable_initializer : bool
                    whether to allow grad of the initializer

        returns
            X, logW
        """
        def init_xs(prop_params, rs, init_params):
            """
            arguments
                init_params :
                    mu
                    cov
                    potential_params
                    forward_potential_params
            """
            potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params
            mu, cov = init_params
            X = random.multivariate_normal(key = rs, mean = mu, cov = cov, shape=[self.N])
            return X
        return init_xs

    def initialize_logW_fn(self, gradable_initializer):
        if not gradable_initializer:
            def init_logWs(X, prop_params):
                potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params

                logW = stop_gradient(self.vforward_potential(X, forward_potential_params[0]) - self.vpotential(X, potential_params[0]))
                return logW
        else:
            def init_logWs(X, prop_params):
                potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params
                logW = self.vforward_potential(X, forward_potential_params[0]) - self.vpotential(X, potential_params[0])
                return logW
        return init_logWs

    def get_fns(self, gradable_initializer=False):
        """
        return tuple:
            sim_prop_fn
            log_weights_fn
            (
                initialize_Xs_fn,
                initialize_logW_fn
            )
        """
        return (self.sim_prop_fn(), self.log_weights_fn(), (self.initialize_Xs_fn(), self.initialize_logW_fn(gradable_initializer)))

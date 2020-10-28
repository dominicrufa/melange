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
        self.potential = potential
        self.vpotential = vmap(self.potential, in_axes = (0,None))
        self.forward_potential = forward_potential
        self.backward_potential = backward_potential
        self.vforward_potential = vmap(self.forward_potential, in_axes=(0,None))

        #kernel
        self.kernel = log_Euler_Maruyma_kernel
        self.vkernel = vmap(self.kernel, in_axes = (0, 0, None, None, None))

        #propagator
        self.vpropagator = vmap(ULA_move, in_axes=(0, None, None, 0, None))

    def sim_prop_fn(self):
        def prop(t, Xp, y, prop_params, model_params, rs):
            N = Xp.shape[0]
            folder_rs = random.split(rs, num=N+1)
            new_rs, runner_rs = folder_rs[0], folder_rs[1:]
            X = self.vpropagator(Xp,
                                 self.forward_potential,
                                 prop_params['dt'],
                                 runner_rs,
                                 prop_params['forward_potential_params'][t])
            return X
        return prop

    def log_weights_fn(self):
        def log_weights(t, Xp, Xc, y, prop_params, model_params):
            potential_diff = self.vpotential(Xp, prop_params['potential_params'][t-1]) - self.vpotential(Xc, prop_params['potential_params'][t])
            logK = self.vkernel(Xp, Xc, self.forward_potential, prop_params['forward_potential_params'][t], prop_params['dt'])
            logL = self.vkernel(Xc, Xp, self.backward_potential, prop_params['backward_potential_params'][t-1], prop_params['dt'])
            return potential_diff + logL - logK
        return log_weights

    def initialize_Xs_fn(self):
        def init_xs(prop_params, rs, init_params):
            mu, cov = init_params
            X = random.multivariate_normal(key = rs, mean = init_params['mu'], cov = init_params['cov'], shape=[self.N])
            return X
        return init_xs

    def initialize_logW_fn(self, gradable_initializer):
        if not gradable_initializer:
            def init_logWs(X, init_params, prop_params):
                logW = stop_gradient(self.vforward_potential(X, prop_params['forward_potential_params'][0]) - self.vpotential(X, prop_params['potential_params'][0]))
                return logW
        else:
            def init_logWs(X, init_params, prop_params):
                logW = self.vforward_potential(X, prop_params['forward_potential_params'][0]) - self.vpotential(X, prop_params['potential_params'][0])
                return logW
        return init_logWs

    def get_fns(self, gradable_initializer=False):
        return (self.sim_prop_fn(), self.log_weights_fn(), (self.initialize_Xs_fn(), self.initialize_logW_fn(gradable_initializer)))

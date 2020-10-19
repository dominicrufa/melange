"""
smc objects
"""
from jax import numpy as jnp
from jax import random, vmap, grad
import numpy as np
from jax.lax import stop_gradient

class BaseSMCReporter(object):
    """
    generalized reporter object for SMC
    """
    def __init__(self, smc_obj, save_Xs=True):
        self.X = np.zeros((smc_obj.T, smc_obj.N, smc_obj.Dx))
        self.ESS = np.zeros(smc_obj.T)
        self.logW = np.zeros(smc_obj.T,smc_obj.N)
    def report(self, t, X, logW):
        self.X[t] = X
        self.logW[t] = logW

class BaseSMCObject(object):
    """
    base class for SMC
    """
    def __init__(self, T, N, Dx, **kwargs):
        """
        arguments
            T : int
                number of iterations
            N : int
                number of particles
            Dx : int
                dimension of x
        """
        self.T = T
        self.N = N
        self.Dx = Dx

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs):
        """
        propagate the Xs

        arguments
            t : int
                time iteration
            Xp : jnp.array(N, Dx)
                previous positions
            y : jnp.array(Dy)
                observation
            prop_params : tuple
                tuple of propagation parameters
            model_params : tuple
                tuple of model parameters
            rs : jax.random.PRNGKey
                random key
        returns
            X : jnp.array(N,Dx)
                new model parameters
        """
        assert len(rs) == self.N, f"the length of propagation randoms should be {self.N}, but it is {len(rs)}"

    def log_weights(self, t, Xp, Xc, y, prop_params, model_params):
        """
        arguments
            t : int
                time iteration
            Xp : jnp.array(N, Dx)
                previous positions
            Xc : jnp.array(N, Dx)
                current positions
            y : jnp.array(Dy)
                observation
            prop_params : tuple
                tuple of propagation parameters
            model_params : tuple
                tuple of model parameters
            rs : jax.random.PRNGKeys
                random keys of lenth N
        returns
            log_weights : jnp.array(N)
                log unnormalized weights
        """
        pass

    def initialize(self, init_params):
        """
        arguments
            init_params : tuple
                arguments for initialization
        """
        pass


class StaticULA(BaseSMCObject):
    """
    unadjusted langevin algorithm class for static models
    """
    def __init__(self, T, N, Dx, potential, forward_potential, backward_potential):
        super().__init__(T,N, Dx)
        from melange.propagators import ULA_move, log_Euler_Maruyma_kernel

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

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs):
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
        folder_rs = random.split(rs, num=self.N+1)
        new_rs, runner_rs = folder_rs[0], folder_rs[1:]
        super().sim_prop(t,Xp, y,prop_params, model_params, runner_rs)
        assert Xp.shape == (self.N, self.Dx), f"uh oh: {Xp.shape}"
        X = self.vpropagator(Xp, self.forward_potential, forward_dts[t], runner_rs, forward_potential_params[t])
        return X

    def log_weights(self, t, Xp, Xc, y, prop_params, model_params):
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

    def initialize(self, prop_params, rs, init_params):
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
        potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params
        mu, cov, gradable_initializer = init_params

        X = self.init_multivar_normal(mu, cov, rs)

        if not gradable_initializer:
            logW = stop_gradient(self.vforward_potential(X, forward_potential_params[0]) - self.vpotential(X, potential_params[0]))
        else:
            logW = self.vforward_potential(X, forward_potential_params[0]) - self.vpotential(X, potential_params[0])

        return X, logW

    def init_multivar_normal(self, mu, cov, rs):
        """
        initialize with multivariate normal distribution
        """
        outs = random.multivariate_normal(key = rs, mean = mu, cov = cov, shape=[self.N])
        return outs

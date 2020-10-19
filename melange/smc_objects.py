"""
smc objects
"""
from jax import numpy as jnp
from jax import random

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
        assert len(rs) == self.N

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


class StaticULA(BaseSMCObject):
    """
    unadjusted langevin algorithm class for static models
    """
    def __init__(self, T, N, Dx, potential, forward_potential, backward_potential):
        super().__init__(T,N, Dx)
        from melange.propagators import ULA_move, log_Euler_Maruyma_kernel

        #define forward/backward_potentials
        self.vforward_potential = vmap(forward_potential, in_axes=(0,None))
        self.vbackward_potential = vmap(forward_potential, in_axes=(0,None))

        #propagator
        #ULA_move(x, potential, dt, key, potential_parameter)
        self.vpropagator = vmap(ULA_move, in_axes=(0, None, None, 0, None))

        #kernel
        self.vkernel = vmap(log_Euler_Maruyma_kernel, in_axes = (0, 0, None, None, None))
        self.vpotential = vmap(potential, in_axes = (0,None))

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs):
        """
        arguments
            prop_params: tuple
                forward_potential_parameters : jnp.array(R)
                forward_dt : float
        """
        super().sim_prop(t,Xp, y,prop_params, model_params, rs)
        forward_potential_parameters, forward_dt = prop_params
        X = self.vpropagator(Xp, self.vforward_potential, forward_dt, rs, forward_potential_parameters)
        return X

    def log_weights(self, t, Xp, Xc, y, prop_params, model_params):
        """
        arguments
            prop_params : tuple
                previous_potential_parameters : jnp.array(Q)
                current_potential_parameters : jnp.array(Q)
                forward_potential_parameters : jnp.array(R)
                backward_potential_parameters : jnp.array(S)
                forward_dts : float
                backward_dts : float
        """
        previous_potential_parames, current_potential_params, forward_potential_params, backward_potential_params, forward_dts, backward_dts = prop_params

        potential_diff = self.vpotential(Xp, potential_params) - self.vpotential(Xc, potential_params)
        logK = self.vkernel(Xp, Xc, self.vforward_potential, forward_potential_params, forward_dts)
        logL = self.vkernel(Xc, Xp, self.vbackward_potential, backward_potential_params, backward_dts)
        return potential_diff + logL - logK
        

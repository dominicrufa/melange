"""
simple reporter utilities for SMC
"""
import numpy as np
from jax.lax import stop_gradient
import jax
from jax.ops import index, index_add, index_update
from jax import numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)

class BaseSMCReporter(object):
    """
    generalized reporter object for SMC
    """
    def __init__(self,
                 T,
                 N,
                 Dx,
                 save_Xs=True):

        self.X = jnp.zeros((T,N,Dx))
        self.nESS = jnp.zeros(T)
        self.logW = jnp.zeros((T,N))
        self.save_Xs=save_Xs

    def report(self, t, reportables):
        """
        arguments
            reportables : tuple
        """
        # X, logZ, nESS = reportables
        # self.X[t] = X
        # self.logZ[t] = logZ
        # self.nESS[t] = nESS
        pass

class vSMCReporter(BaseSMCReporter):
    """
    reporter object for vSMC
    """
    def __init__(self, T, N, Dx, save_Xs=True):
        super().__init__(T,N,Dx,save_Xs)

    def report(self, t,reportables):
        """
        arguments
            reportables: tuple
                X, logZ, nESS
        """
        X, logW, nESS = reportables
        if self.save_Xs:
            self.X = index_update(self.X, index[t,:], stop_gradient(X))

        self.logW = index_update(self.logW, index[t], stop_gradient(logW))
        self.nESS = index_update(self.nESS, index[t], stop_gradient(nESS))

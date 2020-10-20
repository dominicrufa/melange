"""
simple reporter utilities for SMC
"""
import numpy as np

class BaseSMCReporter(object):
    """
    generalized reporter object for SMC
    """
    def __init__(self,
                 T,
                 N,
                 Dx,
                 save_Xs=True):
        self.T = T
        self.N = N
        self.Dx = Dx

        self.X = np.zeros((self.T, self.N, self.Dx))
        self.nESS = np.zeros(self.T)
        self.logZ = np.zeros((self.T,self.N))
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
    def __init__(self, T,N, Dx, save_Xs=True):
        super().__init__(T, N, Dx, save_Xs)

    def report(self, t, reportables):
        """
        arguments
            reportables: tuple
                X, logZ, nESS
        """
        X, logZ, nESS = reportables
        self.X[t] = X
        self.logZ[t] = logZ
        self.nESS[t] = nESS

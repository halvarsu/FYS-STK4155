import numpy as np
from scipy import linalg

class Regression(object):

    """Simple tool for linear, ridge or lasso regression."""

    def __init__(self, X, y, lmbd = 0):
        """TODO: to be defined1. """
        self._X = X
        self._y = y
        self._symX = self._X.T @ self._X
        self._symXInv = linalg.inv(self._symX)
        self._lmbd = lmbd

    @property
    def beta(self):
        try:
            return self._beta
        except AttributeError:
            N = self._symX.shape[0]
            self._beta = linalg.inv(self._symX - self._lmbd*np.eye(N)) @ self._X.T @ self._y
            return self._beta

    @property
    def yhat(self):
        try:
            return self._yhat
        except AttributeError:
            self._yhat = self._X @ self.beta
            return self._yhat

    @property 
    def symXInv(self):
        return self._symXInv

    @property
    def sigma_y(self):
        N = self._y.size
        n = self._symX.shape[0]
        return 1/(N-n) * np.sum((self._y-self.yhat)**2)
        
    @property
    def betaVar(self):
        return self._symXInv * self.sigma_y

    def R2Score(self):
        return R2Score(self._y, self.yhat)

    def squared_error(self):
        return squared_error(self._y, self.yhat)

    def predict(self, X):
        return X @ self.beta 

def squared_error(y, yhat):
    n = y.size
    return 1/n*np.sum((y-yhat)**2)


def R2Score(y, yhat):
    n = y.size
    ymean = np.mean(y)
    return 1 - (np.sum((y-yhat)**2)/np.sum((y-ymean)**2))

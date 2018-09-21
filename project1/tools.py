import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
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




def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def get_X_poly2D(x,y,deg):
    X = []
    for i in range(deg + 1):
        for n in range(i+1):
            X.append(x**n * y**(i-n))
    X = np.array(X).T
    return X

def fit_poly2D(x,y,z, deg = 5, lmbd = 0):
    X = get_X_poly2D(x,y,deg)
    regr = Regression(X,z, lmbd = lmbd)
    return regr

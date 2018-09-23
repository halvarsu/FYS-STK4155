import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import linalg

def generate_data(N = 1000, seed = None):
    if not seed is None:
        np.random.seed(seed)

    x = np.random.random(size = N)
    y = np.random.random(size = N)
    noise = 0.01
    from franke import FrankeFunction
    z = FrankeFunction(x,y) + np.random.normal(0,noise,size = x.size)
    return x,y,z, noise

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
            self._beta = linalg.inv(self._symX + self._lmbd*np.eye(N)) @ self._X.T @ self._y
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

    def r2score(self):
        return r2score(self._y, self.yhat)

    def squared_error(self):
        return squared_error(self._y, self.yhat)

    def predict(self, X):
        return X @ self.beta 

def squared_error(y, yhat):
    n = y.size
    return 1/n*np.sum((y-yhat)**2)


def r2score(y, yhat):
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


def k_fold_val(x, y, z, k = 2, lmbd=0):
    """k_fold validation method on ridge regression. lmbd = 0 gives linear
    regression.
    
    Returns
    -------

    r2_train, mse_train : np arrays
        R2-score and Mean Squared Error in-sample.

    r2_test, mse_test : np arrays
        R2-score and Mean Squared Error out-of-sample.
    """
    N = x.size
    if N%k:
        raise ValueError('N must be divisible by k')
    chunk_size = int(N/k)
    
    r2_test = []
    mse_test = []
    r2_train = []
    mse_train = []
    # print("R2score, Squared error ")
    for i in range(k):
        # But it is the same every time?! No, see rolling at end of loop
        x_test = x[:chunk_size]
        y_test = y[:chunk_size]
        z_test = z[:chunk_size]
        x_train = x[chunk_size:]
        y_train = y[chunk_size:]
        z_train = z[chunk_size:]
        
        regr = fit_poly2D(x_train, y_train, z_train)
        design_test = get_X_poly2D(x_test, y_test, deg =5)
        z_pred = regr.predict(design_test)

        r2_train.append(r2score(z_test, z_pred))
        mse_train.append(squared_error(z_test, z_pred))
        r2_test.append(regr.r2score())
        mse_test.append(regr.squared_error())

        # print("{:6.3f}  {:6.3f}".format(r2[i], se[i]), )
        
        x = np.roll(x, chunk_size)
        y = np.roll(y, chunk_size)
        z = np.roll(z, chunk_size)
    return r2_train,mse_train, r2_test, mse_test
        
def bootstrap(x, y, z, k = 2, lmbd=0):
    """WIP"""
    N = x.size
    if N%k:
        raise ValueError('N must be divisible by k')
    chunk_size = int(N/k)
    indexes = np.arange(N)
    
    for i in range(k):
        x_test = x[:chunk_size]
        y_test = y[:chunk_size]
        z_test = z[:chunk_size]
        x_train = x[chunk_size:]
        y_train = y[chunk_size:]
        z_train = z[chunk_size:]
        print(x_test.size, x_train.size)
        
        regr = fit_poly2D(x_train, y_train, z_train)
        X_test = get_X_poly2D(x_test, y_test, deg =5)
        z_pred = regr.predict(X_test)
        
        print(f"train  {regr.r2score():6.3f}  {regr.squared_error():6.3f}", )
        print(f"test   {tools.r2score(z_test,z_pred):6.3f}  {tools.squared_error(z_test, z_pred):6.3f}", )
        
        x = np.roll(x, chunk_size)
        y = np.roll(y, chunk_size)
        z = np.roll(z, chunk_size)

def get_exp_coeffs(beta, deg = 5, print_beta=True):
    i = 0
    exps = {}
    for n in range(deg+1):
        for m in range(deg+1-n):
            if print_beta and (np.abs(beta[i]) > 0):
                x_str = f"x^{n}" if n else ""
                y_str = f"y^{m}" if m else ""
                c_str = x_str + " "  + y_str if (x_str or y_str) else "c"

                print("{:>7}: {:5.2f}".format(c_str, beta[i]))
            exps[(n,m)] = beta[i] 
            i+=1
    import pandas as pd
    df = pd.Series(exps).unstack()
    df.columns.name = 'y_exponent'
    df.index.name = 'x_exponent'
    return df


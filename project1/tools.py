import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import linalg
from franke import FrankeFunction

def generate_data(N = 1000, seed = None, noise = 0.01):
    if not seed is None:
        np.random.seed(seed)

    x = np.random.random(size = N)
    y = np.random.random(size = N)
    
    z = FrankeFunction(x,y) 
    if noise:
        z += np.random.normal(0,noise,size = z.size)
    return x,y,z, noise

class Regression(object):

    """Simple tool for linear, ridge or lasso regression."""

    def __init__(self, X, y, lmbd = 0):
        """TODO: to be defined1. """
        if X.shape[0] != y.shape[0]:
            raise ValueError('y-dim must equal number of rows in design matrix')
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
    def symX(self):
        return self._symX

    @property
    def sigma_y(self):
        N = self._y.size
        p = self._symX.shape[0]
        return 1/(N-p-1) * np.sum((self._y-self.yhat)**2)
        
    @property
    def betaVar(self):
        return self._symXInv * self.sigma_y

    def r2score(self):
        return r2score(self._y, self.yhat)

    def squared_error(self):
        return squared_error(self._y, self.yhat)

    def predict(self, X):
        return (X @ self.beta ).squeeze()

def squared_error(y, yhat):
    n = y.size
    return 1/n*np.sum((y-yhat)**2)


def r2score(y, yhat):
    n = y.size
    ymean = np.mean(y)
    return 1 - (np.sum((y-yhat)**2)/np.sum((y-ymean)**2))


def fit_regr(design_train, z_train, N = 10,noise = 0.1,method = 'ols', lmbd = None):
    """Helper function for looping over methods with standard values for lmbd"""
    method = method.lower()
    from sklearn.linear_model import Lasso
    
    if method == 'ols':
        lmbd = lmbd or 0
        regr = Regression(design_train,z_train, lmbd = 0.0)
    elif method == 'ridge':
        lmbd = lmbd or 0.5
        regr = Regression(design_train,z_train, lmbd = lmbd)
    else:
        lmbd = lmbd or 0.001
        regr = Lasso(alpha = lmbd, fit_intercept = False)
        regr.fit(design_train, z_train)
    return regr



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
    X = (np.array(X).T).squeeze()
    return X

def default_stat(regr, z_test, z_train, design_test,
        design_train):
    """Default statistic function for k-fold validation"""
    z_pred_test = regr.predict(design_test)
    z_pred_train = regr.predict(design_train)

    r2_test   = r2score(z_test, z_pred_test)
    r2_train  = r2score(z_train, z_pred_train)
    mse_test  = squared_error(z_test, z_pred_test)
    mse_train = squared_error(z_train,z_pred_train)
    return r2_train, mse_train, r2_test, mse_test


def k_fold_val(x, y, z, statistic_func= default_stat, return_average = True,
        k = 2, deg= 5, lmbd=0,  method = 'ridge', compare_ground_truth=False):
    """k-fold validation method on regression methods, calculating some
    statistic given by statistic_func, which takes in the regression object,
    z-data for test and train and design-matrices for test and train.
    'method' must be one of OLS, Ridge or Lasso. 'lmbd = 0' assumed for
    OLS. 
    
    Returns
    -------

    output : numpy array of output from output_func, 
             default = (r2_train, mse_train, r2_test, mse_test)

    """
    N = x.size
    # if N%k:
        # raise ValueError('N must be divisible by k')
    if method.lower() not in ['ols','ridge','lasso']:
        raise ValueError('Invalid method flag, {}'.format(method))
    if method.lower() == 'ols' and lmbd != 0:
        raise ValueError('lmbd != 0 does not make sense for OLS.')
    if method.lower() == 'lasso':
        from sklearn.linear_model import Lasso


    N = x.size
    indexes = np.linspace(0,N,k+1, dtype = int)

    # fold sizes. Might vary with 1
    sizes = np.diff(indexes) 
    
    output = []
    for size in sizes:
        # We roll at end of loop
        x_test = x[:size]
        y_test = y[:size]
        if compare_ground_truth:
            z_test = FrankeFunction(x_test,y_test)   
        else:
            z_test = z[:size]

        x_train = x[size:]
        y_train = y[size:]
        z_train = z[size:]
    
        design_train = get_X_poly2D(x_train, y_train, deg =deg)
        design_test = get_X_poly2D(x_test, y_test, deg =deg)
        
        if method.lower() == 'lasso':
            regr = Lasso( alpha = lmbd ,fit_intercept = False)
            regr.fit(design_train, z_train)
        else:
            regr = Regression(design_train,z_train, lmbd = lmbd)

        output.append(statistic_func(regr, z_test, z_train, design_test, design_train))
        
        x = np.roll(x, - size)
        y = np.roll(y, - size)
        z = np.roll(z, - size)

    output = np.array(output) # :(

    if return_average:
        return np.average(output, axis = 0)
    else:
        return output
    

        

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

def bootstrap(x,y,z, lmbd = 0, method = 'ols',rep=50, smplsize = 50, r2_score = False):
    if method.lower() not in ['ols','ridge','lasso']:
        raise ValueError('Invalid method flag, {}'.format(method))
    if method.lower() == 'ols' and lmbd != 0:
        raise ValueError('lmbd != 0 does not make sense for OLS.')
    if method.lower() == 'lasso':
        from sklearn.linear_model import Lasso

    MSE = np.zeros((rep,))
    if r2_score:
        R2 = np.zeros((rep,))    

    indx = np.arange(x.size)
    for r in range(rep):
        rnd1 = np.random.choice(indx, size = smplsize)
        uniq =  np.unique(rnd1)
        
        train_x = x[rnd1]
        train_y = y[rnd1]
        train_z = z[rnd1]

        mask = np.zeros_like(x, dtype=bool)
        mask[uniq] = True
        test_x = x[~mask]
        test_y = y[~mask]
        test_z = z[~mask]

        X_train = get_X_poly2D(train_x,train_y, deg=5)
        X_test = get_X_poly2D(test_x,test_y, deg=5)
        
        if method.lower() == 'lasso':
            regr = Lasso( alpha = lmbd ,fit_intercept = False)
            regr.fit(X_train, train_z)
            z_pred = regr.predict(X_test)
        if method.lower() == 'ols':
            regr = Regression(X_train, train_z)
            z_pred = regr.predict(X_test)
        if method.lower() == 'ridge':
            regr = Regression(X_train, train_z, lmbd=lmbd)
            z_pred = regr.predict(X_test)
        
        MSE[r] = squared_error(z_pred, test_z)
        if r2_score:
            R2[r] = r2score(z_pred,test_z)
    
    if r2_score: 
        return MSE,R2
    else:    
        return MSE

def bootstrap_predict_point(x,y,z,x0 = 0.5, y0 = 0.5, rep=50, deg = 5, 
        lmbd = 0.1, method = 'ridge'):
    """
    Uses bootstrap to find the average value of the model at a given
    point (or points)
    """
    from sklearn.linear_model import Lasso
    points = np.zeros((rep,np.array(x0).size)).squeeze()
    
    indx = np.arange(x.size)
    for r in range(rep):
        rnd1 = np.random.choice(indx, size = x.size)
        uniq =  np.unique(rnd1)
        
        train_x = x[rnd1]
        train_y = y[rnd1]
        train_z = z[rnd1]

        mask = np.zeros_like(x, dtype=bool)
        mask[uniq] = True

        X_train = get_X_poly2D(train_x,train_y, deg=deg)
        X_test = get_X_poly2D(np.array([x0]),np.array([y0]), deg=deg)
        
        if method.lower() == 'lasso':
            regr = Lasso( alpha = lmbd, fit_intercept = False )
            regr.fit( X_train, train_z )
        else:
            regr = Regression( X_train, train_z, lmbd = lmbd )

        points[r] = regr.predict(X_test)
        
    return points

def bias_var_decomposition(points,x0,y0):
    z0 = FrankeFunction(x0,y0)
    var = np.var(points) #squared
    err = np.mean((points-z0)**2) #squared
    bias = err-noise**2-var #squared
    return np.sqrt(err),np.sqrt(var),np.sqrt(bias)

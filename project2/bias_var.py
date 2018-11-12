import numpy as np
from sklearn.linear_model import Lasso
import sys
sys.path.append('../project1/')
import tools as proj1_tools
import time

def bootstrap(X,y, lmbd=0, rep=50,smplsize=1000, method = 'ols', solve_method = 'svd', test_point=False,point=100):
    mse = []
    r2 = []

    indx = np.arange(X.shape[0])
    for i in range(rep):
        #Lrnd = np.random.choice((X.shape[1]), size = smplsize)
        nrnd = np.random.choice((X.shape[0]), size = smplsize)
        #Luniq =  np.unique(Lrnd)
        nuniq = np.unique(nrnd)
        X_train = X[nrnd][:]
        y_train = y[nrnd]
        mask = np.zeros_like(np.arange(X.shape[0]),dtype=bool)
        mask[nuniq] = True
        X_test = X[~mask][:]
        y_test = y[~mask]
        if test_point == False:        
            X_test = X[~mask][:]
            y_test = y[~mask]
        if test_point == True:
            X_test = X[point][:]
            y_test = y[point]
        if method == 'lasso':
            regr = Lasso(alpha=lmbd,fit_intercept = False)
            regr.fit(X_train,y_train)
        if method != 'lasso':
            regr = proj1_tools.Regression(X_train,y_train,lmbd=lmbd,solve_method = 'svd')
        pred_train = regr.predict(X_train)
        pred_test = regr.predict(X_test)
        mse.append(MSE(pred_test,y_test))
        r2.append(r2score(pred_test,y_test))
    if test_point == True:
        return pred_test
    else:
        return mse,r2

def get_bias_and_variance(X,y, k=10, smplsize=1000, rep=10,  method = 'ols', solve_method = 'svd', lmbd = 0):
    """Uses a combination of k-fold (k=50) and bootstrap (50 repetitions)
    to estimate bias and variance. Access to ground truth data, i.e.
    without noise, is necessary."""

    y_pred = k_fold_val(X, y, k = k, method = method,solve_method = solve_method, return_average = False, lmbd = lmbd,returnval='BVD')

    y_pred_mean = np.mean(y_pred)
    var = np.mean(np.var(y_pred))
    bias_squared = np.mean((y[100] - y_pred_mean)**2)
    return var, bias_squared

def MSE(y, yhat):
    n = y.size
    return 1/n*np.sum((y-yhat)**2)


def r2score(y, yhat):
    n = y.size
    ymean = np.mean(y)
    return 1 - (np.sum((y-yhat)**2)/np.sum((y-ymean)**2))

def k_fold_val(X,y, return_average = True,
        k = 2, lmbd=0,  method = 'ridge', solve_method = 'svd', returnval = 'BVD', smplsize=1000,rep=10):
    
    # if N%k:
        # raise ValueError('N must be divisible by k')
    if method.lower() not in ['ols','ridge','lasso']:
        raise ValueError('Invalid method flag, {}'.format(method))
    if method.lower() == 'ols' and lmbd != 0:
        raise ValueError('lmbd != 0 does not make sense for OLS.')

    output = []

    N = X.shape[0]
    indexes = np.linspace(0,N,k+1, dtype = int)
    t1 = time.time()
    # fold sizes. Might vary with 1
    sizes = np.diff(indexes) 
    for i,size in enumerate(sizes):
        t2 = time.time()
        timeleft = (t2-t1)*(sizes.shape[0]-(i+1))
        print('{}/{}, Estimated time: {}s'.format(i+1,sizes.shape[0],round(timeleft)))
        # We roll at end of loop

        X_train = X[:size]
        y_train = y[:size]
        X_test = X[size:]
        y_test = y[size:]
        
        if method.lower() == 'lasso':
            regr = Lasso(alpha = lmbd, fit_intercept = False)
            regr.fit(X_train, y_train)
        if method.lower() != 'lasso':
            regr = proj1_tools.Regression(X_train,y_train, lmbd = lmbd,
                    solve_method = solve_method)
        
        if returnval == 'BVD':
            y_pred = bootstrap(X_train, y_train,point = 100, smplsize=smplsize, rep = rep, lmbd=lmbd, method = method, solve_method = solve_method)
            output.append(y_pred)
        if returnval != 'BVD':
            y_pred_train = regr.predict(X_train)
            y_pred_test = regr.predict(X_test)

            mse_test  = proj1_tools.squared_error(y_test, y_pred_test)
            mse_train = proj1_tools.squared_error(y_train,y_pred_train)
        
            r2_test   = proj1_tools.r2score(y_test, y_pred_test)
            r2_train  = proj1_tools.r2score(y_train, y_pred_train)
        
            output.append([r2_train,mse_train,r2_test,mse_test])
        
        X = np.roll(X, - size,axis=0)
        y = np.roll(y, - size)
        
        t1 = t2        

    output = np.array(output) # :(

    if return_average:
        return np.average(output, axis = 0)
    else:
        return output

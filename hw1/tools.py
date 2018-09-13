import numpy as np
from scipy import linalg


def get_yhat(X,y):
    symX = X.T @ X  # matrix product
    beta = linalg.inv(symX) @ X.T @ y
    yhat = X @ beta
    return yhat, beta


def squared_error(y,yhat):
    n = y.size
    if n != yhat.size:
        raise ValueError('vectors must be same size')
    return 1/n*np.sum((y-yhat)**2)


def R2score(y,yhat):
    n = y.size

    if n != yhat.size:
        raise ValueError('vectors must be same size')

    ymean = np.mean(y)
    return 1 - (np.sum((y-yhat)**2)/np.sum((y-ymean)**2))

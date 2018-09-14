import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def get_beta_ridge(X,y,symX,lmbd):
    I = np.eye(symX.shape[0])
    return linalg.inv(symX + lmbd*I) @ X.T @ y


def ridge_regression(X,y, lmbd = [1]):
    symX = X.T @ X  # matrix product

    beta_ridge = np.zeros((len(lmbd), symX.shape[0]))
    yhat = np.zeros((len(lmbd), y.size))

    for i,l in enumerate(lmbd):
        beta_ridge[i] = get_beta_ridge(X,y,symX,l)
        yhat[i] = X @ beta_ridge[i]
    return yhat, beta_ridge


def R2_analysis(y,yhat_ridge, lambda_values, log_lambda = True):
    from sklearn.metrics import r2_score
    r2_scores = [r2_score(y, yhat_r) for yhat_r in yhat_ridge]
    print('Lambda     r2')
    for l, r2 in zip(lambda_values, r2_scores):
        if log_lambda:
            print("10^{:<2.0f}:  {:6.2f}".format(np.log10(l), r2))
        else:
            print("{:.2f}:  {:.2f}".format(l, r2))

    if log_lambda:
        plt.semilogx(lambda_values, r2_scores)
    else:
        plt.plot(lambda_values, r2_scores)
    plt.xlabel('$\lambda$')
    plt.ylabel('$R2$')
    plt.show()

def plot(x,y,yhat,yhat_ridge,lambda_values):
    plt.figure(figsize=[12,6])
    sort = np.argsort(x)

    plt.plot(x,y, 'o', label = 'data')
    plt.plot(x[sort],yhat[sort], label = 'linear')

    for l,yhat_r in zip(lambda_values, yhat_ridge):
        plt.plot(x[sort], yhat_r[sort],  label = 'ridge, $\lambda = 10^{{{:.0f}}}$'.format( np.log10(l)))
    # plt.plot(x,y, 'o')
    plt.legend(ncol=2)

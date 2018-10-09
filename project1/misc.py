import numpy as np
from sklearn.linear_model import Lasso

        
def bias_var_decomposition(points,x0,y0):
    """outdated"""
    z0 = FrankeFunction(x0,y0)
    var = np.var(points) #squared
    err = np.mean((points-z0)**2) #squared
    bias = err-noise**2-var #squared
    return np.sqrt(err),np.sqrt(var),np.sqrt(bias)


def load_numerated_files(filebase, max_i = 2):
    for i in range(1,max_i+1):
        yield np.loadtxt(filebase.format(i)).T

def gaussian(x, *p):
    a,  mu, sigma, offset = p
    return a*np.exp(-((x-mu)/sigma)**2) + offset

def formatter(v, err):
    log10 = np.log10(err)
    leading = int(err//(10**int(log10)))
    if leading > 2:
        return round(v, int(-log10))
    else:
        return round(v, int(-log10+1))

def format_err(err):
    log10 = np.log10(err)
    leading = int(err//(10**int(log10)))
    if leading > 2:
        return round(err, int(-log10))
    else:
        return round(err, int(-log10+1))
    
def specifier(err):
    num_zeros = int(np.ceil(-np.log10(err)))
    if num_zeros > 0:
        return f".{num_zeros}f"
    else:
        return ".0f"

from sklearn.linear_model import Lasso

        
def bias_var_decomposition(points,x0,y0):
    """outdated"""
    z0 = FrankeFunction(x0,y0)
    var = np.var(points) #squared
    err = np.mean((points-z0)**2) #squared
    bias = err-noise**2-var #squared
    return np.sqrt(err),np.sqrt(var),np.sqrt(bias)

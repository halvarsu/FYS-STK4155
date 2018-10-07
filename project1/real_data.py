import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tools

def load_terrain_data(file):
    data = imread(file)
    n,m = data.shape

    xlin = np.linspace(0,1,n)
    ylin = np.linspace(0,1,m)

    x,y = np.meshgrid(xlin,ylin)
    return x,y,data

def fit_terrain_data(file, lambd, deg):
    """
    
    """

    x, y, data = load_terrain_data(file)
    x = x.ravel()
    y = y.ravel()
    z = data.ravel()

    X = tools.get_X_poly2D(x,y,deg=deg)
    
    regr = tools.Regression(X,z, lmbd = lambd)
    beta = regr.beta
    zpred = regr.predict(X)

    mse = np.sum( (zpred - z)**2 )/x.shape[0]
    r2 = 1 - np.sum( (zpred - z)**2 )/np.sum( (z- np.mean(z))**2 )

    print('MSE', mse)
    print('R2', r2)
    
    return zpred.reshape(data.shape[::-1]), data

def plot_terrain(data):
    """Simple example of how to plot terrain data"""
    m,n = data.shape

    xmesh = np.arange(n)
    ymesh = np.arange(m)

    [x,y] = np.meshgrid(xmesh, ymesh)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(x,y,data,cmap=plt.cm.viridis,linewidth=0)

    # ax = fig.add_subplot(212)
    # ax.pcolormesh(x,y,data,cmap=plt.cm.viridis,linewidth=0)
    plt.show()

if __name__ == "__main__":
    file = 'data/n59_e010_1arc_v3.tif'
    lambd = 0.01
    deg = 5

    zpred, data = fit_terrain_data(file, lambd, deg)
    plot_terrain(zpred.reshape(data.shape))

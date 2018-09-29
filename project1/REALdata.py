import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import tools

def terrain_data(file, lambd, deg):
    data = imread(file)
    n,m = data.shape

    xlin = np.linspace(0,1,n)
    ylin = np.linspace(0,1,m)

    x,y = np.meshgrid(xlin,ylin)

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    X = tools.get_X_poly2D(x,y,deg=deg)
    X = X[0][:][:]
    z = data.flatten()
    
    regr = tools.Regression(X,z, lmbd = lambd)
    beta = regr.beta
    zpred = regr.predict(X)

    mse = np.sum( (zpred - z)**2 )/x.shape[0]
    r2 = 1 - np.sum( (zpred - z)**2 )/np.sum( (z- np.mean(z))**2 )

    print('MSE', mse)
    print('R2', r2)
    
    return zpred,data

def plot_terrain(data, z):
    m,n = data.shape

    xmesh = np.arange(n)
    ymesh = np.arange(m)

    [x,y] = np.meshgrid(xmesh, ymesh)
    
    zshape = z.reshape(data.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_surface(x,y,zshape,cmap=cm.viridis,linewidth=0)
    plt.show()

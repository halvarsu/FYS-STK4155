import numpy as np
import matplotlib.pyplot as plt
import tools
from franke import FrankeFunction, FrankePlot


def contour_plot(regr, N = 400):
    fig = plt.figure(figsize=[6,5])
    xlin = np.linspace(0,1,N)
    ylin = np.linspace(0,1,N)
    xmesh,ymesh = np.meshgrid(xlin,ylin)
    zmesh = FrankeFunction(xmesh,ymesh)

    # plt.pcolormesh(x,y,z)
    plt.contour(xmesh,ymesh,zmesh)
    plt.axis('equal')
    plt.axis([0,1,0,1])

    zlin = zmesh.ravel()

    X = tools.get_X_poly2D(xmesh.ravel(), ymesh.ravel(), deg = 5)
    zhat = regr.predict(X)

    zerr = zmesh - zhat.reshape(N,N)
    shift = 1 - np.max(zerr)/(np.max(zerr) - np.min(zerr))
    shifted_cmap = tools.shiftedColorMap(plt.cm.coolwarm, midpoint= shift , name='shifted')
    
    m = plt.pcolormesh(xmesh,ymesh, zerr, cmap = shifted_cmap)
    cax = plt.colorbar(m)
    plt.xlabel('x')
    plt.ylabel('y')
    cax.set_label('Error')
    return fig, zerr


def plot_data_3D(x,y,z,regr):
    """Accepts raw data and fitted Regression object, and plots the fitted data regr.zhat 
    with deviations from the observed data z, over a mesh of the original Franke function"""
    zhat = regr.yhat
    beta = regr.beta
    
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize = [10,6])
    ax = fig.add_subplot(111,projection = '3d')


    for i in range(x.size):
        ax.plot([x[i],x[i]],
                [y[i],y[i]],
                [z[i],zhat[i]], c = 'r', lw = 0.5, zorder = 10)
    ax.scatter(x,y,zhat)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f(x,y)$')

    ax = FrankePlot(ax)
    ax.view_init(30, 60)

def plot_covar(regr, deg=5, print_beta = False): 
    """Takes in a fitted Regression object and plots the covariance matrix."""
    beta = regr.beta
    std_beta = np.sqrt(np.diag(regr.betaVar))
    i = 0

    if print_beta:
        for n in range(deg+1):
            for m in range(deg+1-n):
                print(f"x**{n} y**{m}  {beta[i]:5.2f} +- {std_beta[i]:5.2f}")
                i+=1

    fig, ax = plt.subplots(1)
    m = ax.imshow(regr.betaVar, origin='upper')

    orders = list(range(deg+1))
    ax.set_xticks(np.cumsum(orders))
    ax.set_xticklabels(orders)

    ax.set_yticks(np.cumsum(orders))
    ax.set_yticklabels(orders)
    ax.xaxis.set_ticks_position('top')

    plt.colorbar(m)
   

"""
Examples on how to plot the data using the Regression functions
"""
import numpy as np
import matplotlib.pyplot as plt
import tools
from franke import FrankeFunction, FrankePlot

def contour_plot(regr, N = 400, plot_err = True, deg=5):
    if plot_err:
        fig, axes = plt.subplots(1,2, figsize=[11,5])
    else:
        fig, ax = plt.subplots(1,figsize=[6,5])
        axes = [ax]

    xlin = np.linspace(0,1,N)
    ylin = np.linspace(0,1,N)
    xmesh,ymesh = np.meshgrid(xlin,ylin)
    zmesh = FrankeFunction(xmesh,ymesh)

    # plt.pcolormesh(x,y,z)
    for ax in axes:
        ax.contour(xmesh,ymesh,zmesh)
        ax.axis('equal')
        ax.axis([0,1,0,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    zlin = zmesh.ravel()

    X = tools.get_X_poly2D(xmesh.ravel(), ymesh.ravel(), deg = deg)
    zhat = regr.predict(X)

    m = axes[0].imshow(zhat.reshape(N,N), extent=[0,1,0,1], origin='lower')
    cax = plt.colorbar(m, ax=axes[0])
    cax.set_label('Z-value')

    if plot_err:
        zerr = zmesh - zhat.reshape(N,N)
        shift = 1 - np.max(zerr)/(np.max(zerr) - np.min(zerr))
        shifted_cmap = tools.shiftedColorMap(plt.cm.coolwarm, midpoint= shift , name='shifted')
    
        m = axes[1].imshow(zerr, cmap = shifted_cmap,
                extent=[0,1,0,1], origin='lower')
        cax = plt.colorbar(m, ax=axes[1])
        cax.set_label('Error')
    fig.tight_layout()
    return fig


def plot_data_3D(x,y,z,zhat,beta):
    """Accepts raw data and fitted Regression object, and plots the fitted data regr.zhat 
    with deviations from the observed data z, over a mesh of the original Franke function"""
    
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize = [10,6])
    ax = fig.add_subplot(111,projection = '3d')


    for i in range(x.size):
        ax.plot([x[i],x[i]],
                [y[i],y[i]],
                [zhat[i],FrankeFunction(x[i],y[i])], c = 'r', lw = 0.5, zorder = 10)
    ax.scatter(x,y,zhat)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f(x,y)$')

    ax = FrankePlot(ax)
    ax.view_init(30, 60)

def plot_covar(regr, deg=5): 
    """Takes in a fitted Regression object and plots the covariance matrix."""
    beta = regr.beta
    std_beta = np.sqrt(np.diag(regr.betaVar))
    i = 0

    fig, ax = plt.subplots(1)
    m = ax.imshow(regr.betaVar, origin='upper')

    orders = list(range(deg+1))
    ax.set_xticks(np.cumsum(orders)-0.5)
    ax.set_xticklabels(orders)

    ax.set_yticks(np.cumsum(orders)-0.5)
    ax.set_yticklabels(orders)
    ax.xaxis.set_ticks_position('top')

    plt.colorbar(m)
   

from itertools import repeat
def add_letters_to_axes(axes, letters = None, pos = repeat([0.01,0.9])):
    if not letters:
        letters = (chr(i) for i in range(65,91)) # large letters A-Z

    for ax, l, [x,y] in zip(axes, letters, pos):
        print(x,y)
        ax.text(x,y,l,transform = ax.transAxes)

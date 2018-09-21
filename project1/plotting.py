import numpy as np
import matplotlib.pyplot as plt
import tools
from franke import FrankeFunction


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

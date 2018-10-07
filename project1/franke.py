from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def FrankePlot(ax = None):
    custom_ax = bool(ax)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig = plt.gcf()

    # Make data.
    x = np.linspace(0,1,80)#arange(0, 1, 0.05)
    y = np.linspace(0,1,80)#arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)

    # Plot the surface.
    surf = ax.plot_wireframe(x, y, z, 
                           linewidth=1, alpha = 0.2, antialiased=False)

    if not custom_ax:
        # Customize the z axis.
        ax.view_init(30, 40)
        ax.set_zlim(-0.10, 1.40)
        ax.set_zlim(-0.10, 1.40)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.zaxis.set_major_locator(LinearLocator(4))
        ax.set_xticks(np.linspace(0,1,3))
        ax.set_yticks(np.linspace(0,1,3))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()

    return ax

if __name__ == "__main__":
    FrankePlot()

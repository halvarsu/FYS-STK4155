import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})


def plot_mispred(data, x, y):
    import matplotlib.colors as colors
    plt.figure(figsize = [9,9])

    ax = plt.gca()
    data_val = 1-data


    cmap_inv = plt.cm.viridis
    cmap = plt.cm.viridis_r
    norm= colors.LogNorm(vmin=data_val.min(), vmax=data_val.max())
    m = ax.matshow(data_val, norm = norm, cmap = cmap)


    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_xlabel('Number of hidden nodes')

    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(['$10^{{{}}}$'.format(f) for f in y])
    ax.set_ylabel('Eta')

    for i, y_val in enumerate(np.arange(len(y))):
        for j, x_val in enumerate(np.arange(len(x))):
            c = "${0:.1f}\\%$".format( 100*(data_val[i,j]))  
            color = cmap( (0.5+norm(data_val[i,j]))%1 )
            ax.text(x_val, y_val, c, va='center', ha='center', color = color)

    cax = plt.colorbar(m)
    cax.set_label('Misprediction rate')

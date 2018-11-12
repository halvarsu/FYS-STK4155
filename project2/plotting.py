import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class PlotWrap(object):

    """Wrapper for plotting functions, to simplify plot-directories and
    rc-params."""

    def __init__(self, filedir = 'figures/', savefigs = True, params = {}, **kwargs):
        """TODO: to be defined1. """
        default_params = {'font.size':14,
                          'image.cmap':'viridis'}
        default_params.update(params)
        default_params.update(kwargs)

        plt.rcParams.update(default_params)
        self.savefigs = savefigs
        self.filedir = filedir
        


    def plot_mispred(self, data, x, y, filename = 'mispred', title = ''):
        fig = plt.figure(figsize = [8,7])

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
        ax.set_ylabel('Learning rate $\eta$')

        for i, y_val in enumerate(np.arange(len(y))):
            for j, x_val in enumerate(np.arange(len(x))):
                c = "${0:.1f}\\%$".format( 100*(data_val[i,j]))  
                color = cmap( (0.5+norm(data_val[i,j]))%1 )
                ax.text(x_val, y_val, c, va='center', ha='center', color = color)

        cax = plt.colorbar(m)
        cax.set_label('Misprediction rate')
        ax.set_title(title)
        
        fig.tight_layout()
        if self.savefigs:
            plt.savefig(self.filedir + filename + '.pdf')

    def plot_eta_compare(self, super_df, eta_val = -1.5, filename = 'eta_compare'):
        fig, axes = plt.subplots(3, figsize = [6,7], sharex = True)

        lines = []
        handles = []

        for (hidden_act, df), ax in zip(super_df.groupby('hidden_act'), axes):
            n_hidden_values = np.unique(df.nhidden)
            colors = plt.cm.viridis(np.linspace(0,1,len(n_hidden_values)))


            item = df.groupby('eta_val').get_group(eta_val)
            ax.set_title(hidden_act)

            for (nhidden, item), c in zip(item.groupby('nhidden'),colors):
                y =  1-np.array(item['accuracy'].values[0])
                eta_val = item['eta_val'].iloc[0]
                l, = ax.loglog(np.arange(y.size) + 1  , y, c= c, linestyle = '-')

                if hidden_act =='relu':
                    # only add one set of lines to legend
                    lines.append(l)
                    handles.append('$N_h = {}$'.format(nhidden))


            ax.grid()
            # ax.set_ylabel('Misprediction rate')
        axes[2].set_xlabel('Epochs')
        fig.text(0.02, 0.5, 'Misprediction rates', va='center',
                rotation='vertical', fontsize = 16)

        plt.figlegend(lines, handles, loc = (0.75,0.2))
        fig.tight_layout()

        if self.savefigs:
            plt.savefig(self.filedir + filename + '.pdf')

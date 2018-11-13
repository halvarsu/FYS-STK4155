import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
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
        

    def plot_mispred(self, *args, **kwargs):
        kwargs.update({'filename':'mispred',
                'value':'accuracy'})
        self.plot_score(*args, **kwargs)

    def plot_score(self, data, x, y, filename = 'score', title = '', value
            = 'accuracy', lognorm = True, cmap = None):

        assert value in ['accuracy', 'r2']

        fig = plt.figure(figsize = [8,7])

        ax = plt.gca()

        data_val = 1-data if value == 'accuracy' else data



        
        cmap = cmap or plt.cm.viridis_r

        if lognorm:
            norm= colors.LogNorm(vmin=data_val.min(), vmax=data_val.max())
        else:
            norm= colors.Normalize(vmin=np.nanmin(data_val),
                    vmax=np.nanmax(data_val))
        m = ax.matshow(data_val, norm = norm, cmap = cmap)


        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.set_xlabel('Number of hidden nodes')

        ax.set_yticks(np.arange(len(y)))
        ax.set_yticklabels(['$10^{{{}}}$'.format(f) for f in y])
        ax.set_ylabel('Learning rate $\eta$')

        for i, y_pos in enumerate(np.arange(len(y))):
            for j, x_pos in enumerate(np.arange(len(x))):
                if value  == 'accuracy':
                    c = "${0:.1f}\\%$".format( 100*(data_val[i,j]))   
                else:
                    c = "${0:.1f}$".format( data_val[i,j])   
                color = cmap( (0.5+norm(data_val[i,j]))%1 )
                ax.text(x_pos, y_pos, c, va='center', ha='center', color = color)

        cax = plt.colorbar(m)
        cax_label = 'Misprediction rate' if value == 'accuracy' else 'R2 score'
        cax.set_label(cax_label)
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

    def plot_best(self, df_list, filename = 'best_runs', r2 = False):
        best = pd.concat([df.iloc[df.max_accuracy.idxmax()] for df in df_list],axis = 1).T
        best = best.set_index('hidden_act', drop = True)

        for hact, item in best.T.items():
            if r2:
                y = np.array(item['accuracy']) 
                epoch = np.arange(len(y))
                plt.semilogx(epoch, y, label = hact)
            else:   
                y = 1-np.array(item['accuracy'])
                epoch = np.arange(len(y))
                plt.loglog(epoch, y, label = hact)
            
        plt.xlabel('Epoch')

        plt.ylabel('Misprediction' if not r2 else 'R2 Score')
        plt.grid()
        plt.legend()
        if self.savefigs:
            plt.savefig(self.filedir + filename + '.pdf')

def gen_best_table(df_list, max_acc_name = 'Accuracy'):
    """not a plot-function, but close"""
    values = {}


    
    for df in df_list:
        name = df.hidden_act.iloc[0]
        temp = []

        for nhidden, item in df.groupby('nhidden'):
            indx = item.max_accuracy.idxmax
            best = item.loc[indx]

            temp.append(best)
        values[name] = temp
        
    dfs = [pd.DataFrame(val) for val in values.values()]
    best_df = pd.concat(dfs, ignore_index = True).set_index(['hidden_act', 'nhidden'])#.stac
    best_df = best_df[['eta_val','max_accuracy','optimal_epoch']]
    best_df = best_df.swaplevel(axis = 0).unstack()
    best_df.columns = best_df.columns.rename("", level=1)
    best_df.index = best_df.index.rename("$N_h$")
    best_df['eta_val'] = best_df['eta_val'].apply(lambda x:x.apply('10^{{{}}}'.format))
    best_df= best_df.swaplevel(axis=1).sort_index(1)
    best_df = best_df.rename(columns={'eta_val':'$\eta$','max_accuracy':max_acc_name, 'optimal_epoch':'Epoch',
                           'sigmoid':'sig'})

    return best_df.to_latex(float_format = '%.3f', escape = False)

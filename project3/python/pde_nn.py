import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from tqdm import trange

def relative_difference(x,xhat):
    return np.abs(x-xhat)/x

def mean_squared_error(x,xhat):
    return np.mean((x - xhat)**2)

def r2_score(x,xhat):
    return 1 - np.sum((x - xhat)**2) / np.sum((x - np.mean(x))**2)


class DiffusionNN(object):

    """Uses under the hood Tensorflow to solve the 1d partial differential
    equations."""

    def __init__(self, lmbd = 0, 
            Nx = 20, x_bounds = [0, 1],
            Nt = 20, t_bounds = [0, 1], 
            initial_condition = lambda x: tf.sin(np.pi*x),
            verbose = True,
            ):
        """
        Parameters:
        -----------

        """
        # self.sizes = sizes
        self.num_layers = 0
        self.initialize(Nx, x_bounds, Nt, t_bounds)
        self.ic = initial_condition
        self.verbose = True
        self.is_run = False

    def initialize(self, Nx = 20, x_bounds = [0, 1], 
                         Nt = 20, t_bounds = [0, 1]):

        tf.reset_default_graph()
        self.is_run = False
        self.X, self.T  = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], Nx),
                                      np.linspace(t_bounds[0], t_bounds[1], Nt))

        self.t      = tf.reshape(tf.convert_to_tensor(self.T), shape = (-1, 1))
        self.x      = tf.reshape(tf.convert_to_tensor(self.X), shape = (-1, 1))
        domain      = tf.concat([self.x, self.t],-1)
        self.prev_layer = domain


    def add_layer(self, size, act_func):
        self.prev_layer = tf.layers.dense(self.prev_layer, size, 
                                          activation = act_func)
        return self.prev_layer

    def h1(self, x,t):
        return (1-t)*self.ic(x)

    def h2(self, x,t, N):
        return x*(1-x)*t*N

    def run(self, learning_rate, iterations, optimizer = tf.train.AdamOptimizer,
            **kwargs):
        """
        Learning rate : float
            learning rate input to optimizer

        iterations : int
            number of iterations to train

        optimizer : tf optimizer
            function which takes a learning rate, plus arguments given by
            **kwargs

        **kwargs : any
            additional arguments for optimizer
            """
        dnn_output  = tf.layers.dense(self.prev_layer, 1)

        x           = self.x
        t           = self.t

        g_trial     = self.h1(x, t) + self.h2(x, t, dnn_output)
        g_trial_dt, = tf.gradients(g_trial,t)
        g_trial_d2x,= tf.gradients(tf.gradients(g_trial,x),x)
        g_analytic  = tf.exp(-np.pi**2*t) * tf.sin(np.pi*x)

        loss        = tf.losses.mean_squared_error(g_trial_dt, g_trial_d2x)

        minimize    = optimizer(learning_rate, **kwargs).minimize(loss)

        init        = tf.global_variables_initializer()
        self.meshes = []

        with tf.Session() as sess:
            init.run()

            self.g_t       = g_trial.eval()
            self.g_a       = g_analytic.eval()
            self.cost = [loss.eval()]
            self.mse  = [mean_squared_error(self.g_a, self.g_t)]

            _range = trange if self.verbose else range
            for i in _range(iterations):
                sess.run(minimize)

                self.g_t = g_trial.eval()
                self.cost.append(loss.eval())
                self.mse.append(mean_squared_error(self.g_a, self.g_t))
                if i % 100 == 0:
                    self.meshes.append(self.g_t.reshape(self.X.shape))

            self.nn_out = dnn_output.eval().reshape(self.X.shape)

        if self.verbose:
            print('Done')
            print('Mean squared error   = ', self.mse[-1])
            print('Cost                 = ', self.cost[-1])

        self.is_run = True
        return self.cost[-1]


    def plot_error(self, ax = None):
        assert self.is_run
        ax = ax or plt.gca()
        ax.loglog(self.mse, label = 'MSE')
        ax.loglog(self.cost, label = 'Cost')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.legend()
        return ax


    def plot_3d(self):
        assert self.is_run
        figsize = [5,5]
        X, T = self.X, self.T

        fig = plt.figure(figsize=figsize)
        ax1 = fig.gca(projection='3d')
        ax1.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
        s = ax1.plot_surface(X,T,g_dnn_mesh,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax1.set_xlabel('Position $x$')
        ax1.set_ylabel('Time $y$');

        fig = plt.figure(figsize=figsize)
        ax2 = fig.gca(projection='3d')
        ax2.set_title('Analytical solution')
        s = ax2.plot_surface(X,T,g_analy_mesh,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax2.set_xlabel('Position $x$')
        ax2.set_ylabel('Time $t$');

        fig = plt.figure(figsize = figsize)
        ax3 = fig.gca(projection = '3d')
        ax3.set_title('Difference')
        s = ax3.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax3.set_xlabel('Position $x$')
        ax3.set_ylabel('Time $t$')


        fig = plt.figure(figsize = figsize)
        ax4 = fig.gca(projection = '3d')
        ax4.set_title('NN out')
        s = ax4.plot_surface(X,T,nn_out,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax4.set_xlabel('Position $x$')
        ax4.set_ylabel('Time $t$')

        return ax1, ax2, ax3, ax4

    def plot_solutions(self, t_values = [0,0.2,0.85, 1]):
        """
        Plots solution at selected t_values.
        """
        pass

    def animate_mesh(self):
        diff = np.abs(g_dnn_mesh - g_analy_mesh)
        # for i,mesh in enumerate(meshes):
        #     fig = plt.figure(figsize = figsize)
        #     ax = fig.gca(projection = '3d')
        #     ax.set_title('NN out')
        #     s = ax.plot_surface(X,T,np.abs(mesh - g_analy_mesh),linewidth=0,antialiased=False,cmap=cm.viridis)
        #     ax.set_xlabel('Position $x$')
        #     ax.set_ylabel('Time $t$')
        #     ax.set_zlim([0,1])
        #     ax.view_init(30)
        #     plt.savefig('test{}.png'.format(i))
        # pass


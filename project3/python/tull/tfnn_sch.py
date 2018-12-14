import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

## Set up the parameters for the network
num_hidden_neurons = [128]
#num_hidden_neurons = [90]
num_hidden_layers = np.size(num_hidden_neurons) 
num_iter = 20000
lmb = 0.01
learning_rate = 1e-3
batch_size = 100
act_func = 'sigmoid'

animate = False

### Can be shortened
model = tf.dnn

## Data
Nx = 80
x_max = 5
x_np = np.linspace(0, x_max, Nx)

## The construction phase
x     = tf.reshape(tf.convert_to_tensor(x_np),shape=(-1,1))
zeros = tf.convert_to_tensor(np.zeros(x.shape))
points = x

lmbd = 3


#####dnn
previous_layer = points
for l in range(num_hidden_layers):
    current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
    previous_layer = current_layer
dnn_output = tf.layers.dense(previous_layer, 1)

#####loss
# g_trial = x * (x_max - x) * dnn_output
g_trial = x * tf.exp(-2*x) * dnn_output
g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
loss = tf.losses.mean_squared_error(zeros, g_trial_d2x[0] + (lmbd - x**2) * g_trial)

#####train
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# g_analytic = tf.sin(np.pi*x)
g_dnn = None

## The execution phase
draw_interval = 1000
nframes = int(num_iter/draw_interval)
wf_arr = np.zeros((Nx, nframes+1))


with tf.Session() as sess:
    init.run()
    for i in range(num_iter):

        sess.run(traning_op)
        if i % draw_interval == 0:
            print(loss.eval(), i//draw_interval)
            if animate:
                g_dnn = g_trial.eval()
                wf_arr[:,i//draw_interval] = (g_dnn.ravel())**2

    g_dnn = g_trial.eval()

## Compare with the analutical solution

# Plot the results
figs=(5,5)

fig = plt.figure(figsize=figs)
ax = fig.add_subplot(111)
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
s, = ax.plot(x_np,(g_dnn.ravel())**2)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Wavefunc $y$');

def update(i):
    s.set_ydata(wf_arr[:,i%nframes])
    ax.set_title(i%nframes)
    return s,

if animate:
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig,update, interval=30)
plt.savefig('test1.pdf')
plt.show()

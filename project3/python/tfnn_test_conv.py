import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

## Set up the parameters for the network
num_hidden_neurons = [100, 100]
#num_hidden_neurons = [90]
num_hidden_layers = np.size(num_hidden_neurons) 
num_iter = 100000
lmb = 0.01
learning_rate = 0.05
epochs = 100
batch_size = 100
act_func = 'sigmoid'

## Data
Nx = 10
x_np = np.linspace(0,1,Nx)

Nt = 20
t_np = np.linspace(0,2,Nt)

X,T = np.meshgrid(x_np, t_np)

x = X.ravel()
t = T.ravel()


## The construction phase
# zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
zeros = tf.reshape(tf.convert_to_tensor(np.zeros(X.shape)),
        shape=(-1,Nt,Nx,1))
print(zeros.shape)
# x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
# t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(T),shape=(-1,Nt,Nx,1))
x = tf.reshape(tf.convert_to_tensor(X),shape=(-1,Nt,Nx,1))

# import sys
# sys.exit()

points = tf.concat([x,t],1)
X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)


## functions
def u(x):
     return tf.sin(np.pi*x)

def h1(point):
    x,t = point
    return (1-t)*u(x)

def trial(x,P):
    x,t = point
    return h1(point) + x*(1-x) * t * N(x,t,P)

#def d_trial(x,P):
#    return 

#def costfunction(x,P,gamma):
#    N = x.shape[0]
#    return (1/N) * (d_trial(x,p)-trial(x,P))@(d_trial(x,p)-trial(x,P)).T #???

#####dnn
previous_layer = points
# for l in range(num_hidden_layers):
    # current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
    # previous_layer = current_layer

conv1 = tf.layers.conv2d(
      inputs=points,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.sigmoid)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
pool1_flat = tf.reshape(pool1, [-1, Nt * Nx * 8])

#dense = tf.layers.dense(pool1_flat, 1000, activation=tf.nn.sigmoid)
dnn_output = tf.layers.dense(pool1_flat, 1)

print(dnn_output)

#####loss
print(x.shape, t.shape, dnn_output.shape)
g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
g_trial_dt = tf.gradients(g_trial,t)
g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
print(g_trial)

loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])
#####train
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
g_analytic = tf.exp(-np.pi**2 *t) * tf.sin(np.pi*x)
g_dnn = None

## The execution phase
with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        sess.run(traning_op)
        # If one desires to see how the cost function behaves during training
        if i % 100 == 0:
               print(loss.eval())
    g_analytic = g_analytic.eval()
    g_dnn = g_trial.eval()

## Compare with the analutical solution
diff = np.abs(g_analytic - g_dnn)
print('Max absolute difference between analytical solution and TensorFlow DNN = ',np.max(diff))
G_analytic = g_analytic.reshape((Nt,Nx))
G_dnn = g_dnn.reshape((Nt,Nx))
diff = np.abs(G_analytic - G_dnn)


# Plot the results
figs=(5,5)
X,T = np.meshgrid(x_np, t_np)

fig = plt.figure(figsize=figs)
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Time $y$');

fig = plt.figure(figsize=figs)
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(X,T,G_analytic,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Time $t$');

fig = plt.figure(figsize=figs)
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$')

## Take some 3D slices
indx1 = 0
indx2 = int(Nt/2)
indx3 = Nt-1
t1 = t_np[indx1]
t2 = t_np[indx2]
t3 = t_np[indx3]
# Slice the results from the DNN
res1 = G_dnn[indx1,:]
res2 = G_dnn[indx2,:]
res3 = G_dnn[indx3,:]
# Slice the analytical results
res_analytical1 = G_analytic[indx1,:]
res_analytical2 = G_analytic[indx2,:]
res_analytical3 = G_analytic[indx3,:]
# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t1)
plt.plot(x_np, res1)
plt.plot(x_np,res_analytical1)
plt.legend(['dnn','analytical'])
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x_np, res2)
plt.plot(x_np,res_analytical2)
plt.legend(['dnn','analytical'])
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x_np, res3)
plt.plot(x_np,res_analytical3)
plt.legend(['dnn','analytical'])
plt.show()

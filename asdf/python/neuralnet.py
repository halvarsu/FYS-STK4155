import numpy as np
import random


class NeuralNet(object):
    
    def __init__(self, sizes=[], act_func = 'sigmoid', alpha = 0):
        """
        Neural network, where sizes is a list where the length of the list 
        will be the number of layers, with each element corresponding the number
        of neurons in each layer.

        Activation functions:
            - sigmoid
            - step
            - softsign
            - tanh
            - ReLU (with alpha!=0 for leaky ReLU)
            - ELU
        """
        self.sizes = sizes
        self.num_layers = len(sizes) 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.act_func = act_func
        self.alpha = alpha
    
    def __del__(self):
        self.sizes = []

    def add_layer(self, n):
        self.sizes.append(n)
    
    def backpropagate(self, x,y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        out = x
        outputs = [x] # List of activations
        zs = [] # List of weighted inputs
        for b, w in zip(self.biases, self.weights):
            z = get_out(w,x,b)
            zs.append(z)
            out = activation(z)
            outputs.append(out)
        # Start backward [-1]=> last entry
        dC = d_cost(out[-1], y) * d_activation(zs[-1]) 
        grad_b[-1] = dC
        grad_w[-1] = dC@outputs.transpose()
        for l in range(2, self.num_layers):
            d_act = d_activation(zs[-1])
            dC = np.dot(self.weights[-l+1].transpose(), dC) * d_act
            grad_b[-l] = dC
            grad_w[-l] = dC@outputs.transpose()
        return (grad_b, grad_w)    

    def stoc_grad_descent(self, train_data, epochs, batch_size, eta):
        if type(train_data) != type(np.zeros(1)):
            print('Make sure train_data is of type np.ndarray')
        else:
            n = train_data.shape[0]
            for j in range(epochs):
                random.shuffle(train_data)
                batches = [train_data[k:k+batch_size]
                            for k in range(0,n,batch_size)]
            for batch in batches:
                self.update_batch(batch,eta)
            #print('Epoch {0} complete'.format(j))
    
    def update_batch(self,batch,eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            d_grad_b, d_grad_w = self.backpropagate(x,y)
            grad_b = [nb+dnb for nb,dnb in zip(grad_b,d_grad_b)]
            grad_w = [nw+dnw for nw,dnw in zip(grad_w,d_grad_w)]
        self.weights = [w-(eta/batch.shape[0])*nw for w,nw in zip(self.weights,grad_w)]
        self.biases = [b-(eta/batch.shape[0])*nb for b,nb in zip(self.biases,grad_b)]

    def feed_forward(self, out):
        for b, w in zip(self.biases, self.weights):
            out = get_out(w,out,b)
        return out

    def get_out(w,out, b):
        return w@out + b
    
    def d_cost(self, out, y):
        return(out - y)
    
    def activation(x):
        if self.act_func.lower() == 'sigmoid':  
            return 1.0/(1.0 + np.exp(-x))
        if self.act_func.lower() == 'step':
            if x < 0: return 0
            else: return 1
        if self.act_func.lower() == 'softsign':
            return x/1+np.abs(x)
        if self.act_func.lower() == 'tanh':
            return np.tanh(x)
        if self.act_func.lower() == 'relu':
            if x < 0: return self.alpha*x
            else: return x
        if self.act_func.lower() == 'elu':
            if x < 0: return self.alpha*(np.exp(x)-1)
            else: return x

    def d_activation(x):
        if self.act_func.lower() == 'sigmoid':  
            return activation(x)*(1.0 - activation(x))
        if self.act_func.lower() == 'step':
            return 0
        if self.act_func.lower() == 'softsign':
            return activation(x)**2
        if self.act_func.lower() == 'tanh':
            return 1 - activation(x)**2
        if self.act_func.lower() == 'relu':
            if x < 0: return self.alpha
            else: return 1
        if self.act_func.lower() == 'elu':
            if x < 0: return activation(x) + self.alpha
            else: return 1

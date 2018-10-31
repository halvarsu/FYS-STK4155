import numpy as np
import random


class NeuralNet(object):
    
    def __init__(self, sizes=[], act_func = 'sigmoid', alpha = 1):
        """
        Neural network, where sizes is a list where the length of the list 
        will be the number of layers including the input layer, with each
        element corresponding the number of neurons in each layer.

        act_func : str, or list of str

        Available activation functions:
            - sigmoid
            - step
            - softsign
            - tanh
            - ReLU (with alpha!=0 for leaky ReLU)
            - ELU
        """
        self.sizes = sizes
        self.num_layers = len(sizes) 
        self.biases  = [0.1 * np.random.randn(y) for y in sizes[1:]]
        self.weights = [0.1 * np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        if type(act_func) == str:
            act_func = [act_func]

        if len(act_func) == 1:
            a = ActivationFunction(act_func[0])
            self.act_funcs = [a for _ in sizes[:-1]]
        elif len(act_func) == self.num_layers - 1:
            self.act_funcs = [ActivationFunction(s) for s in act_func]
        else:
            msg = 'act_func must be str or list of strings for each layer (except input)'
            raise TypeError(msg)
        # self.act_funcs.insert(0, ActivationFunction('identity'))

        self.alpha = alpha
    
    def __del__(self):
        self.sizes = []

    def add_layer(self, n):
        self.sizes.append(n)
    
    def backpropagate(self, x,y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        zs, outputs = self.feed_forward(x)

        # Start backward [-1]=> last entry
        d_act = self.act_funcs[-1].deriv(zs[-1])
        delta = self.d_cost(outputs[-1], y) * d_act

        grad_b[-1][:] = delta
        if len(outputs) > 1:
            grad_w[-1][:] = np.outer(delta , outputs[-2])
        else:
            grad_w[-1][:] = np.outer(delta, x)

        for l in reversed(range(0, self.num_layers-2)): # l = L-1,...,0 
            d_act = self.act_funcs[l].deriv(zs[l])
            delta = (self.weights[l+1].T @ delta) * d_act
            grad_b[l][:] = delta

            if l > 0:
                grad_w[l][:] = np.outer(delta, outputs[l-1])
            else:
                grad_w[l][:] = np.outer(delta, x)
        return (grad_b, grad_w)    

    
    def update_batch(self,batch,eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        n = len(batch)

        for x,y in batch:
            d_grad_b, d_grad_w = self.backpropagate(x,y)

            # print([eta * np.mean(dw/w) for w,dw in zip(self.weights, d_grad_w)])
            grad_w =  [nw+dnw for nw,dnw in zip(grad_w,d_grad_w)]
            grad_b =  [nb+dnb for nb,dnb in zip(grad_b,d_grad_b)]

        self.weights = [w-(eta/n)*nw for w,nw in zip(self.weights,grad_w)]
        self.biases = [b-(eta/n)*nb for b,nb in zip(self.biases,grad_b)]

    def feed_forward(self, inp):

        z = self.weights[0] @ inp + self.biases[0]
        out = self.act_funcs[0](z)
        outputs = [out] # List of activations
        zs = [z]        # List of weighted z's
        i = 1
        for act_func, b, w in zip(self.act_funcs[1:], self.biases[1:], self.weights[1:]):
            z = w @ out + b 
            out = act_func(z)
            zs.append(z)
            outputs.append(out)
            i+=1
        return zs, outputs

    def get_out(self, w,out, b):
        return w @ out + b
    
    def d_cost(self, out, y):
        return(out - y)
    

class FunctionBase:
    def __init__(self, func_name):
        if type(func_name) != str:
            raise TypeError('func_name must be string')
        self.Function = func_name

    @property
    def Function(self):
        return self._func_name

    @Function.setter
    def Function(self, func_name):
        """Set current function and its derivative to the given func_name"""
        try:
            self._func = getattr(self, func_name)
            self._deriv = getattr(self, 'd_' + func_name)
            self._func_name = func_name
        except ValueError:
            print('func_name must be string')
            raise 

    def __call__(self, x):
        return self.func(x)

    def func(self, x):
        return self._func(x)

    def deriv(self, x):
        return self._deriv(x)

class CostFunction(FunctionBase):

    """Docstring for CostFunction. """

    def __init__(self, cost_func):
        FunctionBase.__init__(self, func_name = cost_func)

    def linear_regression(self, out, y):
        return 0.5*np.sum((out - y)**2)

    def logistic_regression(self, out, y):
        raise NotImplementedError

    def d_linear_regression(self, out, y):
        return out - y

    def d_logistic_regression(self, out, y):
        raise NotImplementedError

class ActivationFunction(FunctionBase):

    """Docstring for ActivationFunction. """

    def __init__(self, activation, alpha = 1):
        FunctionBase.__init__(self, func_name = activation)
        self._alpha = alpha
        

    def identity(self,x):
        return x

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def step(self, x):
        return np.int64(x >= 0)

    def softsign(self, x):
        return x/1+np.abs(x)

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return self._alpha * np.maximum(x, 0)
    
    def elu(self, x):
        return np.choose(x < 0, [x, self._alpha * (np.exp(x)-1)])

    def d_identity(self,x):
        return 1

    def d_sigmoid(self, x):
        return self.sigmoid(x)*(1.0 - self.sigmoid(x))

    def d_step(self, x):
        return 0

    def d_softsign(self, x):
        return activation(x)**2

    def d_tanh(self, x):
        return 1 - activation(x)**2

    def d_relu(self, x):
        return self._alpha * (x > 0)

    def d_elu(x):
        return np.choose(x > 0, [1, self._alpha * np.exp(x)])


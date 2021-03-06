import numpy as np
import random

# class NeuralNetClassification(NeuralNet):

#     def __init__(self, sizes=[], act_func = 'sigmoid', alpha = 1):
#         super().__init__(sizes, act_func, alpha, net_type = 'classification')


class NeuralNet(object):
    
    def __init__(self, sizes=[], act_func = 'sigmoid', alpha = 1,
            net_type = 'regression', lmbd = 0):
        """
        Neural network, where sizes is a list where the length of the list 
        will be the number of layers including the input layer, with each
        element corresponding the number of neurons in each layer.

        act_func : str, or list of str

        net_type : str
            must be either regression or classifier

        alpha : float
            saturation for elu/gradient of relu activation functions

        lmbd : float
            regularization parameter
            

        Available activation functions:
            - sigmoid
            - step
            - softsign
            - tanh
            - ReLU (with alpha!=0 for leaky ReLU)
            - ELU
        """
        if net_type not in ['regression', 'classifier']:
            raise TypeError("invalid net_type flag '%s', must be one of 'regression','classifier'"
                    % net_type)
        elif net_type != 'classifier' and 'softmax' in act_func:
            import warnings
            warnings.warn('Are you sure you want softmax when using regression?') 
        self.net_type = net_type
        self.sizes = sizes
        self.num_layers = len(sizes) 
        self.biases  = [0.1 * np.random.randn(y) for y in sizes[1:]]
        self.weights = [0.1 * np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.lmbd = lmbd

        if type(act_func) == str:
            act_func = [act_func]

        if len(act_func) == 1:
            a = ActivationFunction(act_func[0])
            self.act_funcs = [a for _ in sizes[:-1]]
        if len(act_func) == 2:
            hidden = ActivationFunction(act_func[0])
            output = ActivationFunction(act_func[1])
            self.act_funcs = [hidden for _ in sizes[:-2]] + [output]
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
    
    def backpropagate(self, x, y):
        if np.size(y) != self.sizes[-1]:
            raise ValueError('target must have same size as output layer')
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        zs, outputs = self.feed_forward(x)

        # Start backward [-1]=> last entry

        if self.net_type == 'regression':
            d_act = self.act_funcs[-1].deriv(zs[-1])
            delta = self.d_cost(outputs[-1], y) * d_act
        elif self.net_type == 'classifier':
            delta = outputs[-1] - y

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

    def backpropagate_vectorized(self, x, y, vector_input = True):
        from project2_tools import add_outer_products
        y_shape = np.shape(y)
        x_shape = np.shape(x)
        if vector_input:
            if y_shape[0] != x_shape[0]:
                raise ValueError('x and y must have same first dimension with vector_input')
            if self.sizes[-1] == 1:
                if len(y_shape) == 1:
                    pass
                elif y_shape[1] == 1:
                    pass
                else:
                    raise ValueError('y must have same last dimension as output layer with vector_input')
            else:
                if y_shape[-1] != self.sizes[-1]:
                    raise ValueError('y must have same last dimension as output layer with vector_input')
            n_sets = x.shape[0]
        else:
            if self.sizes[-1] == 1:
                if len(y_shape):
                    raise ValueError('y must have same size as output layer')
            else:
                if y_shape[0] != self.sizes[-1]:
                    raise ValueError('y must have same size as output layer')
            n_sets = 1

        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        zs, outputs = self.feed_forward_vectorized(x)

        # Start backward [-1]=> last entry

        if self.net_type == 'regression':
            d_act = self.act_funcs[-1].deriv(zs[-1])
            delta = self.d_cost(outputs[-1], y) * d_act
        elif self.net_type == 'classifier':
            delta = outputs[-1] - y

        grad_b[-1] = np.mean(delta, axis = 0)
        if len(outputs) > 1:
            grad_w[-1] = np.dot(delta.T, outputs[-2]) * 1/n_sets
        else:
            grad_w[-1] = np.dot(delta.T, x) * 1/n_sets

        for l in reversed(range(0, self.num_layers-2)): # l = L-1,...,0 
            d_act = self.act_funcs[l].deriv(zs[l])
            delta = np.einsum('...ij,...i->...j',self.weights[l+1], delta) * d_act
            # delta = np.mean(np.einsum('...ij,...i->...j',self.weights[l+1], delta) * d_act, axis = 0)
            grad_b[l] = np.mean(delta, axis = 0)

            if l > 0:
                grad_w[l] = np.dot(delta.T, outputs[l-1]) * 1/n_sets
            else:
                grad_w[l] = np.dot(delta.T, x) * 1/n_sets

        return (grad_b, grad_w)    
    
    def update_batch_vectorized(self, x, y,eta):
        n = len(x)

        grad_b, grad_w = self.backpropagate_vectorized(x,y, vector_input=True)
        # grad_b = np.sum(grad_b_arr, axis = 0)
        # grad_w = np.sum(grad_w_arr, axis = 0)

        self.weights = [w-(eta/n)*nw for w,nw in zip(self.weights,grad_w)]
        self.biases = [b-(eta/n)*nb for b,nb in zip(self.biases,grad_b)]

    def update_batch(self,batch,eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        n = len(batch)

        for x,y in batch:
            d_grad_b, d_grad_w = self.backpropagate(x,y)

            grad_w =  [nw+dnw for nw,dnw in zip(grad_w,d_grad_w)]
            grad_b =  [nb+dnb for nb,dnb in zip(grad_b,d_grad_b)]

        self.weights = [w*(1-self.lmbd)-(eta/n)*nw for w,nw in zip(self.weights,grad_w)]
        self.biases = [w*(1-self.lmbd)-(eta/n)*nb for b,nb in zip(self.biases,grad_b)]

    def feed_forward_vectorized(self, inputs):
        # tensordot and matmul ~ equal time

        # z = np.zeros((inputs.shape[0], self.weights[0].shape[0]))
        # for i,data in enumerate(inputs):
        #     z[i] = self.weights[0] @ data + self.biases[0]
        #     p
        # z2 = np.matmul(self.weights[0], inputs.T).T + self.biases[0]
        # z3 = np.einsum('...ij,...j->...i',self.weights[0] , inputs) + self.biases[0]
        # z = np.tensordot(inputs, self.weights[0], axes = [1,1]) + self.biases[0]
        # LOL why not use dot, such speed, much simple
        z = np.dot(inputs, self.weights[0].T) + self.biases[0]

        out = self.act_funcs[0](z)
        zs = [z]        # List of weighted z's
        outputs = [out] # List of activations
        i = 1
        for act_func, b, w in zip(self.act_funcs[1:], self.biases[1:], self.weights[1:]):
            # z = np.einsum('...ij,...j->...i', w, out) + b
            z = np.tensordot(out, w, axes = [1,1]) + b
            out = act_func(z)
            zs.append(z)
            outputs.append(out)
            i+=1
        return zs, outputs

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
        return (out - y)

    def accuracy(self, test_inputs, test_targets):
        """
        Returns accuracy of net for given test data set.
        """
        from project2_tools import from_onehot

        _, outputs = self.feed_forward_vectorized(test_inputs)
        
        mle = np.argmax(outputs[-1], axis = 1)
        target = from_onehot(test_targets)
        accuracy = np.mean(mle == target)
        return accuracy

    def r2_score(self, test_inputs, test_targets):
        """
        Returns r2 of net for given test data set.
        """
        from project2_tools import from_onehot

        _, outputs = self.feed_forward_vectorized(test_inputs)

        test_mean = np.mean(test_targets)
        r2 = 1 - np.sum((test_targets - outputs[-1])**2)/np.sum((test_targets - test_mean)**2)
        return r2
    

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

    def softmax(self, X, axis = -1):
        from project2_tools import softmax
        return softmax(X, axis = -1)

    def d_identity(self,x):
        return 1

    def d_sigmoid(self, x):
        return self.sigmoid(x)*(1.0 - self.sigmoid(x))

    def d_step(self, x):
        return 0

    def d_softsign(self, x):
        return self.softsign(x)**2

    def d_tanh(self, x):
        return 1 - self.tanh(x)**2

    def d_relu(self, x):
        return self._alpha * (x > 0)

    def d_elu(x):
        return np.choose(x > 0, [1, self._alpha * np.exp(x)])

    def d_softmax(self, X, axis = -1):
        """assuming df(x_i)/dx_j with i == j"""
        from project2_tools import softmax
        return softmax(X, axis = -1) * (1 - softmax(X, axis = -1))

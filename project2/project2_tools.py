import numpy as np
from numba import jit, njit
import pickle

def read_t(t,root="./data/IsingData/", dtype = np.int8):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(np.int8).reshape(-1,1600)

def get_available_t(root="./data/IsingData/"):
    import glob
    files = glob.glob(root + '*')
    temperatures = [float(f[f.find('=')+1:][:4]) for f in files]
    return np.sort(temperatures)

def ising_energies1D(spins, J = -1):
    """
    Calculates the energies of the states in the nn Ising Hamiltonian. 
    Parameters:
    -----------

    spins : np.array_like
        array of shape (n_sets,n_spins)
        
    J : float
        coupling energy of spins, default -1
    """

    E = np.sum(-1*spins*np.roll(spins,1, axis = 1), axis = 1)
    return E

def gen_1Ddata(n_spins, n_sets, ret_pairs = True, fit_intercept = True, **kwargs):
    """
    Generates data consisting of spin configurations their 1D ising
    energies. 

    Parameters: 
    -----------
    n_spins : int
        number of lattice points

    n_sets : int 
        number of data sets

    ret_pairs : bool
        if True, return outer product of spins.

    
    Returns:
    --------
    inputs : np.array
        spin configurations or pairs of spin configurations

    targets : np.array
        energies
    """

    spins = np.random.choice(np.array([-1,1],dtype=np.int8), 
                             size = (n_sets, n_spins))
    energies = ising_energies1D(spins, **kwargs)
    if ret_pairs:
        pairs = np.einsum('...i,...j->...ij', spins, spins).reshape(n_sets,-1)
        if fit_intercept:
            pairs = np.concatenate((np.ones((n_sets,1)),pairs),axis=1)
        return pairs, energies
    else:
        if fit_intercept:
            spins = np.concatenate((np.ones((n_sets,1)),spins),axis=1)
        return spins, energies

def to_onehot(classes):
    onehot = np.zeros((classes.size, classes.max()+1))
    onehot[np.arange(classes.size),classes] = 1
    return onehot

def from_onehot(onehot):
    return np.where(onehot)[-1]



def get_2Ddata(t_crit, onehot=False, n_classes = 2, t_range = 0.3):
    """
    Loads 2D ising model data from Mehta et. al., and returns spins as
    input data and classes as output data. Classes may be represented as
    numbers or as onehot-vectors. Number of classes is either 2 or 3. If 3,
    a range given by t_range around t_crit is used to make the critical
    phase. 
    """

    temps = get_available_t()
    data = np.c_[[read_t(t, dtype=np.int8) for t in temps]]

    data[data == 0] = -1

    Nt, n, Nl = data.shape
    classes = np.zeros((Nt, n),dtype = np.int8)

    if n_classes == 2:
        classes[temps > t_crit] = 1
    elif n_classes == 3:
        classes = np.ones((Nt, n),dtype = np.int8)

        ordered = temps > (t_crit + t_range )
        disordered = temps < (t_crit - t_range )
        classes[temps > (t_crit + t_range)] = 2
        classes[temps < (t_crit - t_range)] = 0
    else:
        raise ValueError('n_classes must be one of 2 or 3')

    classes = classes.ravel()
    
    data = data.reshape(-1, Nl)
    if onehot:
        return data, to_onehot(classes)
    else:   
        return data, classes




def MSE_net(test_inputs, test_targets, net):
    """ 
    Returns MSE of net for the given inputs and targets. 
    """
    # cost = []
    # for x,y in zip(test_inputs, test_targets):
    #     zs, outputs = net.feed_forward(x)
    #     cost.append(np.mean((outputs[-1] - y)**2))
    pred = [net.feed_forward(x)[-1][-1] for x in test_inputs]
    mse = np.mean((y - pred)**2)
    return mse

def batches(inputs, targets, n_batches = 10, shuffle = True, zipped = True):
    """
    Creates a generator which yields batches of input and targets.

    Parameters:
    -----------
    inputs, targets : np.array_like, with same first dimension.

    n_batches : int
        number of batches

    shuffle : bool
        if True, shuffle input and targets with same permutation before
        splitting in batches.

    Yields:
    -------

    input_batch : np.array_like
        batch of inputs

    target_batch : np.array_like
        batch of targets
    """
    if shuffle:
        mask = np.arange(len(inputs), dtype = np.int64)
        np.random.shuffle(mask)
        inputs = inputs[mask]
        targets = targets[mask]
    
    if len(inputs) != len(targets):
        raise ValueError('length of inputs and targets must be equal')
    indx = np.linspace(0, len(inputs), n_batches + 1, dtype=int)
    for i in range(n_batches):
        x = inputs[indx[i]:indx[i+1]]
        y = targets[indx[i]:indx[i+1]]
        if zipped:
            batch = [(x,y) for x, y in zip(inputs[indx[i]:indx[i+1]], targets[indx[i]:indx[i+1]])] 
            yield [(_x,_y) for _x,_y in zip(x,y)]
        else:
            yield x, y




def grid_search(hidden_sizes):
    """
    Result production function. Sets up a neural network with one hidden
    layer and performs a grid search over the size of the hidden layer.
    """
    from neuralnetwork import NeuralNetwork

    for h_size in hidden_sizes:
        layer_sizes = [1600, h_size, 1]
        net = NeuralNetwork(layer_sizes, act_func = ['sigmoid', 'identity'])


def stoch_grad_descent(net, inputs, targets, test_data_size = 0, epochs = 100,
        n_batches = 100, eta = 1e-3, verbose = True, 
        statistics_function = MSE_net): 
    """Stochastic gradient descent. Training data is shuffled and split
    into minibatches, which are used to train the network. If
    test_data_size != 0, the data is split into training and test and the
    function given by statistic is run on test data (input and target)
    after each batch. 

    Parameters:
    -----------

    inputs: np.array_like
        input data to the net

    targets: np.array_like
        targets

    test_data_size : float ∈ [0, 1)
        how large part of the data to use for test. 

    epochs : int
        number of epochs

    n_batches : int
        number of batches

    eta : float
        training
        

    statistics_function : function
        should take (net, inputs, targets) as input

    """

    from project2_tools import batches, MSE_net

    if verbose: 
        print('epoch, MSE')

    mse = []
    for j in range(epochs):
        b = batches(input_train, target_train, n_batches = n_batches,
                shuffle = True)


        for k, batch in enumerate(b):
            net.update_batch(batch, eta)
        mse.append(mse_net(input_test, target_test))

        if verbose:
            print('{:5}  {:.2f}'.format(j+1, mse[-1]))#, 'o', markersize = 9)
    return mse


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X. Taken from [1].

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.

    References
    ----------
    [1] https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


@njit
def calc_sum(a,b,c):
    """temporary"""
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            c[:,i,j] = a[:,i] * b[:,j]

@njit
def calc_sum2(a,b,c):
    """temporary"""
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            for k in range(a.shape[0]):
                c[k,i,j] = a[k,i] * b[k,j]

@njit
def calc_sum3(a,b,c, weight):
    """temporary"""
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            for k in range(a.shape[0]):
                c[i,j] += a[k,i] * b[k,j] * weight


@njit
def add_outer_products(x,y,out,weight=1):
    """
    Calculates the outer product elementwise over the second axis in both x
    and y, and adds to out, weighted by weight. In einstein notation,
    out_ij += x_ki * y_kj * weight.
    """
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(x.shape[0]):
                out[i,j] += x[k,i] * y[k,j] * weight

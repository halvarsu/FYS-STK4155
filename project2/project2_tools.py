import numpy as np
import pickle

def read_t(t,root="./data/IsingData/"):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)

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

def gen_1Ddata(n_spins, n_sets, ret_pairs = True, **kwargs):
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
        return pairs, energies
    else:
        return spins, energies


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

def batches(inputs, targets, n_batches = 10, shuffle = True):
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
        batch = [(x,y) for x, y in zip(inputs[indx[i]:indx[i+1]], targets[indx[i]:indx[i+1]])] 
        yield batch

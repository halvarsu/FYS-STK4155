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

def ising_energies1D(states):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    L = states.shape[1]
    J = np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

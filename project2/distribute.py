

def distribute_list(values, pc_list_file):
    import socket
    hostname = socket.gethostname()
    this = hostname.split('.')[0]

    with open(pc_list_file) as infile:
        hosts = [l.strip('\n') for l in infile.readlines()]

    n_comp = len(hosts)
    n_values = len(values)
    indx = np.linspace(0,n_values, n_comp+1).astype(int)
    host_indx = hosts.index(this)

    if host_indx == -1:
        raise ValueError('current host %s not in pc_list_file'%this)
    else:
        return values[indx[host_indx]:indx[host_indx+1]]

def run_classifier():
    from result_functions import run_minibatch_sgd as sgd

    data_input, data_targets = get_2Ddata(t_crit = 2.3, onehot = True)
    temp = train_test_split(data_input, data_targets, test_size = 0.33)
    input_train, input_test, target_train, target_test = temp

    eta_all = np.logspace(-2,2,9)
    eta = distribute_list(eta_all, '/uio/hume/student-u10/halvarsu/pcer.txt')
    print(eta)
    import socket
    host = socket.gethostname().split('.')[0]
    print(host)

    sgd(input_train,target_train, input_test, target_test, eta_values = eta, 
            n_hidden_values = [8, 10, 16, 20, 32, 64, 96, 128],
            file_name_append = host,
            hidden_act_func = 'tanh', max_epochs=100, earlystopping = False)
    print(eta)

def run_regression():
    from sklearn.model_selection import train_test_split
    data_input, data_targets = proj2_tools.gen_1Ddata(n_spins = L, n_sets = N) 
    data_targets = np.atleast_2d(data_targets).T

    temp = train_test_split(data_input, data_targets, test_size = 0.33)
    input_train, input_test, target_train, target_test = temp

    from result_functions import run_minibatch_sgd as sgd

    params = { 'n_batches':200,  'max_epochs':200,
            'tol':1e-10, 'hidden_act_func':'relu',
            'output_act_func':'identity', 'net_type':'regression',
            'earlystopping':False}

    import socket
    host = socket.gethostname().split('.')[0]
    print(host)
    sgd(input_train,target_train, input_test, target_test, eta_values = [1e-2],  n_hidden_values = [[]],
       file_name_append = 'regr_sig' + host, save_df = True, **params)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    import pandas as pd
    from project2_tools import get_2Ddata, to_onehot, from_onehot
    from neuralnet import NeuralNet

    import sys
    try:
        if sys.argv[1] == '1D':
            classifier = True
        elif sys.argv[1] == '2D':
            classifier = False
        else:
            raise ValueError("argument must be '1D' or '2D'")
    except IndexError:
        classifier = True # default

    if classifier:
        run_classifier()
    else:
        run_regression()


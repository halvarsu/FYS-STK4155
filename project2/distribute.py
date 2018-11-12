

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


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    import pandas as pd
    from project2_tools import get_2Ddata, to_onehot, from_onehot
    from neuralnet import NeuralNet

    data_input, data_targets = get_2Ddata(t_crit = 2.3, onehot = True)
    temp = train_test_split(data_input, data_targets, test_size = 0.33)
    input_train, input_test, target_train, target_test = temp

    eta_all = np.logspace(-2,2,9)
    eta = distribute_list(eta_all, '/uio/hume/student-u10/halvarsu/pcer.txt')
    print(eta)
    from result_functions import run_minibatch_sgd_one_hidden as sgd
    import socket
    host = socket.gethostname().split('.')[0]
    print(host)

    sgd(input_train,target_train, input_test, target_test, eta_values = eta, 
            n_hidden_values = [8, 10, 16, 20, 32, 64, 96, 128],
            file_name_append = host,
            hidden_act_func = 'tanh', max_epochs=100, earlystopping = False)
    print(eta)

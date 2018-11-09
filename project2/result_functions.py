import pandas as pd
import os

def run_minibatch_sgd_one_hidden(input_train, target_train, input_test, target_test,
        eta_values = None,
        n_hidden_values = None,
        outputdir = 'output/',
        save_df = True,
        verbose = True, 
        **kwargs
        ):
    """Runs minibatch stochastic gradient descent
    
    Parameters
    ----------
    input_train, target_train, input_test, target_test,

    eta_values : list
        eta values to loop over. Default [0.0001, 0.001, 0.01, 0.1]

    n_hidden_values : list
        hidden layer sizes to loop over. Default [5, 10, 20, 50, 100]

    save_df : bool
        default True

    outputdir : string
        output directory, used in case save_df == True. Default 'output/'

    verbose : bool
        print progress, default True

    **kwargs 
        used to update parameters containing the following defaults:
        params = { 'n_batches':10000, 'min_epochs':40, 'max_epochs':100,
        'tol':1e-10, 'hidden_act_func':'sigmoid'}

    Returns
    -------

    df : pandas.DataFrame
        contains accuracy of test set per epoch, number of epochs and 
        eta, nbatches for each run.
    """

    if save_df and (not os.path.isdir(outputdir)):
        raise ValueError("'outputdir' must be a directory if 'save_df == True'")

    import time
    from IPython.display import clear_output
    from project2_tools import batches

    accuracies = []
    params = { 'n_batches':10000, 'min_epochs':40, 'max_epochs':100,
            'tol':1e-10, 'hidden_act_func':'sigmoid'}
    params.update(kwargs)
    tol = params.pop('tol')
    n_batches = params.pop('n_batches')
    min_epochs = params.pop('min_epochs')
    max_epochs = params.pop('max_epochs')
    hidden_act_func = params.pop('hidden_act_func')
    if params:
        raise TypeError("unrecognized kwarg(s): {}".format(
        ", ".join(params.keys())))

    if eta_values is None:
        eta_values = [0.0001, 0.001, 0.01, 0.1]
    if n_hidden_values is None:
        n_hidden_values = [5, 10, 20, 50, 100]

    for eta in eta_values:
        for n_hidden in n_hidden_values:
            
            layer_sizes = [input_train.shape[1], n_hidden, target_train.shape[1]]

            net = NeuralNet(layer_sizes, 
                            net_type = 'classifier'
                            act_func = [hidden_act_func, 'softmax'], )

            tot = max_epochs * n_batches
            accuracy = []
            current = len(accuracy) + 1

            if verbose:
                start = time.time()

            for j in range(max_epochs):
                b = batches(input_train, target_train, 
                        n_batches = n_batches, zipped = False)

                for k, (x,y) in enumerate(b):
                    if verbose:
                        if k % 1000 == 0:
                            clear_output(wait = True)
                            print('n hidden nodes: ', n_hidden)
                            print('eta: ', eta)
                            print('batch size: ', x.shape[0])
                            print('batch, epoch, accuracy')
                            print('{:5}/{}  {:5}/{}  {:.2f}'.format(
                                k, n_batches, j, max_epochs, 
                                accuracy[-1] if len(accuracy) else 0))
                            now = time.time() 
                            print('Time estimate: {:.0f} seconds left'.format(
                                (now - start)/current * (tot-current)))
                    net.update_batch_vectorized(x, y, eta)

                    current += 1

                acc = net.accuracy(input_test, target_test)
                accuracy.append(acc)

                if len(accuracy) > min_epochs:
                    # decreasing accuracy
                    cond1 = np.all(accuracy[-1] < np.array(accuracy[-5:-1]))  
                    # stabilized accuracy
                    cond2 = np.abs(accuracy[-1] - accuracy[-2])/accuracy[-1] < tol 
                    if cond1 or cond2:
                        print('earlystop at epoch number {}'.format(len(accuracy)))
                        break

            accuracies.append({'eta':eta,'nhidden':layer_sizes[1], 'accuracy':accuracy, 
                               'epochs':len(accuracy),'n_batches':n_batches})

    df = pd.DataFrame(accuracies)
    df['final_accuracy'] = df['accuracy'].str[-1]
    df['max_accuracy'] = df['accuracy'].apply(np.max)
    
    if save_df:
        import glob
        fname = outputdir + 'mb_sgd{}.pickle'
        n_files = len(glob.glob(fname.format('*')))
        df.to_pickle('mb_sgd{}.pickle'.format(n_files))

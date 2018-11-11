import os
import sys
import warnings
import numpy as np

sys.path.append('./')
import result_functions as rf


def test_sgd():
    # Test saving functionality
    x1 = 2*np.ones((10,2))
    x2 = 2*np.ones((10,2))
    y1 = 2*np.ones((10,2))
    y2 = 2*np.ones((10,2))
    expect_file = "test/mb_sgdtest_0.pickle"

    if os.path.exists(expect_file):
        os.remove(expect_file)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.run_minibatch_sgd_one_hidden(x1,x2,y1,y2, eta_values=[1], n_batches = 1,
                n_hidden_values = [10], outputdir = 'test/', verbose = False,
                file_name_append = 'test')

    assert os.path.exists(expect_file)
    os.remove(expect_file)
    

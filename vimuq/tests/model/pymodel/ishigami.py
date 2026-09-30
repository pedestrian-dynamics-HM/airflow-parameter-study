import SALib.test_functions.Ishigami
import numpy as np


def model_function(input: np.ndarray):
    """Connects Ishigami test function (from SALib) with vimuq

    Arguments
    ----------
    input   N x M numpy array, where N = number of samples, M = 3

    Returns
    ----------
    y       1*N numpy array

    """
    y = SALib.test_functions.Ishigami.evaluate(input)
    return y

import numpy as np


def n_F(eigvals, k):
    if k >= len(eigvals):
        print('Cannot compute Dirac distribution since argument is out of range+')

    else:
        return 1. / (1 + np.exp(eigvals[k]))

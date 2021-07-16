import numpy as np
from numpy.linalg import eig


def n_F(eigvals, k):
    nsites = int(len(eigvals) / 2)
    if k >= len(eigvals):
        print('Cannot compute Dirac distribution since argument is out of range+')

    else:
        return 1. / (1 + np.exp(eigvals[k] - eigvals[k + nsites]))

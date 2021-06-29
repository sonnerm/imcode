import numpy as np
from numpy import version
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)


# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
# arguments: correlation coefficients A and DIAGONALIZED dressed density matrix rho_t, ...
def correlator(A, rho_t_diag, branch1, majorana_type1, tau_1, branch2, majorana_type2, tau_2, nsites):
    result = 0
    for k in range(nsites):
        n_F = 1. / (1.+np.exp(rho_t_diag[k, k]))  # fermi-dirac distribution
        result += A[branch2, majorana_type2, tau_2, k+nsites]*A[branch1, majorana_type1, tau_1, k]*n_F - A[branch2, majorana_type2, tau_2, k]*A[branch1, majorana_type1, tau_1, k+nsites]*(1-n_F)

    return result

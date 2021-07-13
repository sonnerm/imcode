import numpy as np
#from numpy.core.einsumfunc import einsum
from numpy.linalg import matrix_power


def evolvers(evolution_matrix,  N_t, nsites, nbr_Floquet_layers, beta_tilde):

    # initialize diagonal matricx that encodes dressing
    D_beta = np.zeros((2 ,2, 2*nsites), dtype=np.complex_)#first dimension: branch, second dimension: sites 0 and 0+L

    D_beta[0,0, 0] = np.exp(beta_tilde)
    D_beta[0,1, nsites] = np.exp(-beta_tilde)
    D_beta[1,0, 0] = np.exp(-beta_tilde)
    D_beta[1,1, nsites] = np.exp(beta_tilde)

    # initialize evolvers
    # first dimension 2 is for forward/backward branch (0 is forward, 1 is backward), second dimension 2 is because we do not compute correlations at every site in the environment but only at site 1 -> two relevant lines (site 0 and 0+L in env.) are extracted (in principle one would have dimension 2*nsites here,too)
    T_tilde = np.zeros((2, nbr_Floquet_layers, 2, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)

    for branch in range (2):
        for tau_index in range(0, nbr_Floquet_layers):
            T_tilde[branch, tau_index] =  D_beta[branch] @ matrix_power(evolution_matrix[branch], tau_index + 1)  @ N_t

    #test if evolvers reproduce correct relations for Ising limit
    #print('compare_test', T_tilde[0,0] @ T_tilde[0,0].T.conj())
    return T_tilde

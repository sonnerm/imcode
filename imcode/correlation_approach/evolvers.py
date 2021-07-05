import numpy as np
from numpy.core.einsumfunc import einsum
from numpy.linalg import matrix_power


def evolvers(M, M_inverse,  N_t, eigenvalues_G_eff, nsites, nbr_Floquet_layers, beta_tilde):

    # initialize diagonal matrices that encode..
    # .. time evolution ..
    D_phi = np.zeros((2,2*nsites, 2*nsites), dtype=np.complex_)
    # .. and dressing
    D_beta = np.zeros((2 ,2, 2*nsites), dtype=np.complex_)#first dimension: branch, second dimension: sites 0 and 0+L
  
    # fill with values (individually for each branch)
    for branch in range (2):
        for i in range(0, nsites):
            D_phi[branch,i, i] = np.exp(-1j*eigenvalues_G_eff[branch,i])
            D_phi[branch,i+nsites, i + nsites] = np.exp(-1j*eigenvalues_G_eff[branch,i+nsites])

    
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
            T_tilde[branch, tau_index] = np.einsum('ij,jk,kl->il', D_beta[branch], M[branch],matrix_power(D_phi[branch], tau_index + 1))#, M_inverse[branch], N_t)  # forward branch
     
    return T_tilde

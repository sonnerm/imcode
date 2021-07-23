import numpy as np
from numpy.linalg import matrix_power


def evolvers(evolution_matrix,  N_t, nsites, nbr_Floquet_layers, beta_tilde):

    # initialize diagonal matricx that encodes dressing
    D_beta = np.zeros((2 ,2, 2*nsites), dtype=np.complex_)#arguments:branch,  sites 0 and 0+L, lattice_site (0-2L)

    D_beta[0,0, 0] = np.exp(-beta_tilde)
    D_beta[0,1, nsites] = np.exp(beta_tilde)
    D_beta[1,0, 0] = np.exp(beta_tilde)
    D_beta[1,1, nsites] = np.exp(-beta_tilde)

    # initialize evolvers
    # first dimension two is for no_dressing (=0) or dressing (=1), second dimension 2 is for no_hermitian_conj_of_N (=0) or hermitian_conjugate_of_n (=1), trhid dimension 2 is for forward/backward branch (0 is forward, 1 is backward), fourth dimension 2 is because we do not compute correlations at every site in the environment but only at site 1 -> two relevant lines (site 0 and 0+L in env.) are extracted (in principle one would have dimension 2*nsites here,too)
    T_tilde = np.zeros((2, 2, 2, nbr_Floquet_layers, 2, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)
    N_t_mod = np.identity((2 * nsites),dtype=np.complex_)
    N_t_mod[0:nsites,:] = N_t[nsites:2*nsites,:].conj()
    N_t_mod[nsites:2*nsites,:] = N_t[0:nsites,:].conj()

    #N_t_mod = np.bmat([[N_t[0:nsites,0:nsites], N_t[nsites:2*nsites,0:nsites].conj()],[N_t[nsites:2*nsites,0:nsites], N_t[0:nsites,0:nsites].conj()]])


    for branch in range (2):
        for tau_index in range(0, nbr_Floquet_layers):
            T_tilde[0, 0,branch, tau_index] =  (matrix_power(evolution_matrix[branch], tau_index + 1)  @ N_t)[(0,nsites),:]#without dressing, without N_hermitian_conj
            T_tilde[0, 1,branch, tau_index] =  (matrix_power(evolution_matrix[branch], tau_index + 1)  @ N_t_mod)[(0,nsites),:]#without dressing, with N_hermitian_conj
            T_tilde[1, 0,branch, tau_index] =  D_beta[branch] @ matrix_power(evolution_matrix[branch], tau_index + 1)  @ N_t#with dressing, without N_hermitian_conj
            T_tilde[1, 1,branch, tau_index] =  D_beta[branch] @ matrix_power(evolution_matrix[branch], tau_index + 1)  @ N_t_mod#with dressing, with N_hermitian_conj
           
    return T_tilde

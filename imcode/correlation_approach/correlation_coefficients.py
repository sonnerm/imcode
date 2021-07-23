import numpy as np


def correlation_coefficients(T_tilde, nsites, nbr_Floquet_layers):

    # initialize correlation coefficients
    # first dimension 2 is for no_dressing (=0) or dressing (=1), second dimension 2 is for no_hermitian_conj_of_N (=0) or hermitian_conj_of_N (=1), third dimension 2 is for forward/backward branch (0 is forward, 1 is backward), fourth dimension 2 is for "Majorana" type (0 is THETA (relative MINUS sign), 1 is ZETA (relative PLUS sign)), fifth axis is for evolution times, sixth axis is for site arguments: (A_ij)-> site argument i=0 always, j in range (0,2L)
    A = np.zeros((2, 2, 2, 2, nbr_Floquet_layers, 2*nsites), dtype=np.complex_)

    for tau in range(0, nbr_Floquet_layers):
        # buffer evolver values that are combined to give correlation coefficients A
   
        # fill array of correlation coefficients
        # A-arguments: no_dressing (=0) or dressing (=1), no_hermitian_conj_of_N (=0) or hermitian_conjugate_of_n (=1), branch (0 = forward, 1 = backward), Majorana type, time, lattice index
        # T-arguments: no_dressing (=0) or dressing (=1), no_hermitian_conj_of_N (=0) or hermitian_conjugate_of_n (=1), branch (0 = forward, 1 = backward), time,  c and c^dagger at site 0 of env. (corresponding to indices 0 and 1, resp.), lattice site k 

        # "Theta/- Majorana" (relative minus sign):
        A[:, :, :, 0, tau,:] = T_tilde[:, :, :, tau, 0, :] - T_tilde[:, :, :, tau, 1, :] #explicitly given arguments: evolution time, c at site j=0 in env. || arguments: evolution time, c^dagger at site j=0 in env.
        # "Zeta/+ Majorana" (relative plus sign):
        A[:, :, :, 1, tau,:] = T_tilde[:, :, :, tau, 0, :] + T_tilde[:, :, :, tau, 1, :]


    return A

    

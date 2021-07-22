import numpy as np


def correlation_coefficients(T_tilde, nsites, nbr_Floquet_layers):

    # initialize correlation coefficients
    # first dimension 2 is for no_dressing (=0) or dressing (=1), second dimension 2 is for no_hermitian_conj_of_N (=0) or hermitian_conj_of_N (=1), third dimension 2 is for forward/backward branch (0 is forward, 1 is backward), fourth dimension 2 is for "Majorana" type (0 is THETA (relative MINUS sign), 1 is ZETA (relative PLUS sign)), fifth axis is for evolution times, sixth axis is for site arguments: (A_ij)-> site argument i=0 always, j in range (0,2L)
    A = np.zeros((2, 2, 2, 2, nbr_Floquet_layers, 2*nsites), dtype=np.complex_)

    # fill with values (individually for each branch)
    for tau in range(0, nbr_Floquet_layers):
        # buffer evolver values that are combined to give correlation coefficients A
   
        # fill array of correlation coefficients
        # A-arguments:  dressing, hermitian_conj_of_N, forward branch, "Theta/- Majorana", time, lattice index
        A[:, :, 0, 0, tau,:] = T_tilde[:, :, 0, tau, 0, :] - T_tilde[:, :, 0, tau, 1, :] #arguments:forward branch, evolution time, site j=0 in env. || arguments:forward branch, evolution time, site j=0+L in env.
    
        # A-arguments: dressing, hermitian_conj_of_N,forward branch, "Zeta/+ Majorana", time, lattice index
        A[:, :, 0, 1, tau,:] = T_tilde[:, :, 0, tau, 0, :] + T_tilde[:, :, 0, tau, 1, :]
       
        # A-arguments:dressing, hermitian_conj_of_N, backward branch, "Theta/- Majorana", time, lattice index
        A[:, :, 1, 0, tau,:] = T_tilde[:, :, 1, tau, 0, :] - T_tilde[:, :, 1, tau, 1, :]
      
        # A-arguments: dressing, hermitian_conj_of_N, backward branch, "Zeta/+ Majorana", time, lattice index
        A[:, :, 1, 1, tau,:] = T_tilde[:, :, 1, tau, 0, :] + T_tilde[:, :, 1, tau, 1, :]


    return A

    

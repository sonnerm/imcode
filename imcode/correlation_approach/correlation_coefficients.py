import numpy as np


def correlation_coefficients(T_tilde,T_tilde_mod, nsites, nbr_Floquet_layers):

    # initialize correlation coefficients
    # first dimension 2 is for no_dressing (=0) or dressing (=1), second dimension 2 is for no_hermitian_conj_of_N (=0) or hermitian_conj_of_N (=1), third dimension 2 is for forward/backward branch (0 is forward, 1 is backward), fourth dimension 2 is for "Majorana" type (0 is THETA (relative MINUS sign), 1 is ZETA (relative PLUS sign)), fifth axis is for evolution times, sixth axis is for site arguments: (A_ij)-> site argument i=0 always, j in range (0,2L)
    A = np.zeros((2, 2, 2, nbr_Floquet_layers, 2*nsites), dtype=np.complex_)
    print (A.shape)
    # fill with values (individually for each branch)
    for tau in range(0, nbr_Floquet_layers):
        # buffer evolver values that are combined to give correlation coefficients A
        # .. on forward branch
        # arguments:forward branch, evolution time, site j=0 in env.
        """
        T_tilde_fw_0 = T_tilde[0, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_fw_L = T_tilde[0, tau, 1]
        # .. on backward branch
        # arguments:forward branch, evolution time, site j=0 in env.
        T_tilde_bw_0 = T_tilde[1, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_bw_L = T_tilde[1, tau, 1]

        T_tilde_fw_0_mod = T_tilde_mod[0, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_fw_L_mod = T_tilde_mod[0, tau, 1]
        # .. on backward branch
        # arguments:forward branch, evolution time, site j=0 in env.
        T_tilde_bw_0_mod = T_tilde_mod[1, tau, 0]
        # arguments:forward branch, evolution time, site j=0+L in env.
        T_tilde_bw_L_mod = T_tilde_mod[1, tau, 1]
        """

        # fill array of correlation coefficients
        # arguments:  forward branch, "Theta/- Majorana", time
        print ('array-shapes:', A[:,0, 0, tau,:].shape, T_tilde[:,0, tau, 0,:].shape)
        A[:,0, 0, tau,:] = T_tilde[:,0, tau, 0,:] - T_tilde[:,0, tau, 1,:] #arguments:forward branch, evolution time, site j=0 in env. || arguments:forward branch, evolution time, site j=0+L in env.
    
        # arguments: forward branch, "Zeta/+ Majorana"
        A[:,0, 1, tau,:] = T_tilde[:,0, tau, 0,:] + T_tilde[:,0, tau, 1,:]
       
        # arguments: backward branch, "Theta/- Majorana"
        A[:,1, 0, tau,:] = T_tilde[:,1, tau, 0,:] - T_tilde[:,1, tau, 1,:]
      
        # arguments: backward branch, "Zeta/+ Majorana"
        A[:,1, 1, tau,:] = T_tilde[:,1, tau, 0,:] + T_tilde[:,1, tau, 1,:]
        print ('array-shapes:', A.shape)

    return A

    

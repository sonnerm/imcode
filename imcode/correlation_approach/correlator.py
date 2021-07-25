import numpy as np

# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
# arguments: correlation coefficients A and DIAGONALIZED dressed density matrix rho_t, ...
def correlator(A, n_expect, branch1, majorana_type1, tau_1, branch2, majorana_type2, tau_2):
    nsites = int(len(n_expect) / 2)
    # bc the tau arguments run from 1, 2, ..., nbr_Floquet_layers (= ntimes + 1). To convert to array indices, subtract one
    tau_1_index = tau_1 - 1
    tau_2_index = tau_2 - 1
    result = 0

    A_1 = A[:, branch1, majorana_type1, tau_1_index,:]#2D array with (2 x 2 * nsites) entries
    A_2 = np.zeros(A_1.shape,dtype=np.complex_)
    A_2[:,0:nsites] =  A[:, branch2, majorana_type2, tau_2_index,nsites : 2*nsites]#2D array with (2 x 2 * nsites) entries
    A_2[:,nsites: 2 * nsites] =  A[:, branch2, majorana_type2, tau_2_index,0 : nsites]#2D array with (2 x 2 * nsites) entries
    print (A_1[0].shape,A_2[0].shape)
    print (A_1[1].shape,A_2[1].shape)
    if tau_2_index == tau_1_index and branch1 == branch2 and majorana_type1 != majorana_type2:#dressing only for second observable (first argument = 0 in second coefficient array)
        if branch1 == 0:
            result = np.einsum('k,k,k-> ',A_1[0] , n_expect, A_2[1])
        else:
            result = np.einsum('k,k,k-> ',A_1[1] , n_expect, A_2[0])
    else:#dressing in both cases (first argument = 1 in both coefficient arrays)
        result = np.einsum('k,k,k-> ',A_1[1] , n_expect, A_2[1])
    

    return result

import numpy as np

# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
# arguments: correlation coefficients A and DIAGONALIZED dressed density matrix rho_t, ...
def correlator(A, n_expect, branch1, majorana_type1, tau_1, branch2, majorana_type2, tau_2, nsites):
    # bc the tau arguments run from 1, 2, ..., nbr_Floquet_layers (= ntimes + 1). To convert to array indices, subtract one
    tau_1_index = tau_1 - 1
    tau_2_index = tau_2 - 1
    result = 0
   
    if tau_2_index == tau_1_index and branch1 == branch2 and majorana_type1 != majorana_type2:#dressing only for second observable (first argument = 0 in second coefficient array)
        if branch1 == 0:
            result = np.einsum('k,k,k-> ',A[0, 0,branch1, majorana_type1, tau_1_index] , n_expect, A[1, 1,branch2, majorana_type2, tau_2_index])
        else:
            result = np.einsum('k,k,k-> ',A[1, 0,branch1, majorana_type1, tau_1_index] , n_expect, A[0, 1,branch2, majorana_type2, tau_2_index])
    else:#dressing in both cases (first argument = 1 in both coefficient arrays)
        result = np.einsum('k,k,k-> ',A[1, 0,branch1, majorana_type1, tau_1_index] , n_expect, A[1, 1,branch2, majorana_type2, tau_2_index])
    

    return result

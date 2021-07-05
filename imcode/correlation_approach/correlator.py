import numpy as np

# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
# arguments: correlation coefficients A and DIAGONALIZED dressed density matrix rho_t, ...
def correlator(A, rho_t_diag, branch1, majorana_type1, tau_1, branch2, majorana_type2, tau_2, nsites):
    tau_1_index = tau_1 - 1 #bc the tau arguments run from 1, 2, ..., nbr_Floquet_layers (= ntimes + 1). To convert to array indices, subtract one
    tau_2_index = tau_2 - 1
    result = 0
    for k in range(nsites):
        n_F = 1. / (1.+np.exp(rho_t_diag[k, k]))  # fermi-dirac distribution
        result += A[branch2, majorana_type2, tau_2_index, k+nsites]*A[branch1, majorana_type1, tau_1_index, k]*n_F + A[branch2, majorana_type2, tau_2_index, k]*A[branch1, majorana_type1, tau_1_index, k+nsites]*(1-n_F)
        #result += (A[branch2, majorana_type2, tau_2_index, k+nsites]*A[branch1, majorana_type1, tau_1_index, k] + A[branch2, majorana_type2, tau_2_index, k]*A[branch1, majorana_type1, tau_1_index, k+nsites]) * 0.5#edit
    return result

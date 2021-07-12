import numpy as np
from fermi_distr import n_F
# seed random number generator


# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
# arguments: correlation coefficients A and DIAGONALIZED dressed density matrix rho_t, ...
def correlator(A, rho_eigvals, branch1, majorana_type1, tau_1, branch2, majorana_type2, tau_2, nsites):
    # bc the tau arguments run from 1, 2, ..., nbr_Floquet_layers (= ntimes + 1). To convert to array indices, subtract one
    tau_1_index = tau_1 - 1
    tau_2_index = tau_2 - 1
    result = 0
    for k in range(nsites):

        result += A[branch2, majorana_type2, tau_2_index, k+nsites]*A[branch1, majorana_type1, tau_1_index, k]*n_F(
            rho_eigvals, k) + A[branch2, majorana_type2, tau_2_index, k]*A[branch1, majorana_type1, tau_1_index, k+nsites]*(1 - n_F(rho_eigvals, k + nsites))

        # infinite temperature limit in Ising case (in xy case, even the infinite temperature density matrix is nontrivial when dressed)
        # result += (A[branch2, majorana_type2, tau_2_index, k+nsites]*A[branch1, majorana_type1, tau_1_index, k] + A[branch2, majorana_type2, tau_2_index, k]*A[branch1, majorana_type1, tau_1_index, k+nsites]) * 0.5 #edit

    # the following simplification is only possible in the Ising limit:
    #result = - 0.5 * np.dot(A[branch1, majorana_type1, tau_1_index] ,A[branch2, majorana_type2, tau_2_index].T.conj())

    return result

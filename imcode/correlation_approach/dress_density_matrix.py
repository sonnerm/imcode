import numpy as np
from numpy.linalg import eig, matrix_power
from scipy.linalg import expm
from scipy import linalg
from reorder_eigenvecs import reorder_eigenvecs
from add_cmplx_random_antisym import add_cmplx_random_antisym
from scipy.linalg import sqrtm


def dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):
    nsites = int (len(rho_0_exponent[0]) / 2)
   
    rho_0 = expm(- rho_0_exponent)
    
    print('nbr.', nbr_Floquet_layers)
    # the following line assumes that rho_0_exponent is given in real space basis..
    rho_dressed = sqrtm(matrix_power(F_E_prime, nbr_Floquet_layers)  @ rho_0 @ rho_0  @ matrix_power(F_E_prime_dagger, nbr_Floquet_layers))

    eigenvals_dressed, eigenvecs_dressed = linalg.eigh(rho_dressed)

    cum = 0
    for i in range(nsites):
        cum += abs(eigenvals_dressed[i] - eigenvals_dressed[i + nsites])

    if cum > 1e-6:
    #By knowledge of the structure of N, i.e. (N = [[A, B^*],[B, A^*]]), we can construct the right part of the matrix from the left part of the matrix "eigenvecs_dressed", such that all phase factors and the order of eigenvectors are as desired.
        N_t = np.bmat([[eigenvecs_dressed[0:nsites,0:nsites], eigenvecs_dressed[nsites:2*nsites,0:nsites].conj()],[eigenvecs_dressed[nsites:2*nsites,0:nsites], eigenvecs_dressed[0:nsites,0:nsites].conj()]])
    else: 
        N_t = np.identity(eigenvecs_dressed.shape[0])


    #print ('N_t\n',N_t)
    #diag_check = N_t.T.conj() @ rho_dressed_exponent @ N_t
    #eigenvals_dressed = -np.diag(diag_check)
    #print('Dressed density matrix diagonalized', diag_check)
    #print('Eigenvalues of exponent:', eigenvals_dressed)

    np.set_printoptions(linewidth=np.nan, precision=6, suppress=True)
    n_expect = np.zeros((2 * nsites))#fermi-Dirac distribution for modes (basis in which dressed density matrix is diagonal). 
    #Note that the true expectation value has an additional factor Z_over_Z0. This factor, however, does not enter the exponent of the IM and is therefore not included here.
    for k in range (nsites):
        eval = eigenvals_dressed[k]
        #n_expect[k] = np.exp(+eval)  / (2 * np.cosh(eval) ) # for < c^dagger c >
        n_expect[k] = eval  / (eval + 1/eval ) # for < c^dagger c >
        n_expect[k + nsites] = 1 - n_expect[k] # for < c c^dagger >
    #print('nexpext',n_expect)
    return n_expect, N_t

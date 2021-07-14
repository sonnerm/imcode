#from numpy.core.einsumfunc import einsum
import numpy as np
from numpy.core.numeric import identity
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg
from reorder_eigenvecs import reorder_eigenvecs


def dress_density_matrix(rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):

    eigenvals = np.zeros(len(rho_0[0]))
    N_t = np.identity(len(rho_0[0]), dtype=np.complex_)

    # assume that rho_0 is given in real space basis ..this need to be generalized for more general density matrices
    rho_dressed = matrix_power(
        F_E_prime, nbr_Floquet_layers)  @ expm(- rho_0)  @ matrix_power(F_E_prime_dagger, nbr_Floquet_layers)

    # else, eigenvalues are zero and N_t is identity
    #if linalg.norm(rho_dressed - identity(len(rho_0[0]))) > 1e-10:
    #if (linalg.norm(F_E_prime)**nbr_Floquet_layers < 1e18):
    print ('F_Enorm',linalg.norm(F_E_prime) )
    # diagonalize dressed density matrix:
    rho_dressed_herm = 0.5*(rho_dressed + rho_dressed.T.conj())
    eigenvals, eigenvecs = linalg.eigh(rho_dressed_herm)

    argsort = np.argsort(- np.real(eigenvals))
    for i in range(int(len(rho_0[0])/2)):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        N_t[:, i] = eigenvecs[:, argsort[i]]
        N_t[:, len(rho_0[0]) - 1 - i] = eigenvecs[:, argsort[i + int(len(rho_0[0])/2)]]

    N_t = reorder_eigenvecs(N_t, int(len(rho_0[0])/2))

    diag = N_t.T.conj() @ rho_dressed_herm @ N_t
    eigenvals = diag.diagonal()

    
    print('dressed_matrix_test')
    print(rho_dressed_herm)
    print('N_t')
    print(N_t)
    print('rho_diagonalized')
    print( diag)
    print('eigenvalues')
    print(eigenvals)
    return eigenvals, N_t  # exponent of dressed density matrix, exponent of diagonalized dressed density matrix, matrix that diagonalizes dressed density matrix rho_dresse

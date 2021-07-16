#from numpy.core.einsumfunc import einsum
from types import DynamicClassAttribute
import numpy as np
from numpy.core.numeric import identity
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg
from reorder_eigenvecs import reorder_eigenvecs


def dress_density_matrix(rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):
    nsites = int (len(rho_0[0]) / 2)
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
    rho_dressed_herm = 0.5*(rho_dressed + rho_dressed.T.conj())#this stabilizes search for eigenvectors
    rho_0_herm = 0.5*(rho_0 + rho_0.T.conj())#this stabilizes search for eigenvectors
    eigenvals, eigenvecs = linalg.eigh(-linalg.logm(rho_dressed_herm))
    eigenvals_0, eigenvecs_0 = linalg.eigh(rho_0_herm)

    N_t = eigenvecs
    


    Z_dressed = 1
    Z_0 = 1
    n_expect = np.zeros((2 * nsites),dtype=np.complex_)
    for k in range (nsites):
        Z_dressed *= (np.exp(-eigenvals[k + nsites]) + np.exp(-eigenvals[k])) 
        Z_0 *= (np.exp(-eigenvals_0[k + nsites]) + np.exp(-eigenvals_0[k])) 

    for k in range (nsites):
        n_expect[k] = Z_dressed / Z_0 * np.exp(-eigenvals[k]) /  (np.exp(-eigenvals[k + nsites]) + np.exp(-eigenvals[k])) 
        n_expect[k + nsites] = Z_dressed / Z_0 * np.exp(-eigenvals[k + nsites]) /  (np.exp(-eigenvals[k + nsites]) + np.exp(-eigenvals[k])) 
    
    print('dressed_matrix_test')
    print(rho_dressed)
    print(rho_dressed_herm)
    print('N_t')
    print(N_t)
    print('eigenvalues')
    print(eigenvals)
    
    return n_expect, N_t  # eigenvalues of exponent of dressed DM, matrix that diagonalizes dressed density matrix rho_dresse

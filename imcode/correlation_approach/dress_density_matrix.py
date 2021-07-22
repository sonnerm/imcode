#from numpy.core.einsumfunc import einsum
from types import DynamicClassAttribute
import numpy as np
from numpy.core.numeric import identity
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg
from reorder_eigenvecs import reorder_eigenvecs


def dress_density_matrix(rho_0_exponent, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):
    nsites = int (len(rho_0_exponent[0]) / 2)
   
    # the following line assumes that rho_0_exponent is given in real space basis..
    rho_dressed = matrix_power(F_E_prime, nbr_Floquet_layers)  @ expm(- rho_0_exponent)  @ matrix_power(F_E_prime_dagger, nbr_Floquet_layers)

    # diagonalize dressed density matrix:
    rho_dressed_exponent = -linalg.logm(rho_dressed)#find exponent of density operator
    rho_dressed_exponent_herm = 0.5*(rho_dressed_exponent + rho_dressed_exponent.T.conj())#this stabilizes search for eigenvectors
    rho_0_exponent_herm = 0.5*(rho_0_exponent + rho_0_exponent.T.conj())#this stabilizes search for eigenvectors
    eigenvals_dressed, eigenvecs_dressed = linalg.eigh(rho_dressed_exponent_herm)
    eigenvals_0, eigenvecs_0 = linalg.eigh(rho_0_exponent_herm)

    N_t = eigenvecs_dressed#this is the matrix that diagonalizes the dressed density matrix is N_t.T.conj() @ rho_dressed @ N_t
    #check:
    diag_check = N_t.T.conj() @ rho_dressed @ N_t
    print('Dressed density matrix diagonalized', diag_check )
    print('Eigenvalues of exponent:', -np.log(np.diag(diag_check)))

    Z_dressed_over_Z_0 = 1#define ratio of partition functions (dressed rho vs. bare rho) -> will be computed in the lines below: Product of partition sums for every mode k  
    for k in range (nsites):
        Z_dressed_over_Z_0 *= (np.exp(-eigenvals_dressed[k + nsites]) + np.exp(-eigenvals_dressed[k])) / (np.exp(-eigenvals_0[k + nsites]) + np.exp(-eigenvals_0[k])) 
       
    n_expect = np.zeros((2 * nsites),dtype=np.complex_)#fermi-Dirac distribution for modes (basis in which dressed density matrix is diagonal)
    for k in range (nsites):
        n_expect[k] =  np.exp(-eigenvals_dressed[k]) /  (np.exp(-eigenvals_dressed[k + nsites]) + np.exp(-eigenvals_dressed[k])) 
        n_expect[k + nsites] = np.exp(-eigenvals_dressed[k + nsites]) /  (np.exp(-eigenvals_dressed[k + nsites]) + np.exp(-eigenvals_dressed[k])) 
    
    return n_expect, N_t , Z_dressed_over_Z_0

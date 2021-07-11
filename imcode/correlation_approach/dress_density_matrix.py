#from numpy.core.einsumfunc import einsum
import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg


def dress_density_matrix(rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):

    #assume that rho_0 is given in real space basis ..this need to be generalized for more general density matrices
    rho_dressed = matrix_power(F_E_prime,nbr_Floquet_layers)  @ expm(- rho_0)  @ matrix_power(F_E_prime_dagger,nbr_Floquet_layers)

    #diagonalize dressed density matrix:
    exp_eigvals, N_t = linalg.eig(rho_dressed)
    eigvals = -np.log(exp_eigvals)#eigenvalues of exponent of gaussian density matrix
    
    return  eigvals, N_t #exponent of dressed density matrix, exponent of diagonalized dressed density matrix, matrix that diagonalizes dressed density matrix rho_dresse

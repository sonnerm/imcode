#from numpy.core.einsumfunc import einsum
import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg


def dress_density_matrix(rho_0, F_E_prime, F_E_prime_dagger, nbr_Floquet_layers):
    #assume that rho_0 is given in real space basis ..this need to be generalized for more general density matrices
    rho_dressed = - linalg.logm( matrix_power(F_E_prime,nbr_Floquet_layers)  @ expm(- rho_0)  @ matrix_power(F_E_prime_dagger,nbr_Floquet_layers)) 

    return rho_dressed #exponent of dressed density matrix

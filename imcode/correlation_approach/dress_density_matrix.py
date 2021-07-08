#from numpy.core.einsumfunc import einsum
import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy import linalg


def dress_density_matrix(rho_0, G_eff, nbr_Floquet_layers):
    #assume that rho_0 is given in real space basis ..this need to be generalized for more general density matrices
    rho_dressed = - linalg.logm(np.einsum('ij,jk,kl->il', matrix_power(expm(0.5j* G_eff[0]),nbr_Floquet_layers) ,expm(- rho_0) ,matrix_power(expm(-0.5j* G_eff[1]),nbr_Floquet_layers)) )

    return rho_dressed #exponent of dressed density matrix

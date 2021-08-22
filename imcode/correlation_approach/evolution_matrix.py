import numpy as np
from scipy.linalg import expm
from scipy import linalg

def evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1):
    F_E_prime = expm(1j*G_g) @ expm(1j * G_XY_even) @ expm(1j * G_XY_odd) @ expm(1 * G_1)
    F_E_prime_dagger = F_E_prime.T.conj()
    F_E_prime_inverse = linalg.inv(F_E_prime)

    # compute evolution matrix:
    evolution_matrix = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)

    evolution_matrix[0] = F_E_prime_inverse # forward branch
    evolution_matrix[1] =  F_E_prime_dagger # backward branch

    return evolution_matrix, F_E_prime, F_E_prime_dagger

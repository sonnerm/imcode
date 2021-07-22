import numpy as np
from scipy.linalg import expm
from scipy import linalg

def evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx=0, Jy=0, g=0):

    # compute evolution matrix:
    evolution_matrix = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    evolution_matrix[0] = linalg.inv(expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd) @ expm( G_1))   # forward branch
    evolution_matrix[1] = (expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd) @ expm( G_1)).T.conj() # backward branch

    F_E_prime = expm(0.5j*G_g) @ expm(0.5j * G_XY_even) @ expm(0.5j * G_XY_odd) @ expm(0.5 * G_1)
    F_E_prime_dagger = (expm(0.5j*G_g) @ expm(0.5j * G_XY_even) @ expm(0.5j * G_XY_odd) @ expm(0.5 * G_1)).T.conj()

    return evolution_matrix, F_E_prime, F_E_prime_dagger

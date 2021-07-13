import numpy as np
from scipy.linalg import expm
from scipy import linalg
from numpy.linalg import matrix_power
import math


def evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1, Jx=0, Jy=0, g=0):

    # non-unitary gate stemming from vacuum projections (note that there is no imaginary j in from of G_1)
    # non-unitary local gate in xy-model that causes eigenvalues to be complex. Contributes only for non-Ising couplings.
    U_1 = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    U_1_inv = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)

    #alpha_0_square = (np.cos(2 * Jx) + np.cos(2 * Jy)) / 2.
    if (linalg.norm(G_1) < 1e18):
        U_1 = expm(0.5 * G_1)
        U_1_inv = expm(- 0.5 * G_1)
    # else, the entire influence matrix is zero due to the prefactor

    # compute evolution matrix:
    evolution_matrix = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)
    evolution_matrix[0] = expm(-1j*G_g) @ expm(-1.j * G_XY_even) @ expm(-1.j *
                                                                        G_XY_odd) @ matrix_power(U_1_inv, 2)  # forward branch
    # is equivalent to np.inverse(expm(-G_1) @ expm(1.j * G_XY_odd) @ expm(1.j * G_XY_even) @ expm(1j*G_g))#backward branch
    evolution_matrix[1] = expm(-1j*G_g) @ expm(-1.j *
                                               G_XY_even) @ expm(-1.j * G_XY_odd) @ matrix_power(U_1, 2)
    F_E_prime = expm(0.5j*G_g) @ expm(0.5j *
                                      G_XY_even) @ expm(0.5j * G_XY_odd) @ U_1
    F_E_prime_dagger = U_1 @ expm(-0.5j *
                                              G_XY_odd) @ expm(-0.5j * G_XY_even) @ expm(-0.5j*G_g)
    return evolution_matrix, F_E_prime, F_E_prime_dagger

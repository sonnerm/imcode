import numpy as np
from scipy.linalg import expm
from scipy import linalg

def evolution_matrix(nsites, G_XY_even, G_XY_odd, G_g, G_1):

    F_1 = expm(1. * G_1)
    F_E =  expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd)
  
    F_E_prime = F_E @ F_1
    F_E_prime_dagger =  F_E_prime.T.conj() 
   
    # compute evolution matrix:
    evol = expm(1.j*G_g) @ expm(1.j * G_XY_even) @ expm(1.j * G_XY_odd) 
  
    evolution_matrix = np.zeros((2, 2 * nsites, 2 * nsites), dtype=np.complex_)

    evolution_matrix[0] = expm(-G_1) @ evol.T.conj()  # forward branch
    evolution_matrix[1] = expm(G_1) @ evol.T.conj()   # backward branch
    np.set_printoptions(linewidth=np.nan, precision=7, suppress=True)
    #print('F_eff')
    #print(F_E_prime)
    #print(evolution_matrix[0])
    return evolution_matrix, F_E_prime, F_E_prime_dagger

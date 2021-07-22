import numpy as np
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
    random_part = np.random.rand(2 * nsites, 2 * nsites) * 1e-12
    # symmetrize random part
    for i in range(2*nsites):
        for j in range(i, 2*nsites):
            random_part[i, j] = random_part[j, i]

    rho_dressed_exponent += random_part#stabilize numerical diagonalization by adding random part
    rho_dressed_exponent_herm = 0.5 * (rho_dressed_exponent + rho_dressed_exponent.T.conj())#this stabilizes search for eigenvectors
    eigenvals_dressed, eigenvecs_dressed = linalg.eigh(rho_dressed_exponent_herm)

    N_t = eigenvecs_dressed#this is the matrix that diagonalizes the dressed density matrix is N_t.T.conj() @ rho_dressed @ N_t
    print ('N_t\n',N_t)
    for i in range (int(nsites / 2)):
        N_t[:,[nsites + i, 2 * nsites - 1 - i]] = N_t[:,[2 * nsites - 1 - i, nsites + i]]#switch columns such that eigenvalues with different signs are nsites entries apart.
    
    #check:
    diag_check = N_t.T.conj() @ rho_dressed @ N_t
    eigenvals_dressed = -np.diag(N_t.T.conj() @ rho_dressed_exponent @ N_t)
    print('Dressed density matrix diagonalized', diag_check )
    print('Eigenvalues of exponent:', eigenvals_dressed)


    n_expect = np.zeros((2 * nsites))#fermi-Dirac distribution for modes (basis in which dressed density matrix is diagonal). 
    #Note that the true expectation value has an additional factor Z_over_Z0. This factor, however, does not enter the exponent of the IM and is therefore not included here.
    for k in range (nsites):
        n_expect[k] = np.exp(-eigenvals_dressed[k]) / (np.exp(-eigenvals_dressed[k]) + np.exp(-eigenvals_dressed[k + nsites]))# for < c^dagger c >
        n_expect[k + nsites] = 1 - n_expect[k] # for < c c^dagger >
    
    return n_expect, N_t 

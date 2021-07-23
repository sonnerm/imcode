from ising_gamma import ising_gamma
import numpy as np
from numpy import version
from evolvers import evolvers
from correlation_coefficients import correlation_coefficients
from correlator import correlator
from tests import test_identity_correlations


# nrb_Floquet_layer = total_time + 1 (total time= 0 corresponds to one Floquet layer)
def IM_exponent(evolution_matrix, N_t, nsites, nbr_Floquet_layers, Jx, Jy, beta_tilde, n_expect):

    # define parameters:
    alpha = np.sqrt(2 * pow(np.cos(Jx)*np.cos(Jy),2) / (np.cos(2*Jx) + np.cos(2*Jy)))

    # procompute evolvers T from which the correlation coefficients A can be inferred
    T_tilde = evolvers(evolution_matrix, N_t, nsites, nbr_Floquet_layers, beta_tilde)
   
    # precompute correlation coefficients A from which we construct the correlation functions, i.e. an array containing all correlation coefficients (computed from evolvers T_tilde)
    A = correlation_coefficients( T_tilde,  nsites, nbr_Floquet_layers)
  
    # define matrix B that is twice exponent of IM:
    B = np.zeros((4 * nbr_Floquet_layers, 4 * nbr_Floquet_layers), dtype=np.complex_)

    test_identity_correlations(A,n_expect, nbr_Floquet_layers)

    for tau_1 in range(nbr_Floquet_layers):
        for tau_2 in range(nbr_Floquet_layers):
            # nbr_Floquet_layers is the number of discrete times (0,1,2,..,t) i.e. nbr_Floquet_layers = t + 1. The variable time_1 takes analytical values t+1, t, ..., 1. This translates to (nbr_Floquet_layers, nbr_Floquet_layers-1, ..., 1). They are saved in an array of length nbr_Floquet_layers with indices nbr_Floquet_layers-1, nbr_Floquet_layers-2, ..., 0
            time_1 = nbr_Floquet_layers - tau_1
            time_2 = nbr_Floquet_layers - tau_2

            G_lesser_zetazeta = 1j * correlator(A, n_expect, 0, 1, time_1, 1, 1, time_2)
            G_greater_zetazeta = -1j * correlator(A, n_expect, 0, 1, time_2, 1, 1, time_1)

            G_lesser_zetatheta = correlator(A, n_expect, 0, 1, time_1, 1, 0, time_2)
            G_greater_zetatheta = - correlator(A, n_expect, 0, 0, time_2, 1, 1, time_1)
       
            G_lesser_thetazeta = correlator(A, n_expect, 0, 0, time_1, 1, 1, time_2)
            G_greater_thetazeta = - correlator(A, n_expect, 0, 1, time_2, 1, 0, time_1)

            G_lesser_thetatheta = -1j * correlator(A, n_expect, 0, 0, time_1, 1, 0, time_2)
            G_greater_thetatheta = 1j * correlator(A, n_expect, 0, 0, time_2, 1, 0, time_1)
           
            G_Feynman_zetazeta = 0
            G_AntiFeynman_zetazeta = 0
            G_Feynman_zetatheta = 0
            G_AntiFeynman_zetatheta = 0
            G_Feynman_thetazeta = 0
            G_AntiFeynman_thetazeta = 0
            G_Feynman_thetatheta = 0
            G_AntiFeynman_thetatheta = 0

            if tau_2 > tau_1:
                G_Feynman_zetazeta = 1j * correlator(A, n_expect, 0, 1, time_1, 0, 1, time_2)
                G_AntiFeynman_zetazeta = -1j * correlator(A, n_expect, 1, 1, time_2, 1, 1, time_1)

                G_Feynman_zetatheta = correlator(A, n_expect, 0, 1, time_1, 0, 0, time_2)
                G_AntiFeynman_zetatheta = - correlator(A, n_expect, 1, 0, time_2, 1, 1, time_1)

                G_Feynman_thetazeta = correlator(A, n_expect, 0, 0, time_1, 0, 1, time_2)
                G_AntiFeynman_thetazeta = - correlator(A, n_expect, 1, 1, time_2, 1, 0, time_1)

                G_Feynman_thetatheta = -1j * correlator(A, n_expect, 0, 0, time_1, 0, 0, time_2)
                G_AntiFeynman_thetatheta = 1j * correlator(A, n_expect, 1, 0, time_2, 1, 0, time_1)

            elif tau_2 < tau_1:  
                G_Feynman_zetazeta = -1j * correlator(A, n_expect, 0, 1, time_2, 0, 1, time_1)
                G_AntiFeynman_zetazeta = 1j * correlator(A, n_expect, 1, 1, time_1, 1, 1, time_2)

                G_Feynman_zetatheta = - correlator(A, n_expect, 0, 0, time_2, 0, 1, time_1)
                G_AntiFeynman_zetatheta = correlator(A, n_expect, 1, 1, time_1, 1, 0, time_2)

                G_Feynman_thetazeta = - correlator(A, n_expect, 0, 1, time_2, 0, 0, time_1)
                G_AntiFeynman_thetazeta = correlator(A, n_expect, 1, 0, time_1, 1, 1, time_2)

                G_Feynman_thetatheta = 1j * correlator(A, n_expect, 0, 0, time_2, 0, 0, time_1)
                G_AntiFeynman_thetatheta = -1j * correlator(A, n_expect, 1, 0, time_1, 1, 0, time_2)

            else:# for tau_1 = tau_2, G_Feynman and G_AntiFeynman are zero for same fields
                G_Feynman_zetatheta = - correlator(A, n_expect, 0, 0, time_2, 0, 1, time_1)
                G_AntiFeynman_zetatheta = correlator(A, n_expect, 1, 1, time_1, 1, 0, time_2)

                G_Feynman_thetazeta = - correlator(A, n_expect, 0, 1, time_2, 0, 0, time_1)
                G_AntiFeynman_thetazeta = correlator(A, n_expect, 1, 0, time_1, 1, 1, time_2)


            #define prefactors for correlators to make notation more compact
            prefac_x = alpha * np.tan(Jx) * np.sqrt(2)
            prefac_y = alpha * np.tan(Jy) * np.sqrt(2)

            # B is the matrix that contains all the correlations, excatly corresponding to the matrix [[G_thetatheta, G_thetazeta],[G_zetatheta, G_zetazeta]] in my notes, without additional prefactors
            B[4 * tau_1, 4 * tau_2] = -1j * G_Feynman_thetatheta * prefac_y**2 
            B[4 * tau_1, 4 * tau_2 + 1] = -1j * G_lesser_thetatheta * prefac_y**2 
            B[4 * tau_1 + 1, 4 * tau_2] = -1j * G_greater_thetatheta * prefac_y**2 
            B[4 * tau_1 + 1, 4 * tau_2 + 1] = -1j * G_AntiFeynman_thetatheta * prefac_y**2 
            
            B[4 * tau_1, 4 * tau_2 + 2] = - 1j * G_Feynman_thetazeta * prefac_x * prefac_y 
            B[4 * tau_1, 4 * tau_2 + 3] = 1j* G_lesser_thetazeta * prefac_x * prefac_y 
            B[4 * tau_1 + 1, 4 * tau_2 + 2] = - 1j * G_greater_thetazeta * prefac_x * prefac_y 
            B[4 * tau_1 + 1, 4 * tau_2 + 3] =1j* G_AntiFeynman_thetazeta * prefac_x * prefac_y 

            B[4 * tau_1 + 2, 4 * tau_2] = -1j * G_Feynman_zetatheta * prefac_x * prefac_y 
            B[4 * tau_1 + 2, 4 * tau_2 + 1] = - 1j * G_lesser_zetatheta * prefac_x * prefac_y 
            B[4 * tau_1 + 3, 4 * tau_2] = 1j*G_greater_zetatheta * prefac_x * prefac_y 
            B[4 * tau_1 + 3, 4 * tau_2 + 1] = 1j*G_AntiFeynman_zetatheta * prefac_x * prefac_y 
            
            B[4 * tau_1 + 2, 4 * tau_2 + 2] = 1j * G_Feynman_zetazeta * prefac_x**2 
            B[4 * tau_1 + 2, 4 * tau_2 + 3] = -1j * G_lesser_zetazeta * prefac_x**2 
            B[4 * tau_1 + 3, 4 * tau_2 + 2] = - 1j * G_greater_zetazeta * prefac_x**2 
            B[4 * tau_1 + 3, 4 * tau_2 + 3] = 1j * G_AntiFeynman_zetazeta * prefac_x**2 

            if tau_1 == tau_2:#equal time correlators between same operators on same branch have only one factor alpha -> cancel out one of the two.
                B[4 * tau_1, 4 * tau_2 + 2] *= 1. / alpha #G_Feynman_thetazeta
                B[4 * tau_1 + 1, 4 * tau_2 + 3] *= 1. / alpha #G_AntiFeynman_thetazeta
                B[4 * tau_1 + 2, 4 * tau_2] *= 1. / alpha #G_Feynman_zetatheta
                B[4 * tau_1 + 3, 4 * tau_2 + 1] *= 1. / alpha #G_AntiFeynman_zetatheta

    for tau in range(nbr_Floquet_layers):  # trivial factor. Note that there is no factor 0.5 here since it is absorbed into B which already has a prefactor 0.5 in the exponent
        B[4 * tau, 4 * tau + 2] +=  1
        B[4 * tau + 1, 4 * tau + 3] -=  1
        B[4 * tau + 2, 4 * tau] -=  1
        B[4 * tau + 3, 4 * tau + 1] += 1

    print('B\n')
    print(B)# this matrix is twice the exponent of the influence matrix.

    return B

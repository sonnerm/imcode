from ising_gamma import ising_gamma
import numpy as np
from numpy import version
from evolvers import evolvers
from correlation_coefficients import correlation_coefficients
from correlator import correlator


def IM_exponent( evolution_matrix, N_t, nsites, nbr_Floquet_layers, Jx, Jy, rho_t):#nrb_Floquet_layer = total_time + 1 (total time= 0 corresponds to one Floquet layer)
    f = 1 #need to prove that this is always tru
    #define parameters:
    #T_xy = 1 / (1 + f * np.tan(Jx) * np.tan(Jy))
    beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))
    alpha = np.sqrt(2 * (np.cos(Jx)*np.cos(Jy)) ** 2 / (np.cos(2*Jx) + np.cos(2*Jy)))


    # procompute evolvers T from which the correlation coefficients A can be inferred
    #T_tilde = evolvers(M_fw, M_fw_inverse, M_bw, M_bw_inverse, N_t, eigenvalues_G_eff_fw,eigenvalues_G_eff_bw, nsites, nbr_Floquet_layers, beta_tilde)  # array containing the evolvers
    T_tilde = evolvers(evolution_matrix, N_t,nsites, nbr_Floquet_layers, beta_tilde)
    print (T_tilde)
    # precompute correlation coefficients A from which we construct the correlation functions
    # array containing all correlation coefficients (computed from evolvers T_tilde)
    A = correlation_coefficients(T_tilde, nsites, nbr_Floquet_layers)
    # exponent of IM
    B = np.zeros((4 * nbr_Floquet_layers, 4 * nbr_Floquet_layers), dtype=np.complex_)

    for tau_1 in range(nbr_Floquet_layers):
        for tau_2 in range(nbr_Floquet_layers):
            # nbr_Floquet_layers is the number of discrete times (0,1,2,..,t) i.e. nbr_Floquet_layers = t + 1. The variable time_1 takes analytical values t+1, t, ..., 1. This translates to (nbr_Floquet_layers, nbr_Floquet_layers-1, ..., 1). They are saved in an array of length nbr_Floquet_layers with indices nbr_Floquet_layers-1, nbr_Floquet_layers-2, ..., 0
            time_1 = nbr_Floquet_layers - tau_1
            time_2 = nbr_Floquet_layers - tau_2

            G_lesser_zetazeta = 1j * \
                correlator(A, rho_t, 0, 1, time_1, 1, 1, time_2, nsites)
            G_greater_zetazeta = -1j * \
                correlator(A, rho_t, 0, 1, time_2, 1, 1, time_1, nsites)


            G_lesser_zetatheta = correlator(
                A, rho_t, 0, 1, time_1, 1, 0, time_2, nsites)
            G_greater_zetatheta = - \
                correlator(A, rho_t, 0, 0, time_2, 1, 1, time_1, nsites)

            G_lesser_thetazeta = correlator(
                A, rho_t, 0, 0, time_1, 1, 1, time_2, nsites)
            G_greater_thetazeta = - \
                correlator(A, rho_t, 0, 1, time_2, 1, 0, time_1, nsites)
  

            G_lesser_thetatheta = -1j * \
                correlator(A, rho_t, 0, 0, time_1, 1, 0, time_2, nsites)
            G_greater_thetatheta = 1j * \
                correlator(A, rho_t, 0, 0, time_2, 1, 0, time_1, nsites)
        

            G_Feynman_zetazeta = 0
            G_AntiFeynman_zetazeta = 0
            G_Feynman_zetatheta = 0
            G_AntiFeynman_zetatheta = 0
            G_Feynman_thetazeta = 0
            G_AntiFeynman_thetazeta = 0
            G_Feynman_thetatheta = 0
            G_AntiFeynman_thetatheta = 0

            if tau_2 > tau_1:
                G_Feynman_zetazeta = G_lesser_zetazeta
                G_AntiFeynman_zetazeta = G_greater_zetazeta

                G_Feynman_zetatheta = G_lesser_zetatheta
                G_AntiFeynman_zetatheta = G_greater_zetatheta

                G_Feynman_thetazeta = G_lesser_thetazeta
                G_AntiFeynman_thetazeta = G_greater_thetazeta

                G_Feynman_thetatheta = G_lesser_thetatheta
                G_AntiFeynman_thetatheta = G_greater_thetatheta

            elif tau_2 < tau_1:  # for i=j, G_Feynman and G_AntiFeynman are zero
                G_Feynman_zetazeta = G_greater_zetazeta
                G_AntiFeynman_zetazeta = G_lesser_zetazeta

                G_Feynman_zetatheta = G_greater_zetatheta
                G_AntiFeynman_zetatheta = G_lesser_zetatheta

                G_Feynman_thetazeta = G_greater_thetazeta
                G_AntiFeynman_thetazeta = G_lesser_thetazeta

                G_Feynman_thetatheta = G_greater_thetatheta
                G_AntiFeynman_thetatheta = G_lesser_thetatheta

            prefac_x = alpha * np.tan(Jx) 
            prefac_y = alpha * np.tan(Jy) 

            B[4 * tau_1, 4 * tau_2] = -1j * G_Feynman_thetatheta * prefac_y**2
            B[4 * tau_1, 4 * tau_2 + 1] = -1j * G_lesser_thetatheta * prefac_y**2
            #print -1j * G_lesser_thetatheta, alpha, np.tan(Jy)**2,  T_xy**2 #edit
            B[4 * tau_1 + 1, 4 * tau_2] = -1j * G_greater_thetatheta * prefac_y**2
            B[4 * tau_1 + 1, 4 * tau_2 + 1] = -1j * G_AntiFeynman_thetatheta * prefac_y**2

            B[4 * tau_1, 4 * tau_2 + 2] = -1 * G_Feynman_thetazeta * prefac_x * prefac_y
            B[4 * tau_1, 4 * tau_2 + 3] = G_lesser_thetazeta * prefac_x * prefac_y
            B[4 * tau_1 + 1, 4 * tau_2 + 2] = -1 * G_greater_thetazeta * prefac_x * prefac_y
            B[4 * tau_1 + 1, 4 * tau_2 + 3] = G_AntiFeynman_thetazeta * prefac_x * prefac_y

            B[4 * tau_1 + 2, 4 * tau_2] = -1 * G_Feynman_zetatheta * prefac_x * prefac_y
            B[4 * tau_1 + 2, 4 * tau_2 + 1] = -1 * G_lesser_zetatheta * prefac_x * prefac_y
            B[4 * tau_1 + 3, 4 * tau_2] = G_greater_zetatheta * prefac_x * prefac_y
            B[4 * tau_1 + 3, 4 * tau_2 + 1] = G_AntiFeynman_zetatheta * prefac_x * prefac_y

            B[4 * tau_1 + 2, 4 * tau_2 + 2] = 1j * G_Feynman_zetazeta * prefac_x**2
            B[4 * tau_1 + 2, 4 * tau_2 + 3] = -1j * G_lesser_zetazeta * prefac_x**2
            B[4 * tau_1 + 3, 4 * tau_2 + 2] = -1j * G_greater_zetazeta * prefac_x**2
            B[4 * tau_1 + 3, 4 * tau_2 + 3] = 1j * G_AntiFeynman_zetazeta * prefac_x**2
    for tau in range(nbr_Floquet_layers):  # trivial factor

        B[4 * tau, 4 * tau + 2] += 0.5
        B[4 * tau + 1, 4 * tau + 3] -= 0.5
        B[4 * tau + 2, 4 * tau] -= 0.5
        B[4 * tau + 3, 4 * tau + 1] += 0.5

    # factor 2 to fit Alessio's notes where we have 1/2 B in exponent of influence matrix
    B = np.dot(2., B)

    print ('B\n')
    print(B)
 
    return B

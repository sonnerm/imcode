import numpy as np
from numpy import version
from evolvers import evolvers
from correlation_coefficients import correlation_coefficients
from correlator import correlator


def IM_exponent(M_fw, M_fw_inverse, M_bw, M_bw_inverse, N_t, eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, ntimes, Jx, Jy, T_xy, rho_t):
    beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))
    alpha = np.sqrt(2 * (np.cos(Jx)*np.cos(Jy)*T_xy) **
                    2 / (np.cos(2*Jx) + np.cos(2*Jy)))
    # procompute evolvers T from which the correlation coefficients A can be inferred
    T_tilde = evolvers(M_fw, M_fw_inverse, M_bw, M_bw_inverse, N_t, eigenvalues_G_eff_fw,
                       eigenvalues_G_eff_bw, nsites, ntimes, beta_tilde)  # array containing the evolvers
    print T_tilde
    # precompute correlation coefficients A from which we construct the correlation functions
    # array containing all correlation coefficients (computed from evolvers T_tilde)
    A = correlation_coefficients(T_tilde, nsites, ntimes)
    # exponent of IM
    B = np.zeros((4 * ntimes, 4 * ntimes), dtype=np.complex_)

    for tau_1 in range(ntimes):
        for tau_2 in range(ntimes):
            # ntimes is the number of discrete times (0,1,2,..,t) i.e. ntimes = t + 1. The variable time_1 takes analytical values t+1, t, ..., 1. This translates to ntimes, ntimes-1, ..., 1. They are saved in an array of length ntimes with indices ntimes-1, ntimes-2, ..., 0
            time_1 = ntimes - 1 - tau_1
            time_2 = ntimes - 1 - tau_2
            G_lesser_zetazeta = 1j * \
                correlator(A, rho_t, 0, 1, time_1, 1, 1, time_2, nsites)
            G_greater_zetazeta = -1j * \
                correlator(A, rho_t, 0, 1, time_2, 1, 1, time_1, nsites)

            G_lesser_zetatheta = correlator(
                A, rho_t, 0, 1, time_1, 1, 0, time_2, nsites)
            G_greater_zetatheta = - \
                correlator(A, rho_t, 0, 0, tau_2, 1, 1, time_1, nsites)

            G_lesser_thetazeta = correlator(
                A, rho_t, 0, 0, time_1, 1, 1, time_2, nsites)
            G_greater_thetazeta = - \
                correlator(A, rho_t, 0, 0, time_2, 1, 0, time_1, nsites)

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

            B[4 * tau_1, 4 * tau_2] = -1j * G_Feynman_thetatheta * \
                alpha**2 * np.tan(Jy)**2 / T_xy**2
            B[4 * tau_1, 4 * tau_2 + 1] = -1j * G_lesser_thetatheta * \
                alpha**2 * np.tan(Jy)**2 / T_xy**2
            B[4 * tau_1 + 1, 4 * tau_2] = -1j * G_greater_thetatheta * \
                alpha**2 * np.tan(Jy)**2 / T_xy**2
            B[4 * tau_1 + 1, 4 * tau_2 + 1] = -1j * \
                G_AntiFeynman_thetatheta * alpha**2 * np.tan(Jy)**2 / T_xy**2

            B[4 * tau_1, 4 * tau_2 + 2] = -1 * G_Feynman_thetazeta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1, 4 * tau_2 + 3] = G_lesser_thetazeta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1 + 1, 4 * tau_2 + 2] = -1 * G_greater_thetazeta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1 + 1, 4 * tau_2 + 3] = G_AntiFeynman_thetazeta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2

            B[4 * tau_1 + 2, 4 * tau_2] = -1 * G_Feynman_zetatheta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1 + 2, 4 * tau_2 + 1] = -1 * G_lesser_zetatheta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1 + 3, 4 * tau_2] = G_greater_zetatheta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * tau_1 + 3, 4 * tau_2 + 1] = G_AntiFeynman_zetatheta * \
                alpha**2 * np.tan(Jy) * np.tan(Jx) / T_xy**2

            B[4 * tau_1 + 2, 4 * tau_2 + 2] = 1j * \
                G_Feynman_zetazeta * alpha**2 * np.tan(Jx)**2 / T_xy**2
            B[4 * tau_1 + 2, 4 * tau_2 + 3] = -1j * \
                G_lesser_zetazeta * alpha**2 * np.tan(Jx)**2 / T_xy**2
            B[4 * tau_1 + 3, 4 * tau_2 + 2] = -1j * \
                G_greater_zetazeta * alpha**2 * np.tan(Jx)**2 / T_xy**2
            B[4 * tau_1 + 3, 4 * tau_2 + 3] = 1j * \
                G_AntiFeynman_zetazeta * alpha**2 * np.tan(Jx)**2 / T_xy**2

    for i in range(ntimes):  # trivial factor

        B[4 * tau_1, 4 * tau_1 + 2] += 0.5
        B[4 * tau_1 + 1, 4 * tau_1 + 3] -= 0.5
        B[4 * tau_1 + 2, 4 * i] -= 0.5
        B[4 * tau_1 + 3, 4 * tau_1 + 1] += 0.5

    # factor 2 to fit Alessio's notes where we have 1/2 B in exponent of influence matrix
    B = np.dot(2., B)
 

    return B

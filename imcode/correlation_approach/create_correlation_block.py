import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)

def create_correlation_block(M, eigenvalues_G_eff, nsites, ntimes, Jx, Jy, g, beta, T_xy, f):
    print('Creating Greens function for time ', ntimes)

    # compute all correlators from which we can construct Keldysh Greens functions

    B = np.zeros((4 * ntimes, 4 * ntimes), dtype=np.complex_)
    # zetazeta
    for i in range(ntimes):
        for j in range(ntimes):

            G_lesser_zetazeta = 0
            G_greater_zetazeta = 0
            G_lesser_zetatheta = 0
            G_greater_zetatheta = 0
            G_lesser_thetazeta = 0
            G_greater_thetazeta = 0
            G_lesser_thetatheta = 0
            G_greater_thetatheta = 0

            # compute KIC_response function directly from analytical formula. Does not work for non-diagonal gates
            KIC_response = False
            if KIC_response:
                gamma = 0
                for k in range(nsites):
                    gamma += abs(M[0, k] - M[nsites, k])**2 * \
                        np.cos(eigenvalues_G_eff[k] * (j-i))

                G_lesser_zetazeta = 1j * gamma
                G_greater_zetazeta = - 1j * gamma

            else:

                downdown_ji = correlator(
                    M, eigenvalues_G_eff, 0, 0, 0, 0, j, i, beta)
                downup_ji = correlator(M, eigenvalues_G_eff,
                                       0, 1, 0, 0, j, i, beta)
                updown_ji = correlator(M, eigenvalues_G_eff,
                                       1, 0, 0, 0, j, i, beta)
                upup_ji = correlator(M, eigenvalues_G_eff,
                                     1, 1, 0, 0, j, i, beta)

                downdown_ij = correlator(
                    M, eigenvalues_G_eff, 0, 0, 0, 0, i, j, beta)
                downup_ij = correlator(M, eigenvalues_G_eff,
                                       0, 1, 0, 0, i, j, beta)
                updown_ij = correlator(M, eigenvalues_G_eff,
                                       1, 0, 0, 0, i, j, beta)
                upup_ij = correlator(M, eigenvalues_G_eff,
                                     1, 1, 0, 0, i, j, beta)

                G_lesser_zetazeta = 1j * \
                    (- downdown_ji - upup_ji + updown_ji + downup_ji)
                G_greater_zetazeta = - 1j * \
                    (- downdown_ij - upup_ij + updown_ij + downup_ij)

                G_lesser_zetatheta = - downdown_ji + upup_ji - updown_ji + downup_ji
                G_greater_zetatheta = downdown_ij - upup_ij - updown_ij + downup_ij

                G_lesser_thetazeta = - downdown_ji + upup_ji + updown_ji - downup_ji
                G_greater_thetazeta = downdown_ij - upup_ij + updown_ij - downup_ij

                G_lesser_thetatheta = 1j * \
                    (downdown_ji + upup_ji + updown_ji + downup_ji)
                G_greater_thetatheta = - 1j * \
                    (downdown_ij + upup_ij + updown_ij + downup_ij)

            G_Feynman_zetazeta = 0
            G_AntiFeynman_zetazeta = 0
            G_Feynman_zetatheta = 0
            G_AntiFeynman_zetatheta = 0
            G_Feynman_thetazeta = 0
            G_AntiFeynman_thetazeta = 0
            G_Feynman_thetatheta = 0
            G_AntiFeynman_thetatheta = 0

            if j > i:
                G_Feynman_zetazeta = G_lesser_zetazeta
                G_AntiFeynman_zetazeta = G_greater_zetazeta

                G_Feynman_zetatheta = G_lesser_zetatheta
                G_AntiFeynman_zetatheta = G_greater_zetatheta

                G_Feynman_thetazeta = G_lesser_thetazeta
                G_AntiFeynman_thetazeta = G_greater_thetazeta

                G_Feynman_thetatheta = G_lesser_thetatheta
                G_AntiFeynman_thetatheta = G_greater_thetatheta

            elif j < i:  # for i=j, G_Feynman and G_AntiFeynman are zero
                G_Feynman_zetazeta = G_greater_zetazeta
                G_AntiFeynman_zetazeta = G_lesser_zetazeta

                G_Feynman_zetatheta = G_greater_zetatheta
                G_AntiFeynman_zetatheta = G_lesser_zetatheta

                G_Feynman_thetazeta = G_greater_thetazeta
                G_AntiFeynman_thetazeta = G_lesser_thetazeta

                G_Feynman_thetatheta = G_greater_thetatheta
                G_AntiFeynman_thetatheta = G_lesser_thetatheta

            B[4 * i, 4 * j] = -1j * G_Feynman_zetazeta * \
                np.tan(Jy)**2 / T_xy**2
            B[4 * i, 4 * j + 1] = -1j * \
                G_lesser_zetazeta * np.tan(Jy)**2 / T_xy**2
            B[4 * i + 1, 4 * j] = -1j * \
                G_greater_zetazeta * np.tan(Jy)**2 / T_xy**2
            B[4 * i + 1, 4 * j + 1] = -1j * \
                G_AntiFeynman_zetazeta * np.tan(Jy)**2 / T_xy**2

            B[4 * i, 4 * j + 2] = -1 * G_Feynman_zetatheta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i, 4 * j + 3] = G_lesser_zetatheta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i + 1, 4 * j + 2] = -1 * G_greater_zetatheta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i + 1, 4 * j + 3] = G_AntiFeynman_zetatheta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2

            B[4 * i + 2, 4 * j] = -1 * G_Feynman_thetazeta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i + 2, 4 * j + 1] = -1 * G_lesser_thetazeta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i + 3, 4 * j] = G_greater_thetazeta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2
            B[4 * i + 3, 4 * j + 1] = G_AntiFeynman_thetazeta * \
                np.tan(Jy) * np.tan(Jx) / T_xy**2

            B[4 * i + 2, 4 * j + 2] = 1j * \
                G_Feynman_thetatheta * np.tan(Jx)**2 / T_xy**2
            B[4 * i + 2, 4 * j + 3] = -1j * \
                G_lesser_thetatheta * np.tan(Jx)**2 / T_xy**2
            B[4 * i + 3, 4 * j + 2] = -1j * \
                G_greater_thetatheta * np.tan(Jx)**2 / T_xy**2
            B[4 * i + 3, 4 * j + 3] = 1j * \
                G_AntiFeynman_thetatheta * np.tan(Jx)**2 / T_xy**2

    for i in range(ntimes):  # trivial factor

        B[4 * i, 4 * i + 2] += 0.5
        B[4 * i + 1, 4 * i + 3] -= 0.5
        B[4 * i + 2, 4 * i] -= 0.5
        B[4 * i + 3, 4 * i + 1] += 0.5

    # factor 2 to fit Alessio's notes where we have 1/2 B in exponent of influence matrix
    B = np.dot(2, B)

    real_check = 0
    for i in range(4 * ntimes):
        for j in range(4 * ntimes):
            real_check += abs(np.imag(B[i, j]))
    print 'real_check', real_check

    random_part_real = np.random.rand(4 * ntimes, 4 * ntimes) * 1e-10
    random_part_imag = np.random.rand(4 * ntimes, 4 * ntimes) * 1e-10 * 1j

    for i in range(4*ntimes):
        for j in range(i, 4*ntimes):
            if i == j:
                random_part_real[i, j] = 0
                random_part_imag[i, j] = 0
                B[i, j] = 0.
            else:
                random_part_real[j, i] = - random_part_real[i, j]
                random_part_imag[j, i] = - random_part_imag[i, j]
                B[i, j] = - B[j, i]

    B += (random_part_real)

    print('computing all temporal correlations')

    print('B\n')
    print B

    # G_schur,R=linalg.schur(np.dot(1j,G_total))
    ews, ves = linalg.eig(np.dot(1j, B))

    argsort = np.argsort(- np.sign(np.real(ews)) * np.abs(ews))
    ves_sorted = np.zeros((4 * ntimes, 4 * ntimes), dtype=np.complex_)
    ews_sorted = np.zeros(4 * ntimes, dtype=np.complex_)

    for i in range(2 * ntimes):
        ves_sorted[:, i] = ves[:, argsort[i]]
        ves_sorted[:, 4 * ntimes - 1 - i] = ves[:, argsort[i + 2 * ntimes]]
        ews_sorted[i] = ews[argsort[i]]
        ews_sorted[4 * ntimes - 1 - i] = ews[argsort[i + 2 * ntimes]]
    print('ews_sorted', ews_sorted)

    R = np.zeros((4 * ntimes, 4 * ntimes))
    for i in range(0, 2 * ntimes):
        R[:, 2 * i] = np.real(ves_sorted[:, i]) * 2**0.5
        R[:, 2 * i + 1] = np.imag(ves_sorted[:, i]) * 2**0.5

    print 'R', R
    schur = np.dot(R.T, B)
    schur = np.dot(schur, R)
    print 'schur'
    print schur

    T, Z = linalg.schur(1j*B, output='complex')
    print np.diag(T)
    print 'T', T
    # print 'Z', Z

    argsort2 = np.argsort(- np.sign(np.real(np.diag(T))) * np.abs(np.diag(T)))
    print argsort2
    ves_sorted2 = np.zeros((4 * ntimes, 4 * ntimes), dtype=np.complex_)
    ews_sorted2 = np.zeros(4 * ntimes, dtype=np.complex_)

    for i in range(2 * ntimes):
        ves_sorted2[:, (2 * i) + 1] = (Z[:, argsort2[(4 * ntimes) - 1 - i]])
        ves_sorted2[:, 2 * i] = (Z[:, argsort2[i]])
        ews_sorted2[(2 * i) + 1] = np.diag(T)[argsort2[(4 * ntimes) - 1 - i]]
        ews_sorted2[2 * i] = np.diag(T)[argsort2[i]]

    print 'ews_sort2', ews_sorted2

    print ves_sorted2
    #ves_sorted2 *= 2**0.5

    for i in range(2 * ntimes):
        ves_sorted2[:, 2*i] *= np.exp(-1j * np.angle(ves_sorted2[0, 2*i]))
        ves_sorted2[:, 2*i + 1] *= np.exp(1j * np.angle(
            ves_sorted2[0, 2*i]/ves_sorted2[0, 2*i + 1]))
    ves_sorted2 *= np.exp(-1j * np.pi/4)
    print '2', ves_sorted2
    # for i in range (2 * ntimes):
    #    ves_sorted2[:,2*i +1] =  np.real(ves_sorted2[:,2*i] )*2**0.5
    #    ves_sorted2[:,2*i] = np.imag(ves_sorted2[:,2*i] )*2**0.5
    # print '3',ves_sorted2

    G_schur = np.dot(ves_sorted2.T, B)
    G_schur = np.dot(G_schur, ves_sorted2)

    #R = ves_sorted2
    # print ews_sorted
    # for i in range(2 * ntimes):
    #    ews_sorted[i] = G_schur[2 * i,2 * i + 1]
    #    ews_sorted[i + 2 * ntimes] = G_schur[2 * i +1,2 * i]

    print ews_sorted

    schur_check = 0

    for i in range(4 * ntimes):
        for j in range(i, 4 * ntimes):
            schur_check += abs(G_schur[i, j])
    for i in range(2 * ntimes):
        schur_check -= ews_sorted[i]

    print('Check that R yields Schur form \n', schur_check, '\nG_schur\n')
    print(G_schur)
    identity_check = np.dot(R, R.conj().T)
    # , '\n', identity_check
    print('unity_check\n', np.trace(identity_check)/(4 * ntimes))

    #br = np.dot(R, G_schur)
    #br = np.dot(br, R.T)
    # print 'Back_rotated\n', br
    # compute correlation function block
    corr_block_diag = np.zeros((8 * ntimes, 8 * ntimes))

    ews_sorted = np.real(ews_sorted)
    for i in range(0, 2 * ntimes):
        #Theta = np.arctan(G_schur[i,i + 1])
        Theta = np.arctan(ews_sorted[i])
        corr_block_diag[2 * i, 2 * i] = np.cos(Theta)**2
        corr_block_diag[2 * i + 1, 2 * i + 1] = np.cos(Theta)**2

        corr_block_diag[2 * i, 2 * i + 4 * ntimes + 1] = - \
            np.cos(Theta)*np.sin(Theta)
        corr_block_diag[2 * i + 1, 2 * i + 4 *
                        ntimes] = np.cos(Theta)*np.sin(Theta)

        corr_block_diag[2 * i + 4 * ntimes, 2 *
                        i + 1] = np.cos(Theta)*np.sin(Theta)
        corr_block_diag[2 * i + 4 * ntimes + 1, 2 * i] = - \
            np.cos(Theta)*np.sin(Theta)

        corr_block_diag[2 * i + 4 * ntimes, 2 *
                        i + 4 * ntimes] = np.sin(Theta)**2
        corr_block_diag[2 * i + 4 * ntimes + 1, 2 * i +
                        4 * ntimes + 1] = np.sin(Theta)**2

    corr_block_diag2 = np.zeros((8 * ntimes, 8 * ntimes), dtype=np.complex_)
    for i in range(0, 2 * ntimes):
        #Theta = np.arctan(G_schur[i,i + 1])
        ew = ews_sorted[i]
        norm = 1 + abs(ew)**2
        corr_block_diag2[2 * i, 2 * i] = 1/norm
        corr_block_diag2[2 * i + 1, 2 * i + 1] = 1/norm

        corr_block_diag2[2 * i, 2 * i + 4 * ntimes + 1] = - ew/norm
        corr_block_diag2[2 * i + 1, 2 * i + 4 *
                         ntimes] = ew/norm

        corr_block_diag2[2 * i + 4 * ntimes, 2 *
                         i + 1] = ew.conj()/norm
        corr_block_diag2[2 * i + 4 * ntimes + 1, 2 * i] = - ew.conj()/norm

        corr_block_diag2[2 * i + 4 * ntimes, 2 *
                         i + 4 * ntimes] = abs(ew)**2/norm
        corr_block_diag2[2 * i + 4 * ntimes + 1, 2 * i +
                         4 * ntimes + 1] = abs(ew)**2/norm

    compare_blocks = 0
    for i in range(0, 8*ntimes):
        for j in range(0, 8*ntimes):
            compare_blocks += abs(corr_block_diag[i,
                                  j] - corr_block_diag2[i, j])
    print 'compare_block', compare_blocks

    #corr_block_diag = corr_block_diag2

    # print 'corr_block_diag\n', corr_block_diag
    double_R = np.bmat([[R, np.zeros((4 * ntimes, 4 * ntimes))],
                       [np.zeros((4 * ntimes, 4 * ntimes)), R]])

    # print 'double R\n', double_R
    identity_check2 = np.dot(double_R.conj().T, double_R)
    print 'unity_check2\n', np.trace(identity_check2)/(8 * ntimes), '\n', identity_check

    corr_block_back_rotated = np.dot(double_R, corr_block_diag)
    corr_block_back_rotated = np.dot(corr_block_back_rotated, double_R.T)
    # print 'corr_block_back_rotated\n', corr_block_back_rotated

    eigenvalues_correlations, ev_correlations = eigsh(
        corr_block_diag, 8*ntimes)
    print(ntimes, eigenvalues_correlations)
    eigenvalues_correlations, ev_correlations = eigsh(
        corr_block_back_rotated, 8*ntimes)
    print(eigenvalues_correlations)

    return corr_block_back_rotated


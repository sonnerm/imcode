import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]

# mode= 1 for hermitian matrix, mode = 1 for skew symmetric, real matrix


def reorder_eigenvecs(M, half_matrix_dimension, mode=1):
    # rearrange M to bring it into from ((A, B.conj()), (B, A.conj()))
    counter = 0  # to check if rearrangement was correct
    if mode == 1:
        # adjust relative phases of eigenvectors
        for i in range(half_matrix_dimension):
            for j in range(half_matrix_dimension):
                if abs(M[j, i]) > 1e-10:
                    # print 'multiplying coloumn ', i+ half_matrix_dimension, 'with sign', np.sign(np.real(M[j, i])) * np.sign(np.real(M[j + half_matrix_dimension, i + half_matrix_dimension]))
                    M[:, i + half_matrix_dimension] *= np.sign(np.real(M[j, i])) * np.sign(
                        np.real(M[j + half_matrix_dimension, i + half_matrix_dimension]))
                    break

        # check if rearranement was correct
        for column in range(0, half_matrix_dimension):
            for line in range(0, half_matrix_dimension):
                if abs(M[line, column] - M[line + half_matrix_dimension, column + half_matrix_dimension].conj()) < 1e-8 or M[line, column] < 1e-6:
                    counter += 1
                if abs(M[line + half_matrix_dimension, column] - M[line, column + half_matrix_dimension].conj()) < 1e-8 or M[line + half_matrix_dimension, column] < 1e-6:
                    counter += 1

    elif mode == 0:
        for column in range(0, 2 * half_matrix_dimension, 2):
            for i in range(column + 1, 2 * half_matrix_dimension):
                checker = 0
                diff = 0
                for j in range(0, 2 * half_matrix_dimension):
                    diff = abs(abs(M[j, column]) - abs(M[j, i]))
                    if diff < 1e-6:
                        checker += 1
                if checker == 2 * half_matrix_dimension:
                    M[:, [column + 1, i]] = M[:, [i, column + 1]]
                    # print 'switched columns', column + 1,  'and ', i
                    break

        # adjust relative phases of eigenvectors THIS NEEDS TO BE UPDATED SINCE ALSO COMPLEX PHASES SHOULD BE DETECTED
        for i in range(0, 2 * half_matrix_dimension - 1, 2):
            for j in range(2 * half_matrix_dimension):
                if abs(M[j, i]) > 1e-10:
                    # print 'multiplying coloumn ', i + 1, 'with phase factor', np.angle(M[j, i] / M[j, i + 1].conj())
                    M[:, i + 1] *= np.exp(1j *
                                          np.angle(M[j, i] / M[j, i + 1].conj()))
                    break

        for i in range(0, 2 * half_matrix_dimension, 2):
            for j in range(2 * half_matrix_dimension):
                if abs(M[j, i]) > 1e-10:
                    # print 'multiplying coloumn ', i + 1, 'with sign', np.sign(np.real(M[j, i])) * np.sign(np.real(M[j, i + 1]))
                    M[:, i + 1] *= np.sign(np.real(M[j, i])) * \
                        np.sign(np.real(M[j, i + 1]))
                    break

        # check if rearranement was correct
        for column in range(0, half_matrix_dimension):
            for line in range(0, 2 * half_matrix_dimension):
                if abs(M[line, column] - M[line, column + 1].conj()) < 1e-6 or M[line, column] < 1e-6:
                    counter += 1

    if counter == 2 * half_matrix_dimension * half_matrix_dimension:
        print('Matrix constructed successfully.. ')
        # print M
    else:

        print('Erroneous rearrangement!', counter)
        # print M
    return M


def matrix_diag(nsites, Jx=0, Jy=0, g=0):
    # define generators for unitary transformation

    # G_XY - two-site gates (XX + YY)
    G_XY_odd = np.zeros((2 * nsites, 2 * nsites))
    G_XY_even = np.zeros((2 * nsites, 2 * nsites))

    Jp = (Jx + Jy)
    Jm = (Jy - Jx)

    if abs(Jm) < 1e-10:
        Jm = 1e-10
    if abs(g) < 1e-10:
        g = 1e-10

    eps = 1e-6  # lift degeneracy
    G_XY_odd[0, nsites - 1] += eps
    G_XY_odd[nsites - 1, 0] += eps
    G_XY_odd[nsites, 2 * nsites - 1] += -eps
    G_XY_odd[2 * nsites - 1, nsites] += -eps

    G_XY_odd[nsites - 1, nsites] -= eps
    G_XY_odd[0, 2 * nsites - 1] += eps
    G_XY_odd[2 * nsites - 1, 0] += eps
    G_XY_odd[nsites, nsites - 1] -= eps

    for i in range(0, nsites - 1, 2):
        G_XY_even[i, i + 1] = Jp
        G_XY_even[i + 1, i] = Jp
        G_XY_even[i, i + nsites + 1] = -Jm
        G_XY_even[i + 1, i + nsites] = Jm
        G_XY_even[i + nsites, i + 1] = Jm
        G_XY_even[i + nsites + 1, i] = -Jm
        G_XY_even[i + nsites, i + nsites + 1] = -Jp
        G_XY_even[i + nsites + 1, i + nsites] = -Jp

    for i in range(1, nsites - 1, 2):
        G_XY_odd[i, i + 1] = Jp
        G_XY_odd[i + 1, i] = Jp
        G_XY_odd[i, i + nsites + 1] = -Jm
        G_XY_odd[i + 1, i + nsites] = Jm
        G_XY_odd[i + nsites, i + 1] = Jm
        G_XY_odd[i + nsites + 1, i] = -Jm
        G_XY_odd[i + nsites, i + nsites + 1] = - Jp
        G_XY_odd[i + nsites + 1, i + nsites] = - Jp

    # G_g - single body kicks
    G_g = np.zeros((2 * nsites, 2 * nsites))
    for i in range(nsites):
        G_g[i, i] = - 2 * g
        G_g[i + nsites, i + nsites] = 2 * g

    # G_1 - residual gate coming from projecting interaction gate of xy-model on the vacuum at site 0
    G_1 = np.zeros((2 * nsites, 2 * nsites))

    beta_tilde = np.arctanh(np.tan(Jx) * np.tan(Jy))

    G_1[0, 0] = 2 * beta_tilde
    G_1[nsites, nsites] = -2 * beta_tilde

    # give out explicit form of generators
    print('G_XY_even = ')
    print(G_XY_even)

    print('G_XY_odd = ')
    print(G_XY_odd)

    print('G_g = ')
    print(G_g)

    print('G_1 = ')
    print(G_1)

    # unitary gate is exp-product of exponentiated generators
    U_XY = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:
        print('Combined even and odd gate')
        print('G=')
        print(G_XY_even + G_XY_odd)
        G_XY = G_XY_even + G_XY_odd
        U_XY = expm(1j * G_XY)
    else:
        # note that there is no imaginary j in from of G_1
        U_XY = np.matmul(expm(1.j * G_XY_even), expm(1.j * G_XY_odd))
        U_XY = np.matmul(U_XY, expm(G_1))

    if abs(g) > 1e-8:
        U = np.matmul(expm(1j*G_g), U_XY)
    else:
        U = U_XY

    print('U= ', U)
    # G_eff is equivalent to generator for composed map (in principle obtainable through Baker-Campbell-Hausdorff)
    G_eff = -1j * linalg.logm(U)

    print('G_eff = ')
    print(G_eff)

    random_part = np.random.rand(2 * nsites, 2 * nsites) * 1e-10
    # symmetrize random part
    for i in range(2*nsites):
        for j in range(i, 2*nsites):
            random_part[i, j] = random_part[j, i]
    G_eff += random_part
    # compute eigensystem of G_eff. Set of eigenvectors "eigenvectors_G_eff" diagnonalizes G_eff

    eigenvalues_G_eff = np.zeros(2 * nsites, dtype=np.complex_)
    eigenvectors_G_eff = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    if abs(Jy) < 1e-10 or abs(Jx) < 1e-10:
        eigenvalues_G_eff, eigenvectors_G_eff = linalg.eig(
            0.5 * (G_eff + G_eff.conj().T))
    else:
        eigenvalues_G_eff, eigenvectors_G_eff = linalg.eig(G_eff)

    eigenvector_check = 0
    for i in range(nsites):
        eigenvector_check += linalg.norm(np.dot(G_eff, eigenvectors_G_eff[:, i]) - np.dot(
            eigenvalues_G_eff[i], eigenvectors_G_eff[:, i]))
    print 'eigenvector_check', eigenvector_check

    argsort = np.argsort(- eigenvalues_G_eff)
    ves_sorted = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex_)
    ews_sorted = np.zeros(2 * nsites, dtype=np.complex_)

    for i in range(nsites):  # sort eigenvectors and eigenvalues such that the first half are the ones with positive real part, and the second half have negative real parts
        ves_sorted[:, i] = eigenvectors_G_eff[:, argsort[i]]
        ves_sorted[:, 2 * nsites - 1 -
                   i] = eigenvectors_G_eff[:, argsort[i + nsites]]
        ews_sorted[i] = eigenvalues_G_eff[argsort[i]]
        ews_sorted[2 * nsites - 1 - i] = eigenvalues_G_eff[argsort[i + nsites]]

    M = ves_sorted

    M = reorder_eigenvecs(M, nsites)  # adjust phases

    eigenvector_check = 0
    for i in range(nsites):
        eigenvector_check += linalg.norm(np.dot(G_eff,
                                         M[:, i]) - np.dot(ews_sorted[i], M[:, i]))
    print 'eigenvector_check 2', eigenvector_check

    print('M\n')
    print(M)

    # diagonalize G_eff with eigenvectors to check:
    # D_temp = np.dot(M.conj().T, G_eff)#works only in unitary case
    D_temp = np.dot(linalg.inv(M), G_eff)
    D = np.dot(D_temp, M)# this is the diagonal matrix with eigenvalues of G_eff on the diagonal
    print('D= ')
    print(D)
    diag_check = 0

    for i in range(0, 2 * nsites):
        for j in range(i + 1, 2 * nsites):
            diag_check += abs(D[i, j])
    print 'diag_check', diag_check


    eigenvalues_G_eff = D.diagonal()# this makes sure that the order of the eigenvalues corresponds to the order of the eigenvectors in the matrix M
    #eigenvalues_G_eff = ews_sorted
    print ('eigenvalues of G_eff: ')
    print(eigenvalues_G_eff)

    print('Diagonalization of Generator completed..')
    return M, np.real(eigenvalues_G_eff), eigenvalues_G_eff.size / 2 # this is only valid in the unitary case where the eigenvalues are indeed real


# i and j are site indices, s and sp specify whether the fermionic operators have a dagger (=1) or not (=0), t2 and t1 denote times, M is the matrix of eigenvetors (as columns) and eigenvalues_G_eff contains eigenvalues of G_eff )
# returns greater correlation function
def correlator(M, eigenvalues_G_eff, s, sp, i, j, t2, t1, beta=0):
    result = 0
    nsites = int(eigenvalues_G_eff.size / 2)
    if beta > 0:
        for k in range(2 * nsites):
            for kp in range(2 * nsites):
                for l in range(0, nsites):
                    result += M[j + sp * nsites, kp] * M[i + s * nsites, k] * np.exp(-1j * (
                        eigenvalues_G_eff[kp] * t2 + eigenvalues_G_eff[k] * t1)) * (M.conj().T[k, l] * M.conj().T[kp, l + nsites] * 1/(1 + np.exp(beta)) + M.conj().T[k, l + nsites] * M.conj().T[kp, l] * 1/(1 + np.exp(-beta)))
    else:
        for k in range(nsites):

            result += 0.5 * M[j + (1 - sp) * nsites, k].conj() * M[i + s * nsites, k] * np.exp(1j * eigenvalues_G_eff[k] * (
                t2 - t1)) + 0.5 * M[j + sp * nsites, k] * M[i + (1-s) * nsites, k].conj() * np.exp(-1j * eigenvalues_G_eff[k] * (t2 - t1))
    return result


np.set_printoptions(precision=6, suppress=True)


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


def entropy(correlation_block, ntimes, time_cut):
    if time_cut == 0:
        # take this as default value if nothing has been specified otherwise
        time_cut = max(ntimes / 2, 1)

    correlation_block_reduced = np.bmat([[correlation_block[0: 4 * time_cut, 0:  4 * time_cut], correlation_block[0: 4 * time_cut, 4 * ntimes: 4 * (time_cut + ntimes)]], [
        correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 0:  4 * time_cut], correlation_block[4 * ntimes: 4 * (time_cut + ntimes), 4 * ntimes: 4 * (time_cut + ntimes)]]])
    # print correlation_block_reduced
    eigenvalues_correlations, ev_correlations = eigsh(
        correlation_block_reduced, 8 * time_cut)
    eigenvalues_correlations[::-1].sort()
    # print 'cut:' , cut , ', reduced eigenvalue correlations:',eigenvalues_correlations

    entropy = 0
    for i in range(0, 4 * time_cut, 1):
        kappa = eigenvalues_correlations[i]
        if kappa < 1:  # for kappa = 1, entropy has no contribution
            entropy += - kappa * np.log(kappa) - (1-kappa) * np.log(1-kappa)

    return entropy


max_time1 = 20
max_time2 = 20
stepsize1 = 8
stepsize2 = 16
entropy_values = np.zeros(
    (int(max_time1/stepsize1) + int(max_time2/stepsize2) + 3, max_time2 + stepsize2))
times = np.zeros(int(max_time1/stepsize1) + int(max_time2/stepsize2))
print 'here', int(max_time1/stepsize1), int(max_time2/stepsize2)
nsites = 150


Jx = 0
Jy = 1.06
g = np.pi/4
beta = 0
iterator = 1

fig, ax = plt.subplots(2)

M, eigenvalues_G_eff, nsites = matrix_diag(nsites, Jx, Jy, g)
f = 0
for k in range(nsites):
    f += abs(M[0, k])**2 - abs(M[nsites, k])**2 + \
        2j * imag(M[0, k]*M[nsites, k].conj())
T_xy = 1 / (1 + f * np.tan(Jx) * np.tan(Jy))
print 'T_xy', T_xy, 'f=', f


for time in range(stepsize1, max_time1, stepsize1):  # 90, nsites = 200,
    correlation_block = create_correlation_block(
        M, eigenvalues_G_eff, nsites, time, Jx, Jy, g, beta, T_xy, f)
    time_cuts = np.arange(1, time)
    #times[iterator] = time
    entropy_values[iterator, 0] = time
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, time, cut)
    iterator += 1


for time in range(max_time1, max_time2 + stepsize2, stepsize2):  # 90, nsites = 200,
    correlation_block = create_correlation_block(
        M, eigenvalues_G_eff, nsites, time, Jx, Jy, g, beta, T_xy, f)
    time_cuts = np.arange(1, time)
    #times[iterator] = time
    entropy_values[iterator, 0] = time
    for cut in time_cuts:
        entropy_values[iterator, cut] = entropy(correlation_block, time, cut)
    iterator += 1

print(entropy_values)

max_entropies = np.zeros(iterator)
half_entropies = np.zeros(iterator)
for i in range(iterator):
    max_entropies[i] = max(entropy_values[i, 1:])
    if entropy_values[i, 0] % 2 == 0:
        halftime = entropy_values[i, 0] / 2
        half_entropies[i] = entropy_values[i, int(halftime)]

print(max_entropies)
print(half_entropies)


ax[0].plot(entropy_values[:iterator, 0], max_entropies,
           'ro-', label=r'$max_t S$, ' + r'$J_x={},J_y={}, g={}, \beta = {}, L={}$'.format(Jx, Jy, g, beta, nsites))
ax[0].plot(entropy_values[:iterator, 0], zero_to_nan(half_entropies),
           'ro--', label=r'$S(t/2)$, ' + r'$J_x={},J_y={}, g={}, \beta = {}, L={}$'.format(Jx, Jy, g, beta, nsites), color='green')
ax[0].set_xlabel(r'$t$')

ax[0].yaxis.set_ticks_position('both')
ax[0].tick_params(axis="y", direction="in")
ax[0].tick_params(axis="x", direction="in")
ax[0].legend(loc="lower right")
# ax[0].set_ylim([0,1])
ax[0].set_xlabel(r'$t$')


print eigenvalues_G_eff
gamma_test_range = 100
gamma_test_vals = np.zeros(gamma_test_range)
for i in range(gamma_test_range):
    gamma_test = 0
    for k in range(nsites):
        gamma_test += abs(M[0, k] - M[nsites, k])**2 * \
            np.cos(eigenvalues_G_eff[k] * i)
    gamma_test_vals[i] = gamma_test
    # print 'i=',i, abs(M[0,i] - M[nsites,i])**2 , eigenvalues_G_eff[i], '\n'
    # print 'gamma_test', i, gamma_test

#ax[1].plot(np.arange(0,2 * nsites), np.sort(eigenvalues_G_eff),'.')

ax[1].plot(np.arange(0, gamma_test_range), gamma_test_vals,
           'ro-', label=r'$\gamma(t)$')
#ax[1].plot(np.arange(0,gamma_test_range), 5 * np.arange(0,gamma_test_range, dtype=float)**(-1.5), label= r'$t^{-3/2}$')
print('gamma', gamma_test_vals[0])
ax[1].set_xlabel(r'$t$')
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# ax[1].set_ylim([1e-6,1])
ax[1].legend(loc="lower left")

ax[1].yaxis.set_ticks_position('both')
ax[1].tick_params(axis="y", direction="in")
ax[1].tick_params(axis="x", direction="in")

np.savetxt('../../../../data/correlation_approach/ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' + str(g) + '_beta=' +
           str(beta) + '_L=' + str(nsites) + '.txt', entropy_values,  delimiter=' ', fmt='%1.5f')


plt.savefig('../../../../data/correlation_approach/ent_entropy_Jx=' + str(Jx) + '_Jy=' + str(Jy) + '_g=' +
            str(g) + '_beta=' + str(beta) + '_L=' + str(nsites) + '.png')

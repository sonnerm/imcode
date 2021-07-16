import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)


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
                if abs(M[line, column]) < 1e-6 or abs(M[line, column] - M[line + half_matrix_dimension, column + half_matrix_dimension].conj())/abs(M[line, column]) < 1e-6:
                    counter += 1
                #else: 
                #    print ('incorrect reordering', line, column, line + half_matrix_dimension, column + half_matrix_dimension, abs(M[line, column] - M[line + half_matrix_dimension, column + half_matrix_dimension].conj())/abs(M[line, column]))
                if abs(M[line + half_matrix_dimension, column]) < 1e-6 or abs(M[line + half_matrix_dimension, column] - M[line, column + half_matrix_dimension].conj())/abs(M[line + half_matrix_dimension, column]) < 1e-6 :
                    counter += 1
                #else:
                #    print ('incorrect reordering', line + half_matrix_dimension, column, line, column + half_matrix_dimension, abs(M[line + half_matrix_dimension, column] - M[line, column + half_matrix_dimension].conj())/abs(M[line + half_matrix_dimension, column]) )

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

        print('Erroneous rearrangement!', 2 * half_matrix_dimension * half_matrix_dimension- counter)
        # print M
    return M


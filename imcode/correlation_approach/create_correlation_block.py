from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)
from IM_exponent import IM_exponent
from add_cmplx_random_antisym import add_cmplx_random_antisym
from rotation_matrix_for_schur import rotation_matrix_for_schur

def create_correlation_block(M_fw, M_fw_inverse, M_bw, M_bw_inverse, N_t, eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, ntimes, Jx, Jy, T_xy, rho_t):
    print('Creating Greens function for time ', ntimes)

    # compute all correlators from which we can construct Keldysh Greens functions
    B = IM_exponent(M_fw, M_fw_inverse, M_bw, M_bw_inverse, N_t, eigenvalues_G_eff_fw, eigenvalues_G_eff_bw, nsites, ntimes, Jx, Jy, T_xy, rho_t)
    
    B = add_cmplx_random_antisym(B, 1e-8)#add random antisymmetric part to matrix to lift degeneracies and stabilize numerics

    print('computing all temporal correlations')

    print('B\n')
    print B

    R, ews_sorted = rotation_matrix_for_schur(B)
    corr_block_diag = np.zeros((8 * ntimes, 8 * ntimes))

    ews_sorted = np.real(ews_sorted)##enable these to be complex!!!!!
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
    print 'unity_check2\n', np.trace(identity_check2)/(8 * ntimes)

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


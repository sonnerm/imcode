from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
from numpy import version
from numpy.lib.type_check import imag
from scipy.linalg import expm, schur, eigvals
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
np.set_printoptions(suppress=False, linewidth=np.nan)
from rotation_matrix_for_schur import rotation_matrix_for_schur
from numpy.core.einsumfunc import einsum

def create_correlation_block(B):
    dim_B = len(B)

    print('Creating correlation block for total time', dim_B /4)
    R = np.zeros((dim_B, dim_B), dtype=np.complex_)
    B_schur = np.zeros((dim_B, dim_B), dtype=np.complex_)
    R, B_schur = rotation_matrix_for_schur(B)
    print 'h1'
    """
    corr_block_diag = np.zeros((2 * dim_B, 2*dim_B))#REAL - WORKS ONLY IN ISING CASE
    #ews_sorted = ews_sorted)##enable these to be complex!!!!!
    for i in range(0, dim_B / 2):
        #Theta = np.arctan(G_schur[i,i + 1])
        Theta = np.arctan(ews_sorted[i])
        corr_block_diag[2 * i, 2 * i] = np.cos(Theta)**2
        corr_block_diag[2 * i + 1, 2 * i + 1] = np.cos(Theta)**2

        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - \
            np.cos(Theta)*np.sin(Theta)
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = np.cos(Theta)*np.sin(Theta)

        corr_block_diag[2 * i + dim_B, 2 * i + 1] = np.cos(Theta)*np.sin(Theta)
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - \
            np.cos(Theta)*np.sin(Theta)

        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = np.sin(Theta)**2
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = np.sin(Theta)**2
    """
    corr_block_diag2 = np.zeros((2 * dim_B, 2 * dim_B), dtype=np.complex_)
    for i in range(0, dim_B / 2):
        ew = B_schur[i,i + 1]
        #ew = ews_sorted[i]
        norm = 1 + abs(ew)**2
        corr_block_diag2[2 * i, 2 * i] = 1/norm
        corr_block_diag2[2 * i + 1, 2 * i + 1] = 1/norm

        corr_block_diag2[2 * i, 2 * i + dim_B + 1] = - ew/norm
        corr_block_diag2[2 * i + 1, 2 * i + dim_B] = ew/norm

        corr_block_diag2[2 * i + dim_B, 2 *
                         i + 1] = ew.conj()/norm
        corr_block_diag2[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm

        corr_block_diag2[2 * i + dim_B, 2 *
                         i + dim_B] = abs(ew)**2/norm
        corr_block_diag2[2 * i + dim_B + 1, 2 * i +
                         dim_B + 1] = abs(ew)**2/norm

    print 'h2'
    """
    compare_blocks = 0
    for i in range(0, 2 * dim_B):
        for j in range(0, 2 * dim_B):
            compare_blocks += abs(corr_block_diag[i,
                                  j] - corr_block_diag2[i, j])
    print 'compare_block', compare_blocks
    """
    #corr_block_diag = corr_block_diag2

    # print 'corr_block_diag\n', corr_block_diag
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],
                       [np.zeros((dim_B, dim_B),dtype=np.complex_), R]])
    print 'h3'
    print 'double R\n', double_R
    identity_check2 = np.dot(double_R.conj().T, double_R)#check that double_R is unitary just like R is (should be trivial)
    print 'unity_check2\n', np.trace(identity_check2)/(2 * dim_B)

    corr_block_back_rotated = einsum('ij,jk,kl->il',double_R, corr_block_diag2,double_R.T)
    print 'corr_block_back_rotated\n', corr_block_back_rotated
    print 'h4'
    eigenvalues_correlations, ev_correlations = eigsh(
        corr_block_diag2, 2 * dim_B)
    print(dim_B/4, eigenvalues_correlations)
    eigenvalues_correlations, ev_correlations = eigsh(
        corr_block_back_rotated, 2 * dim_B)
    print(eigenvalues_correlations)

    return corr_block_back_rotated


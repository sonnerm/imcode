from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
from scipy import linalg
np.set_printoptions(suppress=False, linewidth=np.nan)
from rotation_matrix_for_schur import rotation_matrix_for_schur

def create_correlation_block(B, ntimes):
    dim_B = 4 * ntimes

    print('Creating correlation block for total time', ntimes)
    R = np.zeros((dim_B, dim_B), dtype=np.complex_)
    R, eigenvalues = rotation_matrix_for_schur(B)#the eigenvalues given out here are always real positive (R is defined in such a way). The negetiave counterparts are not included in this vector.

    #compute correation block in diagonal basis with only entries this phases of fermionic operators are defined such that the eigenvalues of B are real
    corr_block_diag = np.zeros((8 * ntimes, 8 * ntimes))
    for i in range(0, 2 * ntimes):
        ew = eigenvalues[i]
        norm = 1 + abs(ew)**2
        corr_block_diag[2 * i, 2 * i] = 1/norm
        corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm
        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm
        corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm#conjugation is formally correct but has no effect since eigenvalues are real anyways
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm
        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm
    #print ('corr_block_diag\n', corr_block_diag)
    
    #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])

    corr_block_back_rotated = double_R @ corr_block_diag @ double_R.T.conj()#rotate correlation block back from diagonal basis to original fermion basis
    #print ('corr_block_back_rotated\n', corr_block_back_rotated)

    #check that eigenvalues coincide in both bases (half of them should be 0 and half of them should be 1 in both cases)
    eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_diag)
    #print(eigenvalues_correlations)
    eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_back_rotated)
    #print(eigenvalues_correlations)

    return corr_block_back_rotated


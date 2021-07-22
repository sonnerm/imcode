from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
from scipy import linalg
np.set_printoptions(suppress=False, linewidth=np.nan)
from rotation_matrix_for_schur import rotation_matrix_for_schur

def create_correlation_block(B, ntimes):
    dim_B = 4 * ntimes

    print('Creating correlation block for total time', ntimes)
    R = np.zeros((dim_B, dim_B), dtype=np.complex_)
    B_schur = np.zeros((dim_B, dim_B), dtype=np.complex_)
    R, B_schur = rotation_matrix_for_schur(B)

    eigenvalues = np.zeros(2 * ntimes, dtype=np.complex_)
    for i in range(0,2*ntimes):
        eigenvalues[i] = B_schur[2 * i,2 * i + 1]#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.
  

    #compute correation block in diagonal basis with complex entries (is only used as check)
    corr_block_diag = np.zeros((8 * ntimes, 8 * ntimes), dtype=np.complex_)
    for i in range(0, 2 * ntimes):
        ew = eigenvalues[i]
        norm = 1 + abs(ew)**2
        corr_block_diag[2 * i, 2 * i] = 1/norm
        corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm
        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm
        corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm
        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm

    

    #compute correation block in diagonal basis with phases redefined such that all entries are real (this is the correlation block that is actually used for further computation)
    corr_block_diag2 = np.zeros((8 * ntimes, 8 * ntimes), dtype=np.complex_)
    for i in range(0, 2 * ntimes):
        ew = eigenvalues[i]
        norm = 1 + abs(ew)**2
        corr_block_diag2[2 * i, 2 * i] = 1/norm
        corr_block_diag2[2 * i + 1, 2 * i + 1] = 1/norm
        corr_block_diag2[2 * i, 2 * i + dim_B + 1] = - abs(ew)/norm
        corr_block_diag2[2 * i + 1, 2 * i + dim_B] = abs(ew)/norm
        corr_block_diag2[2 * i + dim_B, 2 * i + 1] = abs(ew)/norm
        corr_block_diag2[2 * i + dim_B + 1, 2 * i] = - abs(ew)/norm
        corr_block_diag2[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm
        corr_block_diag2[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm

    #this is the matrix that contains the phases, such that D_phases @ corr_block_diag2 @ D_phases.T.conj() = corr_block_diag
    D_phases = np.zeros((4 * ntimes, 4 * ntimes), dtype=np.complex_)
    for i in range(2 * ntimes):
        D_phases[2 * i,2 * i] = np.exp(0.5j * np.angle(eigenvalues[i]))
        D_phases[2 * i + 1,2 * i + 1] = np.exp(0.5j * np.angle(eigenvalues[i]))


    double_phases = np.bmat([[D_phases, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), D_phases.conj()]])

    #Check that correlation matrix with phases is reproduced correctly
    C_test = double_phases @ corr_block_diag2 @ double_phases.T.conj()#check that this is equivalent to corr_block_diag
    diff = 0
    for i in range (len(C_test)):
        for j in range(len(C_test)):
            diff += abs(C_test[i,j] - corr_block_diag[i,j])
    print ('corr_block_diag (complex)\n', corr_block_diag)
    print ('corr_block_diag2 (real)\n', corr_block_diag2)
    print ('corr_block_test (should be equivalent to corr_block_diag)\n', C_test)
    print ('correlation difference', diff)


    R = R @ D_phases#redefine rotation matrix R to contains the phases.

    #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])

    corr_block_back_rotated = double_R @ corr_block_diag2 @ double_R.T.conj()#rotate correlation block back from diagonal basis to origianal fermion basis
    print ('corr_block_back_rotated\n', corr_block_back_rotated)

    #check that eigenvalues coincide in both bases (half of them should be 0 and half of them should be 1 in both cases)
    eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_diag2)
    print(eigenvalues_correlations)
    eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_back_rotated)
    print(eigenvalues_correlations)

    return corr_block_back_rotated


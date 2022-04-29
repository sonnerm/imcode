from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
import h5py
from datetime import datetime
#from memory_profiler import profile
from scipy import linalg
np.set_printoptions(suppress=False, linewidth=np.nan)
#@profile
def create_correlation_block(B, ntimes, filename):
    dim_B = B.shape[0]
    #print('dimB',dim_B,B.shape[0],4*ntimes)

    print('Creating correlation block for total time', ntimes)
    R = np.zeros((dim_B, dim_B), dtype=np.complex_)
    now = datetime.now()
    #print('Diagonalize B', now)
    R, eigenvalues = rotation_matrix_for_schur(B)#the eigenvalues given out here are always real positive (R is defined in such a way). The negetiave counterparts are not included in this vector.
    now = datetime.now()
    #print('End diag of B', now)
    #compute correation block in diagonal basis with only entries this phases of fermionic operators are defined such that the eigenvalues of B are real
    corr_block_diag = np.zeros((2 * dim_B, 2 * dim_B))
    for i in range(0, dim_B // 2):
        ew = eigenvalues[i] 
        norm = 1 + abs(ew)**2
        corr_block_diag[2 * i, 2 * i] = 1/norm # <d_k d_k^\dagger>
        corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm # <d_{-k} d_{-k}^\dagger>
        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm # <d_k d_{-k}>
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm # <d_{-k} d_{k}>
        corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm #<d_{k}^dagger d_{-k}^\dagger> .. conjugation is formally correct but has no effect since eigenvalues are real anyways
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm #<d_{-k}^dagger d_{k}^\dagger>
        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm #<d_{k}^dagger d_{k}>
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm #<d_{-k}^dagger d_{-k}>
    #print ('corr_block_diag\n', corr_block_diag)
    
    #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])
    
    corr_block_back_rotated = double_R @ corr_block_diag @ double_R.T.conj()#rotate correlation block back from diagonal basis to original fermion basis
    #print ('corr_block_back_rotated\n', corr_block_back_rotated)

    """
    #store for Michael:
    filename_correlations =  filename + '_correlations'
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_t='+ str(ntimes), (corr_block_back_rotated.shape[0],corr_block_back_rotated.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_block_back_rotated[:,:]
    print('Correlations stored for Michael.')
    """
    #check that eigenvalues coincide in both bases (half of them should be 0 and half of them should be 1 in both cases)
    #eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_diag)
    #print(eigenvalues_correlations)
    #eigenvalues_correlations, ev_correlations = linalg.eigh(corr_block_back_rotated)
    #print(eigenvalues_correlations)

    return corr_block_back_rotated


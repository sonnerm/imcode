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

    random_part = np.random.rand(dim_B,dim_B) * 1.e-8
    B += random_part - random_part.T #add small antisymmetric part to make sure that the schur decomposition does not suffer from numerical issues to the degeneracies in B

    hermitian_matrix = B.T @ B.conj()#create hermitian matrix, whose eigenvectors define the rotation matrix that rotates B into Schur form
    eigenvalues_hermitian_matrix, R = linalg.eigh(hermitian_matrix)#compute rotation matrix as eigenvalues of hermitian_matrix, (eigsh is not as reliable)
    
    B_schur_complex = R.T.conj() @ B @ R.conj() #this is the Schur form of B, where the entries are generally complex
    eigenvalues_complex = np.diag(B_schur_complex,k=1)[::2]#define Schur-eigenvalues such that the order is in correspondence with the order of the eigenvectors in R.

    #create matrix that contains the phases, which can be absorbed in R, such that  R.T.conj() @ B @ R.conj() is real and all entries in the upper right triangular matrix are positive.
    D_phases = np.zeros((dim_B, dim_B), dtype=np.complex_)
    np.fill_diagonal(D_phases[::2,::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))
    np.fill_diagonal(D_phases[1::2,1::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))

    #update rotation matrix to include phases, such that Schur-values become real
    R = R @ D_phases #R is generally complex

    B_schur_real = R.T.conj() @ B @ R.conj()#this is Schur-form of B, but now with real Schur-values
    eigenvalues_real = np.real(np.diag(B_schur_real,k=1)[::2])#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.

    #compute correlation block in diagonal basis with only entries this phases of fermionic operators are defined such that the eigenvalues of B are real
    corr_block_diag = np.zeros((2 * dim_B, 2 * dim_B))
    for i in range(0, dim_B // 2):
        ew = eigenvalues_real[i] 
        norm = 1 + abs(ew)**2
        corr_block_diag[2 * i, 2 * i] = 1/norm # <d_k d_k^\dagger>
        corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm # <d_{-k} d_{-k}^\dagger>
        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm # <d_k d_{-k}>
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm # <d_{-k} d_{k}>
        corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm #<d_{k}^dagger d_{-k}^\dagger> .. conjugation is formally correct but has no effect since eigenvalues are real anyways
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm #<d_{-k}^dagger d_{k}^\dagger>
        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm #<d_{k}^dagger d_{k}>
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm #<d_{-k}^dagger d_{-k}>

    #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])
    
    corr_block_back_rotated = double_R @ corr_block_diag @ double_R.T.conj()#rotate correlation block back from diagonal basis to original fermion basis
    
    """#store for Michael:
    filename_correlations =  filename + '_correlations'
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_t='+ str(ntimes), (corr_block_back_rotated.shape[0],corr_block_back_rotated.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_block_back_rotated[:,:]
    print('Correlations stored for Michael.')"""


    """#store for Benedikt:
    filename_correlations =  filename + '_correlations_Benedikt'
    corr_block_back_rotated_Ben = np.zeros(corr_block_back_rotated.shape,dtype=np.complex_)
    corr_block_back_rotated_Ben[::2,::2] = corr_block_back_rotated[:dim_B,:dim_B]
    corr_block_back_rotated_Ben[::2,1::2] = corr_block_back_rotated[:dim_B,dim_B:]
    corr_block_back_rotated_Ben[1::2,::2] = corr_block_back_rotated[dim_B:,:dim_B]
    corr_block_back_rotated_Ben[1::2,1::2] = corr_block_back_rotated[dim_B:,dim_B:]
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_t='+ str(ntimes), (corr_block_back_rotated_Ben.shape[0],corr_block_back_rotated_Ben.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_block_back_rotated_Ben[:,:]
    print('Correlations stored for Benedikt.')"""
    

    return corr_block_back_rotated


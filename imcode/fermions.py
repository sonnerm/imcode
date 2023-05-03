import numpy as np
import pfapack.pfaffian as pf
import pandas as pd
import math
import ttarray as tt
import scipy.integrate as integrate
import numpy.linalg as la
def _get_mat(even,odd):
    ret=np.zeros((2,2,2))
    ret[1,0,1]=odd[0]
    ret[1,1,0]=odd[1]
    ret[0,0,0]=even[0]
    ret[0,1,1]=even[1]
    return ret
_FERMI_A=_get_mat([1,1],[1,1])
_FERMI_B=_get_mat([1,-1],[1,1])
def brickwork_fermi_to_spin(im,truncate=True):
    t=int(math.log2(im.shape[0])/4)
    Omps=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(t*2-1)+[_FERMI_B[...,0][...,None]])
    cprev=im.chi
    im=im*Omps
    if truncate:
        im.truncate(chi_max=max(cprev)) #later exact replication of chi structure
    return im

def fermiexp_to_fermicorr(B):
    # random_part = np.random.rand(dim_B,dim_B) * 1.e-8
    # B += random_part - random_part.T #add small antisymmetric part to make sure that the schur decomposition does not suffer from numerical issues to the degeneracies in B
    # hermitian_matrix = B.T @ B.conj()#create hermitian matrix, whose eigenvectors define the rotation matrix that rotates B into Schur form
    # _, R = la.eigh(hermitian_matrix)#compute rotation matrix as eigenvalues of hermitian_matrix, (eigsh is not as reliable)
    # B_schur_complex = R.T.conj() @ B @ R.conj() #this is the Schur form of B, where the entries are generally complex
    # eigenvalues_complex = np.diag(B_schur_complex,k=1)[::2]#define Schur-eigenvalues such that the order is in correspondence with the order of the eigenvectors in R.
    # #create matrix that contains the phases, which can be absorbed in R, such that  R.T.conj() @ B @ R.conj() is real and all entries in the upper right triangular matrix are positive.
    # D_phases = np.zeros((dim_B, dim_B), dtype=np.complex_)
    # np.fill_diagonal(D_phases[::2,::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))
    # np.fill_diagonal(D_phases[1::2,1::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))
    # #update rotation matrix to include phases, such that Schur-values become real
    # R = R @ D_phases #R is generally complex
    #
    # B_schur_real = R.T.conj() @ B @ R.conj()#this is Schur-form of B, but now with real Schur-values
    # eigenvalues_real = np.real(np.diag(B_schur_real,k=1)[::2])#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.
    #
    # #compute correlation block in diagonal basis with only entries this phases of fermionic operators are defined such that the eigenvalues of B are real
    # corr_block_diag = np.zeros((2 * dim_B, 2 * dim_B))
    # for i in range(0, dim_B // 2):
    #     ew = eigenvalues_real[i] 
    #     norm = 1 + abs(ew)**2
    #     corr_block_diag[2 * i, 2 * i] = 1/norm # <d_k d_k^\dagger>
    #     corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm # <d_{-k} d_{-k}^\dagger>
    #     corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm # <d_k d_{-k}>
    #     corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm # <d_{-k} d_{k}>
    #     corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm #<d_{k}^dagger d_{-k}^\dagger> .. conjugation is formally correct but has no effect since eigenvalues are real anyways
    #     corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm #<d_{-k}^dagger d_{k}^\dagger>
    #     corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm #<d_{k}^dagger d_{k}>
    #     corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm #<d_{-k}^dagger d_{-k}>
    # #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    dim_B = B.shape[0]
    B_large = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)
    B_large[:dim_B, :dim_B] = B.T.conj()*0.5
    B_large[3*dim_B:, 3*dim_B:] = 0.5*B

    for i in range(3):
        i_n = (i+1)*dim_B
        B_large[i_n:i_n+dim_B, i_n-dim_B:i_n] = -0.5 * np.eye(dim_B)
        B_large[i_n-dim_B:i_n, i_n:i_n+dim_B] = 0.5 * np.eye(dim_B)

    B_large_inv = la.inv(B_large)
    corr_pfaff = np.zeros((2*dim_B,2*dim_B),dtype=np.complex_)
    corr_pfaff[:dim_B,:dim_B] = np.eye(dim_B)
    for i in range (dim_B):
        for j in range (dim_B):
            corr_pfaff[i,j] += 0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+2*dim_B,j+dim_B], [i+2*dim_B,j+dim_B]]))
            corr_pfaff[i+dim_B,j+dim_B] += 0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+dim_B,j+2*dim_B], [i+dim_B,j+2*dim_B]]))
            corr_pfaff[i,j+dim_B] += 0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+2*dim_B,j+2*dim_B], [i+2*dim_B,j+2*dim_B]]))
            corr_pfaff[i+dim_B,j] += 0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+dim_B,j+dim_B], [i+dim_B,j+dim_B]]))
    jcorr = corr_pfaff   
    # double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])
    # jcorr = np.array(double_R @ corr_block_diag @ double_R.T.conj())#rotate correlation block back from diagonal basis to original fermion basis
    L=jcorr.shape[0]//2
    jcorrn=np.zeros_like(jcorr)
    jcorrn[::2,::2]=jcorr[:jcorr.shape[0]//2,:jcorr.shape[1]//2]
    jcorrn[1::2,::2]=jcorr[jcorr.shape[0]//2:,:jcorr.shape[1]//2]
    jcorrn[1::2,1::2]=jcorr[jcorr.shape[0]//2:,jcorr.shape[1]//2:]
    jcorrn[::2,1::2]=jcorr[:jcorr.shape[0]//2,jcorr.shape[1]//2:]
    roti=np.diag([1,0]*L,1)[:-1,:-1]+np.diag([1j,0]*L,-1)[:-1,:-1]+np.diag([1,-1j]*L)
    jcorrf=(roti@jcorrn@roti.T.conj())
    jcorrf=(jcorrf-np.diag(np.diag(jcorrf)))/2
    jcorrf[2::4,:]=-jcorrf[2::4,:]
    jcorrf[3::4,:]=-jcorrf[3::4,:]
    jcorrf[:,2::4]=-jcorrf[:,2::4]
    jcorrf[:,3::4]=-jcorrf[:,3::4]
    return jcorrf
def fermicorr_to_circuit(corr,nbcutoff=1e-10):
    import freeferm
    return freeferm.real.corr_to_circuit(corr,nbcutoff)
def circuit_to_mps(circuit,t,chi=128,svdcutoff=1e-10):
    import freeferm
    return freeferm.apply_circuit_to_mps(freeferm.mps_vac(4*t,cluster=((16,),)*t),circuit,chi=128,cutoff=1e-12)


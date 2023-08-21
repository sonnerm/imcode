import numpy as np
import pfapack.pfaffian as pf
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
def fermi_to_spin(im,truncate=True):
    '''
        Converts the IM from fermionic to spin-chain form or vice versa. This
        effectively changes the Jordan Wigner order of the fermions from along the
        MPS order to along the Keldysh contour.
        
    '''
    t=int(math.log2(im.shape[0])/4)
    Omps=tt.frommatrices([_FERMI_A[0,...][None,...]]+[_FERMI_B,_FERMI_A]*(t*2-1)+[_FERMI_B[...,0][...,None]])
    cprev=im.chi
    im=im*Omps
    if truncate:
        im.truncate(chi_max=max(cprev)) #later exact replication of chi structure
    return im

def fermiexp_to_fermicorr(B):
    '''
        Computes the correlation matrix from a fermionic exponent. Might be
        delegated to freeferm in the future
    '''
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
            corr_pfaff[i,j] += 0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+2*dim_B,j+dim_B], [i+2*dim_B,j+dim_B])])
            corr_pfaff[i+dim_B,j+dim_B] += 0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+dim_B,j+2*dim_B], [i+dim_B,j+2*dim_B])])
            corr_pfaff[i,j+dim_B] += 0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+2*dim_B,j+2*dim_B], [i+2*dim_B,j+2*dim_B])])
            corr_pfaff[i+dim_B,j] += 0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+dim_B,j+dim_B], [i+dim_B,j+dim_B])])
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
    '''
        Computes the quantum circuit which converts the vacuum into the
        gaussian state defined by the correlation matrix corr using the
        extended Fishman-White algorithm. Thin wrapper around
        freeferm.real.corr_to_circuit.
    '''
    import freeferm
    return freeferm.real.corr_to_circuit(corr,nbcutoff)
def circuit_to_mps(circuit,t,chi=128,svdcutoff=1e-10):
    '''
        Applies a quantum circuit to the vacuum MPS. Thin wrapper around
        freeferm.apply_circuit_to_mps.
    '''
    import freeferm
    return freeferm.apply_circuit_to_mps(freeferm.mps_vac(4*t,cluster=((16,),)*t),circuit,chi,svdcutoff)

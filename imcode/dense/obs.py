import numpy as np
import numpy.linalg as la
from .utils import rdm_entropy,sz

def folded_entropy(vec):
    L=int(np.log2(len(vec)))
    return rdm_entropy(reduced_density_matrix(list(range(L//4))+list(range(3*L//4,L))))

def flat_entropy(vec):
    L=int(np.log2(len(vec)))
    return rdm_entropy(reduced_density_matrix(list(range(L//4))+list(range(3*L//4,L))))
def entropy(vec):
    pass

def boundary_obs(im,obs):
    return np.sum(im*obs)
def embedded_obs(left_im,obs_op,right_im):
    return np.sum(left_im*(obs_op@right_im))
def zz_state(t):
    ret=np.zeros((2,2**(t-1),2,2**(t-1)))
    ret[0,:,0,:]=1
    ret[0,:,1,:]=-1
    ret[1,:,0,:]=-1
    ret[1,:,1,:]=1
    return np.ravel(ret)
def embedded_czz(im,lop):
    t=int(np.log2(im.shape[0]))//2
    return embedded_obs(im,lop,zz_state(t)*im)
def boundary_czz(im,lop):
    t=int(np.log2(im.shape[0]))//2
    st=lop@zz_state(t)
    return boundary_obs(im,st)
def embedded_norm(im,lop):
    return embedded_obs(im,lop,im)
def boundary_norm(im,lop):
    st=open_boundary_im(t)
    apply(lop,st)
    return boundary_obs(im,lop,im)
# def im_czz(left_im,site_op=None,right_im=None):
#     t=int(np.log2(left_im.shape[0]))
#     return np.sum(zz_op(t)*left_im*site_op*right_im)

def direct_czz(F,t,i,j):
    L=int(np.log2(F.shape[0]))
    return np.trace(la.matrix_power(F,t)@sz(L,i)@la.matrix_power(F.T.conj(),t)@sz(L,j))/(2**(L-1))

def spectral_function(L,eigs,eigv):
    eigs=eigs*(2**L)/2/np.pi # unfolding
    bins=np.arange(0,100,1/10)
    op_sz=np.array(get_imbrie_p(np.array([1.0]+[0.0]*(L-1)),np.zeros((L,)),np.zeros((L,))).todense())
    op_sx=np.array(get_imbrie_p(np.zeros((L,)),np.array([1.0]+[0.0]*(L-1)),np.zeros((L,))).todense())
    M_sx=eigv.T.conj()@op_sx@eigv
    M_sz=eigv.T.conj()@op_sz@eigv
    Ed1,Ed2=np.meshgrid(eigs,eigs)
    Ed=Ed1-Ed2
    hist_sx=np.histogram(Ed,bins,weights=np.abs(M_sx)**2)[0]
    hist_sz=np.histogram(Ed,bins,weights=np.abs(M_sz)**2)[0]
    hist_c=np.histogram(Ed,bins)[0]
    return hist_c,hist_sx,hist_sz
def direct_sff(eigs,tau):
    res=0
    for i1,e1 in enumerate(eigs):
        for e2 in eigs[:i1]:
            res+=np.cos((e1-e2)*tau)*2
    return res
def sfft_sff(sfft,L):
    return

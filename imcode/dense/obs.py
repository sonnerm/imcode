import numpy as np
from .utils import rdm_entropy

def folded_temporal_entropy(vec):
    L=int(np.log2(len(vec)))
    return rdm_entropy(reduced_density_matrix(list(range(L//4))+list(range(3*L//4,L))))

def flat_temporal_entropy(vec):
    L=int(np.log2(len(vec)))
    return rdm_entropy(reduced_density_matrix(list(range(L//4))+list(range(3*L//4,L))))


def im_czz(left_im,site_op=None,right_im=None):
    t=int(np.log2(left_im.shape[0]))
    return np.sum(zz_op(t)*left_im*site_op*right_im)


def direct_czz(F,t,i,j):
    L=int(np.log2(F.shape[0]))
    return np.trace(la.matrix_power(F,t)@sz(L,i)@la.matrix_power(F.T.conj(),t)@sz(L,j))

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

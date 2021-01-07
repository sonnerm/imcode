import numpy as np
from .utils import entropy

def folded_temporal_entropy(vec):
    return entropy([])

def flat_temporal_entropy(vec):
    pass

def entropy(vec):
    pass

def correlator_zz_im(vec):
    pass

def correlator_zz_direct(vec):
    pass

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

def ktau(eigs):
    taus=np.floor(np.exp(np.linspace(0,np.log(len(eigs)),201)))
    res=np.zeros(taus.shape,dtype=float)
    for i1,e1 in enumerate(eigs):
        for e2 in eigs[:i1]:
            res+=np.cos((e1-e2)*taus)*2
    return taus,res

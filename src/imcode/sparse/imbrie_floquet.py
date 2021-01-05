import scipy.sparse as sp
import pickle
import copy
import gmpy
import uuid
import numpy as np
import scipy.sparse.linalg as sla
import numpy.linalg as la
import scipy.linalg as scla
from .ed_utils import *
def get_imbrie_F_p(h,g,J,T):
    F0=np.diag(np.exp(np.diag(-0.5j*T*np.array(get_imbrie_p(h,np.zeros_like(g),J).todense()))))
    U1=scla.hadamard(2**len(h))
    # D1=np.diag((U1@get_imbrie_p(np.zeros_like(h),g,np.zeros_like(J)).todense()@U1.T)/2**len(h))
    D1=np.array(np.diag(get_imbrie_p(g,np.zeros_like(h),np.zeros_like(J)).todense()))
    F1=(U1.T@np.diag(np.exp(-0.5j*T*D1))@U1)/2**len(h)
    return F0@F1
def naive_apply_imbrie_F_p(D1,D2,vec):
    U1=scla.hadamard(D1.shape[0])
    V1=U1@vec/2**(len(h))
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
def calc_imbrie_floquet(L,h,g,j,H,G,J,T,inner):
    res=[]
    for i in range(inner):
        Hs=np.random.uniform(-H+h,H+h,L)
        Gs=np.random.uniform(-G+g,G+g,L)
        Js=np.random.uniform(-J+j,J+j,L)
        F=get_imbrie_F_p(Hs,Gs,Js,T)
        eigs,eigv=la.eigh(F+F.T.conj())
        eigs=np.log(np.diag(eigv.T.conj()@F@eigv)).imag
        sec=trivial_sector(L)
        fid=uuid.uuid4()
        ret={ "model":"imbrie_floquet",
        "H":{"random":"uniform","d":2*H,"m":h,"sample":Hs},
        "G":{"random":"uniform","d":2*G,"m":g,"sample":Gs},
        "J":{"random":"uniform","d":2*J,"m":j,"sample":Js},
        "drive":{"part":"G","T":T},
        "algorithm":"fd", "boundary":"P",
        "central_entropy":get_S(eigv.T,sec),"L":L,
        "floquet_energies":eigs,"uuid":fid}
        # hist_c,hist_sx,hist_sz=spectral_function(L,eigs,eigv)
        # ret["hist_sx"]=hist_sx
        # ret["hist_sz"]=hist_sz
        # ret["hist_c"]=hist_c
        ret["tau"],ret["ktau"]=ktau(eigs)
        res.append(ret)
        np.save(open("wfs/wfs_%s"%str(fid),"wb"),eigv.T)
        # save_ret=copy.deepcopy(ret)

        # save_ret["eigenstates"]=eigv.T
        # pickle.dump(save_ret,open("%s.pickle"%str(fid),"wb"))
    return res

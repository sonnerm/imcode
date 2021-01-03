import numpy as np
import h5py
import os
# import seaborn
# import matplotlib.pyplot as plt
import pickle
import copy
import functools
import gmpy
import numpy.linalg as la
import scipy.sparse as sp
import scipy.linalg as scla
import scipy.sparse.linalg as spla
def trivial_sector(L):
    return (L,{i:i for i in range(2**L)},[i for i in range(2**L)])

def write_dict_to_hdfobj(hdfobj,pre,d):
    if isinstance(d,dict):
        for k,v in d.items():
            write_dict_to_hdfobj(hdfobj,"%s/%s"%(pre,k),v)
    elif not isinstance(d,dict):
        # print((type(d),pre))
        hdfobj[pre]=d
@functools.lru_cache(None)
def disorder_sector(L):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**(2*L)):
        if gmpy.popcount(i>>L)==gmpy.popcount(i^((i>>L)<<L)):
            sec[i]=cn
            invsec.append(i)
            cn+=1
    return (2*L,sec,invsec)
def get_imbrie(h,g,J):
    h=h[::-1]
    g=g[::-1]
    J=J[::-1]
    print("h,g,J: %s %s %s"%(str(h),str(g),str(J)))
    mat=sp.dok_matrix((2**len(h),2**len(h)),dtype=complex)
    pm_mask=0b01
    mp_mask=0b10
    xor_mask=0b11
    for i in range(2**len(h)):
        cdiag=0
        for p in range(len(h)-1):
            if (i&(pm_mask<<p)==0) != (i&(mp_mask<<p)==0):
                cdiag-=J[p]/2
            cdiag+=J[p]/4
            cdiag+=((i&(1<<p))==0)*h[p]
            mat[(i,i^(1<<p))]+=g[p]/2
        mat[(i,i)]=cdiag+((i&(1<<(len(h)-1)))==0)*h[-1]-sum(h)/2
        mat[(i,i^(1<<(len(h)-1)))]=g[-1]/2
    return mat

def get_imbrie_p(h,g,J):
    mat=get_imbrie(h,g,J[:-1])
    L=h.shape[0]
    for v in range(2**L):
        cdiag=0
        if (v&1==0) != (v&(1<<(L-1))==0):
            cdiag-=J[-1]/2
        cdiag+=J[-1]/4
        mat[(v,v)]+=cdiag
    return mat
def get_imbrie_F_p(h,g,J,T):
    F0=np.diag(np.exp(np.diag(-0.5j*T*np.array(get_imbrie_p(h,np.zeros_like(g),J).todense()))))
    U1=scla.hadamard(2**len(h))
    # D1=np.diag((U1@get_imbrie_p(np.zeros_like(h),g,np.zeros_like(J)).todense()@U1.T)/2**len(h))
    D1=np.array(np.diag(get_imbrie_p(g,np.zeros_like(h),np.zeros_like(J)).todense()))
    F1=(U1.T@np.diag(np.exp(-0.5j*T*D1))@U1)/2**len(h)
    return F0@F1

def etat(g):
    return np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
def Jt(g):
    return -np.pi/4-np.log(np.tan(g))*0.5j
def gt(J):
    return np.arctan(1.0j*np.exp(2j*J))

def dualU(eta,J,g):
    gn=gt(J)
    Jn=Jt(g)
    etan=eta+etat(g)-etat(gn)
    return (etan,Jn,gn)
def get_F(N,eta,J,g,h=0.0):
    return get_imbrie_F_p(np.array([2*h]*N,dtype=complex),np.array([2*g]*N,dtype=complex),np.array([4*J]*N,dtype=complex),2.0)*np.exp(N*eta)
def apply_F_dual(N,F,sector,vec):
    v=np.zeros((2**(2*N),),dtype=complex)
    v[sector[2]]=vec
    v=v.reshape((2**N,2**N))
    v=F@v
    v=F.conj()@v.T
    v=v.T.reshape((2**(2*N),))
    return v[sector[2]]
def get_F_dual_op(T,eta,J,g,h=0.0,sec=None):
    if sec is None:
        sec=disorder_sector(T)
    F=get_F(T,*dualU(eta,J,g),h=h)
    return spla.LinearOperator((len(sec[2]),len(sec[2])),lambda v:apply_F_dual(T,F,sec,v))

def sff_avg(T,max_N,J,g):
    op=get_F_dual_op(T,0.0,J,g)
    tr=np.zeros((max_N,),dtype=complex)
    D=len(disorder_sector(T)[2])
    for j in range(D):
        vec=np.zeros((D,),dtype=float)
        vec[j]=1.0
        for i in range(max_N):
            vec=op@vec
            tr[i]+=vec[j]
    return tr

def lev_avg(T,J,g):
    op=get_F_dual_op(T,0.0,J,g)
    return spla.eigs(op,k=2)

def sff_sample(T,J,g,h):
    tr=0.0
    D=2**(2*T)
    for j in range(D):
        vec=np.zeros((D,),dtype=float)
        vec[j]=1.0
        for hc in h:
            op=get_F_dual_op(T,0.0,J,g,hc,trivial_sector(2*T))
            vec=op@vec
        tr+=vec[j]
    return tr
#sff_sample(3,J/8,g/4,h/4)
def lev_doc(doc):
    T=doc["T"]
    assert doc["model"]=="imbrie_floquet_h_avg"
    assert doc["boundary"]=="P"
    assert doc["algorithm"]=="dual_sff_ed"
    assert doc["drive"]["T"]==1.0
    assert doc["drive"]["part"]=="G"
    J=doc["J"]
    g=doc["G"]
    lev,levv=lev_avg(T,(J+1e-20)/8,(g+1e-20)/4)
    if np.abs(lev[0])>np.abs(lev[1]):
        sd={"lev1":lev[0],"lev2":lev[1],"levv1":levv[:,0],"levv2":levv[:,1]}
    else:
        sd={"lev1":lev[1],"lev2":lev[0],"levv1":levv[:,1],"levv2":levv[:,0]}
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        write_dict_to_hdfobj(f,"",sd)
def sff_doc(doc):
    T=doc["T"]
    Nm=doc["L_max"]
    assert doc["model"]=="imbrie_floquet_h_avg"
    assert doc["boundary"]=="P"
    assert doc["algorithm"]=="dual_sff_tr"
    assert doc["drive"]["T"]==1.0
    assert doc["drive"]["part"]=="G"
    J=doc["J"]
    g=doc["G"]
    lev,levv=lev_avg(T,J/8,g/4)
    sd={"sff":sff_avg(T,Nm,J/8,g/4)}
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        write_dict_to_hdfobj(f,"",sd)

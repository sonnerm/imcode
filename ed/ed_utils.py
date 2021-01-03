import numpy as np
import gmpy
import numpy.linalg as la
import scipy.sparse as sp
def get_product_state(sector):
    st=np.zeros((len(sector[2]),))
    num="10"*(sector[0]//2)+("1" if (sector[0]%2) else "")
    st[sector[1][int(num,2)]]=1.0
    return st

def trivial_sector(L):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**L):
        sec[i]=cn
        invsec.append(i)
        cn+=1
    return (L,sec,invsec)
def sz_sector(L,sz):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**L):
        if 2*gmpy.popcount(i)-L==sz:
            sec[i]=cn
            invsec.append(i)
            cn+=1
    return (L,sec,invsec)
def get_xxz(t,U,W,sector):
    t=t[::-1]
    U=U[::-1]
    W=W[::-1]
    print("t,U,W: %s %s %s"%(str(t),str(U),str(W)))
    mat=sp.dok_matrix((len(sector[2]),len(sector[2])))
    pm_mask=0b01
    mp_mask=0b10
    xor_mask=0b11
    for i,v in enumerate(sector[2]):
        cdiag=0
        for p in range(sector[0]-1):
            if (v&(pm_mask<<p)==0) != (v&(mp_mask<<p)==0):
                mat[(i,sector[1][v^(xor_mask<<p)])]=t[p]/2
                cdiag-=U[p]/2
            cdiag+=U[p]/4
            cdiag+=((v&(1<<p))==0)*W[p]
        mat[(i,i)]=cdiag+((v&(1<<(sector[0]-1)))==0)*W[sector[0]-1]-sum(W)/2
    return mat
def get_xxz_p(t,U,W,sector):
    mat=get_xxz(t[:-1],U[:-1],W,sector)
    for i,v in enumerate(sector[2]):
        cdiag=0
        if (v&1==0) != (v&(1<<(sector[0]-1))==0):
            mat[(i,sector[1][v^(1+(1<<(sector[0]-1)))])]=t[-1]/2
            cdiag-=U[-1]/2
        cdiag+=U[-1]/4
        mat[(i,i)]+=cdiag
    return mat

def get_imbrie(h,g,J):
    h=h[::-1]
    g=g[::-1]
    J=J[::-1]
    print("h,g,J: %s %s %s"%(str(h),str(g),str(J)))
    mat=sp.dok_matrix((2**len(h),2**len(h)))
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

def imbalance(sector):
    t=np.zeros((sector[0]-1,))
    U=np.zeros((sector[0]-1,))
    W=np.tile([1,-1],sector[0]//2+1)[:sector[0]]
    return get_xxz(t,U,W,sector)

def get_S(evecs,sector):
    return [get_entropy(e,sector)[sector[0]//2] for e in evecs]
def get_first_pair_dm(state,sector):
    full_state=np.zeros((2**sector[0],),dtype=complex)
    for i,s in enumerate(state):
        full_state[sector[2][i]]=s
    rho=full_state.reshape((4,2**(sector[0]-2)))
    rho=rho@rho.T.conj()
    return rho
def get_central_pair_dm(state,sector):
    vecs=[np.zeros((2**(sector[0]-2),),dtype=complex) for x in range(4)]
    L=sector[0]
    for i,s in enumerate(state):
        ps=sector[2][i]
        num=(ps&(3<<(L//2-1)))>>(L//2-1)
        ni1=(ps>>(L//2+1))<<(L//2-1)
        ni2=ps&((1<<(L//2-1))-1)
        vecs[num][ni1+ni2]+=s
    m=np.zeros((4,4),dtype=complex)
    for i in range(4):
        for j in range(4):
            m[i,j]=vecs[i].T.conj()@vecs[j]
    return m
def get_entropy(state,sector):
    full_state=np.zeros((2**sector[0],),dtype=complex)
    for i,s in enumerate(state):
        full_state[sector[2][i]]=s
    ent=[]
    for i in range(sector[0]+1):
        rho=full_state.reshape((2**i,2**(sector[0]-i)))
        if i>=sector[0]//2:
            rho=rho.T.conj()@rho
        else:
            rho=rho@rho.T.conj()
        ev=la.eigvalsh(rho)
        ent.append(-np.sum(np.maximum(ev,0.0)*np.log(np.minimum(np.maximum(ev,1e-32),1.0))))
    return ent
def get_r(evals):
    eigs=np.sort(np.real(np.log(evals)*(-1.0j)))
    eigd=np.diff(eigs)
    return np.mean(np.minimum(eigd[:-1],eigd[1:])/np.maximum(eigd[:-1],eigd[1:]))

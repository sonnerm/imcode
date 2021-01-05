import numpy as np
import gmpy
import matplotlib.pyplot as plt
import copy
import functools
import gmpy
import numpy.linalg as la
import scipy.sparse as sp
import scipy.linalg as scla
import scipy.sparse.linalg as spla
@functools.lru_cache(None)
def disorder_sector(L):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**(2*L)):
        if gmpy.popcount((i>>L)&(~(1<<(L-1))))==gmpy.popcount((i^((i>>L)<<L))&(~(1<<(L-1)))):
            sec[i]=cn
            invsec.append(i)
            cn+=1
    return (2*L,sec,invsec)
def get_random_J_keldysh_naive(T):
    ret=np.zeros((2**(2*T)))
    for F in range(T):
        invec=np.zeros((2**T))
        for i in range(2**(T-1)):
            if gmpy.popcount(i)==F:
                invec+=functools.reduce(np.kron,[np.array([2,0])]+[np.array([1,-1]) if b=="1" else np.array([1,1]) for b in bin(i+2**(T-1))[3:]])
        ret+=np.kron(invec,invec)
    return ret

def get_random_J_keldysh(T):
    ret=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        ret[i]=get_random_J_element(gmpy.popcount(i&((1<<(T-1))-1)),gmpy.popcount(i>>(T)&(~(1<<(T-1)))),T)
    return ret*get_keldysh_boundary(T)
def binom(a,b):
    if a<0 or b<0:
        return 0
    return int(gmpy.bincoef(a,b))
@functools.lru_cache(None)
def get_random_J_element(Nm,Np,T):
    def sub_random_J(N,F,T):
        return sum([(-1)**i*binom(N,i)*binom(T-1-N,F-i) for i in range(0,N+1)])
    return sum([sub_random_J(Nm,F,T)*sub_random_J(Np,F,T) for F in range(T)])

def project(M,sector):
    s1,s2=np.meshgrid(sector[2],sector[2])
    return M[s2,s1]
def trivial_sector(L):
    return (L,{i:i for i in range(2**L)},[i for i in range(2**L)])
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
def fwht_naive(a):
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
def fwht(a):
    h = 1
    slen=len(a)
    while h < slen:
        a=a.reshape((slen//h,h))
        a[::2,:],a[1::2,:]=a[::2,:]+a[1::2,:],a[::2,:]-a[1::2,:]
        a=a.reshape((slen,))
        h *= 2

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
def get_imbrie_diag_slower(h,J):
    h=h[::-1]
    J0=J[-1]
    J=J[-2::-1]
    print("h,J: %s %s"%(str(h),str(J)))
    mat=np.zeros((2**len(h)),dtype=complex)
    pm_mask=0b01
    mp_mask=0b10
    xor_mask=0b11
    L=len(h)
    for i in range(2**L):
        cdiag=0
        for p in range(L-1):
            if (i&(pm_mask<<p)==0) != (i&(mp_mask<<p)==0):
                cdiag-=J[p]/2
            cdiag+=J[p]/4
            cdiag+=((i&(1<<p))==0)*h[p]
        if (i&1==0) != (i&(1<<(L-1))==0):
            cdiag-=J0/2
        cdiag+=J0/4
        mat[i]=cdiag+((i&(1<<(L-1)))==0)*h[-1]-sum(h)/2
    return mat
def get_imbrie_diag(hs,Js):
    ret=np.zeros((2**len(hs)),dtype=complex)
    for i,h in enumerate(hs):
        cret=np.ones((2**i,2,2**(len(hs)-1-i)))/2
        cret[:,1,:]=-1/2
        ret+=np.ravel(cret*h)
    for i,J in enumerate(Js[:-1]):
        cret=np.ones((2**i,2,2,2**(len(Js)-2-i)))/4
        cret[:,1,0,:]=-1/4
        cret[:,0,1,:]=-1/4
        ret+=np.ravel(cret)*J
    cret=np.ones((2,2**(len(Js)-2),2))/4
    cret[1,:,0]=-1/4
    cret[0,:,1]=-1/4
    ret+=np.ravel(cret)*Js[-1]
    return ret

def etat(g):
    return np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
def Jt(g):
    return -np.pi/4-np.log(np.tan(g))*0.5j
def gt(J):
    return np.arctan(1.0j*np.exp(2j*J))
def dualU(J,g):
    gn=gt(J)
    Jn=Jt(g)
    return (Jn,gn,etat(g),etat(gn))

def embed(v,sec):
    if len(v.shape)==2 and v.shape[1]==1:
        v=v.reshape((v.shape[0],))
    ret=np.zeros((2**sec[0]),dtype=v.dtype)
    ret[sec[2]]=v
    return ret
def dense_kron(Ms):
    return functools.reduce(np.kron,Ms)
@functools.lru_cache(None)
def get_keldysh_boundary_naive(T):
    Pm=np.array([[1,1],[1,1]])
    Id=np.array([[1,0],[0,1]])
    pb=dense_kron([Pm]+[Id]*(T-1)+[Pm]+[Id]*(T-1))
    return np.diag((scla.hadamard((2**(2*T)))@pb@scla.hadamard((2**(2*T))))/(2**(2*T)))

# @functools.lru_cache(None)
def get_keldysh_boundary_slower(T):
    ret=np.ones((2**(2*T)))
    for i in range(2**(2*T)):
        if (i>>(2*T-1)) or ((i>>(T-1))&1):
            ret[i]=0
        else:
            ret[i]=4
    return ret
def get_keldysh_boundary(T):
    ret=np.zeros((2,2**(T-1),2,2**(T-1)))
    ret[0,:,0,:]=4
    return np.ravel(ret)


# def get_imbrie_F_p_diag(h,g,J,T):
#     D1=np.exp(-0.5j*T*np.array(get_imbrie_p(h,np.zeros_like(g),J).diagonal()))
#     D2=np.exp(-0.5j*T*np.array(get_imbrie_p(g,np.zeros_like(h),np.zeros_like(J)).diagonal()))
#     return D1,D2
def apply_F_dual(sec,T,D1,D2,v):
    v=embed(v,sec)
    fwht(v)
    v=D2*v/v.shape[0]
    fwht(v)
    v=D1*v
    return v[sec[2]]

def apply_F_dual_ad(sec,T,D1,D2,v):
    v=embed(v,sec)
    v=D1.conj()*v
    fwht(v)
    v=D2.conj()*v/v.shape[0]
    fwht(v)
    return v[sec[2]]

def get_F_dual_op(T,J,g,h,sec):
    Jt,gt,eta1,eta2=dualU(J,g)

    gt=np.array([0.0]+[2*gt.conj()]*(T-1)+[0.0]+[-2*gt]*(T-1))
    D2=np.exp(1.0j*get_imbrie_diag(gt,np.zeros_like(gt)))
    D2*=np.exp(-(2*T-2)*eta2.real)
    D2/=2
    D2*=get_keldysh_boundary(T)
    # print(gt,eta2,D2)
    # D2=np.ones_like(D2)/2

    h=np.array([0.0]+[2*h]*(T-1)+[0.0]+[-2*np.array(h).conj()]*(T-1))
    Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
    D1=np.exp(-1.0j*get_imbrie_diag(h,Jt))
    D1*=np.exp(2*T*eta1.real)
    # D1=np.ones_like(D1)
    # return D2
    return spla.LinearOperator((len(sec[2]),len(sec[2])),lambda v:apply_F_dual(sec,T,D1,D2,v),lambda v:apply_F_dual_ad(sec,T,D1,D2,v))

def get_F_dual(T,J,g,h,sec):

    Jt,_,eta1,_=dualU(J,g)
    h=np.array([0]+[2*h]*(T-1)+[0]+[-2*np.array(h).conj()]*(T-1))
    Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
    U1=np.diag(np.exp(-1.0j*np.array(get_imbrie_p(h,np.zeros_like(h),Jt).diagonal())))
    U1*=np.exp(eta1.real*(2*T))
    Pm=np.array([[1,1],[1,1]])
    # Pm=np.array([[1,0],[0,1]])
    Tm1=np.array([[np.exp(1.0j*J),np.exp(-1.0j*J)],[np.exp(-1.0j*J),np.exp(1.0j*J)]])
    Tm2=Tm1.conj()
    # Pm=np.eye(2)
    # Tm1=np.eye(2)
    # Tm2=np.eye(2)
    U2=dense_kron([Pm]+[Tm1]*(T-1)+[Pm]+[Tm2]*(T-1))/2
    ret=project(U1@U2,sec)
    print(np.trace(U2))
    # return U2
    return ret
# disorder_sector(4)[1][172]
# (np.log(np.trace(get_F_dual(T,0.1,0.4,0.0,trivial_sector(2*T))))-np.log(np.trace(get_F_dual(T-1,0.1,0.4,0.0,trivial_sector(2*(T-1))))))/(dualU(0.1,0.4))[2].real
def get_dense_sample(T,J,g,h):
    return get_F_dual(T,J,g,h,trivial_sector(2*T))

def get_dense_hr(T,J,g):
    return get_F_dual(T,J,g,0.0,disorder_sector(T))
def get_dense_Jr(T,g):
    pass
def get_op_sample(T,J,g,h):
    return get_F_dual_op(T,J,g,h,trivial_sector(2*T))
def get_op_hr(T,J,g):
    return get_F_dual_op(T,J,g,0.0,disorder_sector(T))

# np.angle(M1[1,0])+np.angle(M2[1,0]))
def calc_zz_direct(T,J,g,h):
    M=get_imbrie_F_p(h*2,np.array([g*2]*len(h)),np.array([J*4]*len(h)),2.0)
    op_z=np.array(get_imbrie_p(np.array([2.0]+[0.0]*(len(h)-1)),np.array([0.0]*len(h)),np.array([0.0]*len(h))).todense())
    return np.trace(op_z@la.matrix_power(M,T)@op_z@la.matrix_power(M.conj().T,T))/op_z.shape[0]
def calc_zz_transfer_sample(T,J,g,hs):
    Sz=np.array([[1,0],[0,-1]])
    Id=np.array([[1,0],[0,1]])
    M=dense_kron([Sz]+[Id]*(T-1)+[Sz]+[Id]*(T-1))
    # M=np.eye(M.shape[0])
    for h in hs:
        M=get_op_sample(T,J,g,h)@M
    return np.trace(M)

def calc_zz_transfer_hr(T,J,g,Nmax):
    Sz=np.array([[1,0],[0,-1]])
    Id=np.array([[1,0],[0,1]])
    M=dense_kron([Sz]+[Id]*(T-1)+[Sz]+[Id]*(T-1))
    for h in hs:
        M=get_dense_sample(T,J,g,h)@M
    return np.trace(M)

# Fo=get_op_sample(5,0.4*np.pi,0.2*np.pi,0.3*np.pi)@np.eye(1024)
# Fd=get_dense_sample(5,0.4*np.pi,0.2*np.pi,0.3*np.pi)
# F=np.loadtxt("../keldysh.dat",delimiter=",",dtype=complex)
# (np.angle(Fn)/np.pi).T
# np.angle(F)/np.pi
def lev_sample(T,J,g,h):
    op=get_op_sample(T,J,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T)))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=op@vec
    #     if np.isclose(nvec,vec).all():
    #         return (i,vec)
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op,k=1,which="LM"))

def rev_sample(T,J,g,h):
    op=get_op_sample(T,J,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T)))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=(op.adjoint()@vec.conj()).conj()
    #     if np.isclose(nvec,vec).all():
    #         return (i,np.ravel(vec))
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op.adjoint(),k=1,which="LM"))
# def lev_hr(T,J,g):
#     return spla.eigs(get_op_hr(T,J,g),k=1,which="LM")

def lev_hr_apply(T,J,g):
    op=get_op_hr(T,J,g)
    vec=np.random.random(len(disorder_sector(T)[2]))
    vec/=np.sqrt(np.dot(vec,vec))
    for i in range(2*T):
        vec=op@vec
        vec/=np.sqrt(np.dot(vec,vec))
    return vec
def lev_hr(T,J,g):
    op=get_op_hr(T,J,g)
    vec=np.random.random(len(disorder_sector(T)[2]))+1.0j*np.random.random(len(disorder_sector(T)[2]))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=op@vec
    #     if np.isclose(nvec,vec).all():
    #         return (i,vec)
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op,k=1,which="LM"))

# def lev_hr_full(T,J,g):
#     op=get_op_sample(T,J,g,0.0)
#     (i,vec)=lev_hr(T,J,g)
#     return (i,op@embed(vec,disorder_sector(T)))
# len(lev_hr_full(12,1.0,0.2)[1])

def rev_hr(T,J,g):
    op=get_op_hr(T,J,g)
    vec=np.random.random(len(disorder_sector(T)[2]))+1.0j*np.random.random(len(disorder_sector(T)[2]))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=(op.adjoint()@vec.conj()).conj()
    #     if np.isclose(nvec,vec).all():
    #         return (i,np.ravel(vec))
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op.adjoint(),k=1,which="LM"))

def get_F_Jr(T,g,h,sec):
    Jt,gt,eta1,eta2=dualU(1.0,g)

    # gt=np.array([0.0]+[2*gt]*(T-1)+[0.0]+[-2*gt.conj()]*(T-1))
    D2=get_random_J_keldysh(T)
    # D2*=np.exp(-(2*T-2)*eta2.real)
    D2/=2 #?

    h=np.array([0.0]+[2*h]*(T-1)+[0.0]+[-2*np.array(h).conj()]*(T-1))
    Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
    D1=np.exp(-1.0j*get_imbrie_diag(h,Jt))
    D1*=np.exp(2*T*eta1.real)
    # return D2
    return spla.LinearOperator((len(sec[2]),len(sec[2])),lambda v:apply_F_dual(sec,T,D1,D2,v),lambda v:apply_F_dual_ad(sec,T,D1,D2,v))
def get_op_Jr(T,g):
    return get_F_Jr(T,g,0.0,disorder_sector(T))

def rev_Jr(T,g):
    op=get_op_Jr(T,g)
    vec=np.random.random(len(disorder_sector(T)[2]))+1.0j*np.random.random(len(disorder_sector(T)[2]))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    for i in range(2*T):
        nvec=(op.adjoint()@vec.conj()).conj()
        if np.isclose(nvec,vec).all():
            return (i,np.ravel(vec))
        vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    print("Warning")
    return (-1,spla.eigs(op.adjoint(),k=1,which="LM"))
def lev_Jr(T,g):
    op=get_op_Jr(T,g)
    vec=np.random.random(len(disorder_sector(T)[2]))+1.0j*np.random.random(len(disorder_sector(T)[2]))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=op@vec
    #     if np.isclose(nvec,vec).all():
    #         return (i,vec)
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op,k=1,which="LM"))

def get_op_Jr_hf(T,g,h):
    return get_F_Jr(T,g,h,trivial_sector(2*T))

def rev_Jr_hf(T,g,h):
    op=get_op_Jr_hf(T,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T),))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=(op.adjoint()@vec.conj()).conj()
    #     if np.isclose(nvec,vec).all():
    #         return (i,np.ravel(vec))
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op.adjoint(),k=1,which="LM"))
def lev_Jr_hf(T,g,h):
    op=get_op_Jr_hf(T,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T),))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    # for i in range(2*T):
    #     nvec=op@vec
    #     if np.isclose(nvec,vec).all():
    #         return (i,vec)
    #     vec=nvec/np.sqrt(np.dot(nvec.conj(),nvec))
    # print("Warning")
    return (-1,spla.eigs(op,k=1,which="LM"))


# Ts,gs,hs,Js,lev,rev=[],[],[],[],[],[]
# for T in range(1,12):
#     for g in np.linspace(1e-20,np.pi/4,21):
#         for h in np.linspace(1e-20,np.pi/2,4)[1:]:
#             for J in np.linspace(1e-20,np.pi/4,4)[1:]:
#                 Ts.append(T)
#                 gs.append(g)
#                 hs.append(h)
#                 Js.append(J)
#                 lev.append(lev_sample(T,g,J,h))
#                 rev.append(rev_sample(T,g,J,h))
#                 print("T: %i g: %.2f h: %.2f J: %.2f"%(T,g/np.pi*4,h/np.pi*2,J/np.pi*2))
# import pickle
# print("test")
# pickle.dump((Ts,gs,hs,lev,rev),open("lrev_jr_hf.pickle","wb"))
# import h5py
# lev=np.array([l[1] for l in lev])
# rev=np.array([r[1] for r in rev])
# Ts=np.array(Ts)
# hs=np.array(hs)
# Js=np.array(Js)
# gs=np.array(gs)
# for T in range(12):
#     f=h5py.File("lrev_sample_n/lrev_sample_T_%i.h5"%T,"w")
#     f["Ts"]=Ts[Ts==T]
#     f["gs"]=gs[Ts==T]
#     f["hs"]=hs[Ts==T]
#     f["Js"]=Js[Ts==T]
#     f["lev"]=np.array(list(lev[Ts==T]))
#     f["rev"]=np.array(list(rev[Ts==T]))
#     f.close()

Ts,gs,lev,rev=[],[],[],[],[],[]
for T in range(1,13):
    # for J in np.linspace(np.pi/4*4/5,np.pi/4,21):
        for g in np.linspace(np.pi/4*4/5,np.pi/4,21):
            Ts.append(T)
            # Js.append(J)
            gs.append(g)
            lev.append(lev_hr(T,g,g))
            rev.append(rev_hr(T,g,g))
            print("T: %i g: %.2f J: %.2f g"%(T,g/np.pi*4,J/np.pi*4))


@functools.lru_cache(None)
def get_weights(g,T):
    ws=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        dm=count_dm(i,T)
        ws[i]=np.cos(g)**(2*T-dm)*np.sin(g)**dm*(-1)**(count_diff(i,T)//2)
    return ws

def count_dm(n,T):
    n=int(n)
    T=int(T)
    return gmpy.popcount(((n>>1)|((n&1)<<(2*T-1)))^n)
def count_diff(n,T):
    n=int(n)
    T=int(T)
    dms=((n>>1)|((n&1)<<(2*T-1)))^n
    mask=(1<<(2*T-1))|((1<<(T-1))-1)
    return gmpy.popcount(dms&mask)-gmpy.popcount(dms&(~mask))
# g=np.pi/4
# T=5
# Fm=get_F_dual_op(T,np.pi/4,g,0.2,trivial_sector(10))@np.eye(1024)
# vec=np.random.random(1024)
# vec=Fm@vec
# czz=-(np.sum(np.reshape(get_weights(g,T)*vec*vec.conj(),(2,16,2,16))[1,:,0,:])-np.sum(np.reshape(get_weights(g,T)*vec*vec.conj(),(2,16,2,16))[1,:,1,:]))/np.sum(get_weights(g,T)*vec*vec.conj())
# czz
# np.isclose(Fm@vec,vec).all()

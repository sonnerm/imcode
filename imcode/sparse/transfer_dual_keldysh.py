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

def embed(v,sec):
    if len(v.shape)==2 and v.shape[1]==1:
        v=v.reshape((v.shape[0],))
    ret=np.zeros((2**sec[0]),dtype=v.dtype)
    ret[sec[2]]=v
    return ret



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
    return (-1,spla.eigs(op,k=1,which="LM"))

def get_op_Jr_hf(T,g,h):
    return get_F_Jr(T,g,h,trivial_sector(2*T))

def rev_Jr_hf(T,g,h):
    op=get_op_Jr_hf(T,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T),))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    return (-1,spla.eigs(op.adjoint(),k=1,which="LM"))
def lev_Jr_hf(T,g,h):
    op=get_op_Jr_hf(T,g,h)
    vec=np.random.random((2**(2*T)))+1.0j*np.random.random((2**(2*T),))
    vec/=np.sqrt(np.dot(vec.conj(),vec))
    return (-1,spla.eigs(op,k=1,which="LM"))

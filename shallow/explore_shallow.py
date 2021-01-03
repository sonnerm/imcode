import numpy as np

import gmpy
import matplotlib.pyplot as plt
import transfer_dual_keldysh
import copy
import functools
import gmpy
import numpy.linalg as la
import scipy.sparse as sp
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import functools
import itertools
# prefix="/home/michael/floquet_mbl/shallow"
prefix="shallow"
# !mkdir shallow
def count_domvals(L,i):
    return gmpy.popcount((i>>1)^i)-(i>>(L-1))
def calculate_influence(J,g,h,state,rle=False):
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    cstate=np.ravel(np.eye(2**len(h)))
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    # if J.shape>1:
    #     Ujdiag=[functools.reduce(np.dot,x) for x in itertools.product([np.array([[np.exp(1.0j*Jc),np.exp(1.0j*J)],[0,0]]) for Jc in J[1:]])]
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    # print(Ujdiag)
    Tr=np.kron(W,W.conj())@np.diag(np.ravel(np.outer(Uhdiag,Uhdiag.conj())*np.outer(Ujdiag,Ujdiag.conj())))
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    if rle:
        mats={"b":la.eig(Tr@np.kron(Vu,Vd.conj())),"a":la.eig(Tr@np.kron(Vd,Vu.conj())),"+":la.eig(Tr@np.kron(Vu,Vu.conj())),"-":la.eig(Tr@np.kron(Vd,Vd.conj()))}
        mats={x[0]:(x[1][1],x[1][0],la.inv(x[1][1])) for x in mats.items()}
        acc=cstate
        for s in rle(state):
            acc=mats[s[0]][0]@np.diag(mats[s[0]][1]**s[1])@mats[s[0]][2]@acc
        return (cstate@acc)/2**h.shape[0]
    else:
        mats={"b":Tr@np.kron(Vu,Vd.conj()),"a":Tr@np.kron(Vd,Vu.conj()),"+":Tr@np.kron(Vu,Vu.conj()),"-":Tr@np.kron(Vd,Vd.conj())}
        acc=cstate
        for s in state:
            acc=mats[s]@acc
        return (cstate@acc)/2**h.shape[0]
def rle(s):
    cch=s[0]
    cco=0
    for c in s:
        if c==cch:
            cco+=1
        else:
            yield (cch,cco)
            cco=1
            cch=c
    yield (cch,cco)

# def get_number(t0,tt,fw,bw):
#     T=len(fw)+1
#     st=t0+fw+tt+bw[::-1]
#     return int(st,2)
# def calculate_influence_slow(J,g,h,state):
#     svec=np.ones((2**(2*len(state)+2)))
#     h=np.array(h)
#     if len(h.shape)==0:
#         h=h[np.newaxis]
#     for hc in h:
#         Jf=transfer_dual_keldysh.get_F_dual_op(len(state)+1,J,g,hc,transfer_dual_keldysh.trivial_sector(2*len(state)+2))
#         svec=Jf.adjoint()@svec
#
#     upper=""
#     lower=""
#     for u in state:
#         if u=="+" or u=="b":
#             upper+="1"
#         else:
#             upper+="0"
#         if u=="+" or u=="a":
#             lower+="1"
#         else:
#             lower+="0"
#     return svec[get_number("1","1",upper,lower)]
def calculate_loschmidt(J,g,h,ts,tm):
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    cstate=np.ravel(np.eye(2**len(h)))
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    # if J.shape>1:
    #     Ujdiag=[functools.reduce(np.dot,x) for x in itertools.product([np.array([[np.exp(1.0j*Jc),np.exp(1.0j*J)],[0,0]]) for Jc in J[1:]])]
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    # print(Ujdiag)
    Tr=np.kron(W,W.conj())@np.diag(np.ravel(np.outer(Uhdiag,Uhdiag.conj())*np.outer(Ujdiag,Ujdiag.conj())))
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    Bm=np.linalg.matrix_power(Tr@np.kron(Vu,Vd.conj()),ts)
    acc=cstate
    ret=[1.0]
    for _ in range(tm):
        acc=Bm@acc
        ret.append((cstate@acc)/2**h.shape[0])
    return ret
def calculate_loschmidt_hr(J,g,N,samples,ts,tm):
    sum=np.array(calculate_loschmidt(J,g,np.random.random(N)*2*np.pi,ts,tm))
    for _ in range(samples-1):
        sum+=np.array(calculate_loschmidt(J,g,np.random.random(N)*2*np.pi,ts,tm))
    return sum/samples



# t1=calculate_loschmidt_hr(0.6,0.6,2,100,1,100)
# t2=calculate_loschmidt_hr(0.6,0.6,4,100,1,100)

def verify():
    data=np.loadtxt("/home/michael/floquet_mbl/data_erg/ls_256.csv",dtype=complex)
    Ts=data[0,3:]
    data=data[1:].reshape((11,11,11,data.shape[1]))
    J,g,h=data[5,5,5][:3]
    ct=np.array(calculate_loschmidt(J,g,[h]*5,1,10))[2:]
    assert np.isclose(ct.conj(),data[5,5,5][4:13]).all()
# gs=np.linspace(0,np.pi/4,41,endpoint=True)
# plt.plot(calculate_loschmidt_hr(0.2,0.2,4,1,1,10000))
# samples=1000
# Tsind=[(1,100),(10,100),(100,100),(1000,100),(10000,100)]
# Ts=[np.nan,np.nan]+sum([list(range(0,Tsind[0][1]*Tsind[0][0]+Tsind[0][0],Tsind[0][0]))]+[list(range(ts[0]*11,ts[1]*ts[0]+ts[0],ts[0])) for ts in Tsind[1:]],[])

# for N in range(5,8):
#     print(N)
#     res=[np.array(Ts)]
#     for g in gs:
#         print(g)
#         cres=[g,g]+list(calculate_loschmidt_hr(g,g,N,samples,Tsind[0][0],Tsind[0][1]))
#         for ts in Tsind[1:]:
#             cres.extend(calculate_loschmidt_hr(g,g,N,samples,ts[0],ts[1])[11:])
#         res.append(np.array(cres))
#     np.savetxt("%s/ls_%i.csv"%(prefix,N),np.array(res),header="Loschmidt Echo")



def calculate_blipdist(J,g,h,tau,ts,tm):
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    cstate=np.ravel(np.eye(2**len(h)))
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    # if J.shape>1:
    #     Ujdiag=[functools.reduce(np.dot,x) for x in itertools.product([np.array([[np.exp(1.0j*Jc),np.exp(1.0j*J)],[0,0]]) for Jc in J[1:]])]
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    # print(Ujdiag)
    Tr=np.kron(W,W.conj())@np.diag(np.ravel(np.outer(Uhdiag,Uhdiag.conj())*np.outer(Ujdiag,Ujdiag.conj())))
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    Bm=Tr@np.kron(Vu,Vd.conj())
    Am=Tr@np.kron(Vd,Vu.conj())
    BBm=la.matrix_power(Bm,tau)
    AAm=la.matrix_power(Am,tau)

    Pm=Tr@np.kron(Vu,Vu.conj())
    Cm=(Pm+Tr@np.kron(Vd,Vd.conj()))/2
    CCm=la.matrix_power(Cm,ts)
    PPm=la.matrix_power(Pm,ts)
    left=cstate@BBm
    rc=AAm@cstate
    rp=np.copy(rc)
    retc,retp=[],[]
    for _ in range(tm):
        rc=CCm@rc
        rp=PPm@rp
        retc.append((left@rc)/2**h.shape[0])
        retp.append((left@rp)/2**h.shape[0])
    return retp,retc
def calculate_blipdist_hr(J,g,N,samples,tau,ts,tm):
    sum=np.array(calculate_blipdist(J,g,np.random.random(N)*2*np.pi,tau,ts,tm))
    for _ in range(samples-1):
        sum+=np.array(calculate_blipdist(J,g,np.random.random(N)*2*np.pi,tau,ts,tm))
    return sum/samples
# gs=np.linspace(0,np.pi/4,41,endpoint=True)
# samples=1000
# Tsind=[(1,100),(10,100),(100,100),(1000,100),(10000,100)]
# Ts=[np.nan,np.nan]+sum([list(range(0,Tsind[0][1]*Tsind[0][0]+Tsind[0][0],Tsind[0][0]))]+[list(range(ts[0]*11,ts[1]*ts[0]+ts[0],ts[0])) for ts in Tsind[1:]],[])
# tau=2
# for N in range(1,5):
#     print(N)
#     resc=[np.array(Ts)[1:]]
#     resp=[np.array(Ts)[1:]]
#     for g in gs:
#         print(g)
#         cp,cc=calculate_blipdist_hr(g,g,N,samples,tau,Tsind[0][0],Tsind[0][1])
#         cresc=[g,g]+list(cc)
#         cresp=[g,g]+list(cp)
#         for ts in Tsind[1:]:
#             cp,cc=calculate_blipdist_hr(g,g,N,samples,tau,ts[0],ts[1])
#             cresp.extend(cp[10:])
#             cresc.extend(cc[10:])
#         resc.append(np.array(cresc))
#         resp.append(np.array(cresp))
#     np.savetxt("%s/bdc2_%i.csv"%(prefix,N),np.array(resc),header="Blipdistance Average sojourn, tau=2 ")
#     np.savetxt("%s/bdp2_%i.csv"%(prefix,N),np.array(resp),header="Blipdistance Positive sojourn, tau=2 ")
def calculate_random_hr(J,g,N,samples,sampconf,Mf,Mb):
    pass

def calculate_periodic_hr(J,g,N,samples,Mf,Mb):
    pass

import numpy as np

# import explore_shallow as xs

import gmpy
import matplotlib.pyplot as plt
import transfer_dual_keldysh as tdk
import copy
import functools
import gmpy
import numpy.linalg as la
import scipy.sparse as sp
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import functools
import itertools
# !rsync -zhv fiteo:shallow/* /home/michael/floquet_mbl/data_mbl/shallow/
# prefix="/home/michael/floquet_mbl/shallow"
prefix="shallow"
# !mkdir shallow
def count_domvals(L,i):
    return gmpy.popcount((i>>1)^i)-(i>>(L-1))
def fold_state(fw,bw):
    dicti={("u","u"):"+",("u","d"):"b",("d","u"):"a",("d","d"):"-"}
    return "".join([dicti[cs] for cs in zip(fw,bw[::-1])])


def generate_random_conf_par(T,S,even):
    while True:
        fw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
        bw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
        cfw=np.array(["u"]+list(fw)+["u"])
        cbw=np.array(["u"]+list(bw)+["u"])
        if even and sum(cfw[1:]!=cfw[:-1])==sum(cbw[1:]!=cbw[:-1]):
            break
        elif not even and sum(cfw[1:]!=cfw[:-1])!=sum(cbw[1:]!=cbw[:-1]):
            break
    return fold_state(fw,bw)

def generate_random_conf(T,S):
    fw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
    bw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
    return fold_state(fw,bw)

def generate_random_conf_sym(T,S):
    fw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
    return fold_state(fw,fw)

def generate_random_conf_perm(T,S):
    fw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
    bw=np.hstack([fw[T//2:],fw[:T//2]])
    return fold_state(fw,bw[::-1])

def generate_random_conf_perm_sym(T,S):
    fw=np.random.permutation(list("u"*(S)+"d"*(T-1-S)))
    bw=np.hstack([fw[T//2:],fw[:T//2][::-1]])
    return fold_state(fw,bw[::-1])

def calculate_influence(J,g,h,state,rle=False):
    # print(state)
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    fw,bw=unfold_state(state)
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    # if J.shape>1:
    #     Ujdiag=[functools.reduce(np.dot,x) for x in itertools.product([np.array([[np.exp(1.0j*Jc),np.exp(1.0j*J)],[0,0]]) for Jc in J[1:]])]
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    # print(Ujdiag)
    Tr=W@np.diag(Uhdiag*Ujdiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    acc=np.eye(2**h.shape[0])
    for s in fw:
        acc=mats[s]@acc
    for s in bw:
        acc=mats[s].conj().T@acc
    return np.trace(acc)/2**h.shape[0]

# def rle(s):
#     cch=s[0]
#     cco=0
#     for c in s:
#         if c==cch:
#             cco+=1
#         else:
#             yield (cch,cco)
#             cco=1
#             cch=c
#     yield (cch,cco)

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
def calculate_loschmidt_slow(J,g,h,ts,tm):
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
    Tr=W@np.diag(Uhdiag*Ujdiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    Fw=Tr@Vu
    Bw=(Tr@Vd).conj().T
    left=np.eye(2**h.shape[0])
    right=np.eye(2**h.shape[0])
    ret=[1.0]
    Fw=la.matrix_power(Fw,ts)
    Bw=la.matrix_power(Bw,ts)
    for _ in range(tm):
        left=Fw@left
        right=Bw@right
        ret.append(np.trace(right@left)/2**h.shape[0])
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

# for N in range(6,8):
#     print(N)
#     res=[np.array(Ts)]
#     for g in gs:
#         print(g)
#         cres=[g,g]+list(calculate_loschmidt_hr(g,g,N,samples,Tsind[0][0],Tsind[0][1]))
#         for ts in Tsind[1:]:
#             cres.extend(calculate_loschmidt_hr(g,g,N,samples,ts[0],ts[1])[11:])
#         res.append(np.array(cres))
#     np.savetxt("%s/ls_%i.csv"%(prefix,N),np.array(res),header="Loschmidt Echo")



# def calculate_blipdist(J,g,h,tau,ts,tm):
#     h=np.array(h)
#     if len(h.shape)==0:
#         h=h[np.newaxis]
#     cstate=np.ravel(np.eye(2**len(h)))
#     W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
#     W=functools.reduce(np.kron,[W]*h.shape[0])
#     Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
#     # if J.shape>1:
#     #     Ujdiag=[functools.reduce(np.dot,x) for x in itertools.product([np.array([[np.exp(1.0j*Jc),np.exp(1.0j*J)],[0,0]]) for Jc in J[1:]])]
#     Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
#     # print(Ujdiag)
#     Tr=np.kron(W,W.conj())@np.diag(np.ravel(np.outer(Uhdiag,Uhdiag.conj())*np.outer(Ujdiag,Ujdiag.conj())))
#     Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
#     Vd=Vu.conj()
#     Bm=Tr@np.kron(Vu,Vd.conj())
#     Am=Tr@np.kron(Vd,Vu.conj())
#     BBm=la.matrix_power(Bm,tau)
#     AAm=la.matrix_power(Am,tau)
#
#     Pm=Tr@np.kron(Vu,Vu.conj())
#     Cm=(Pm+Tr@np.kron(Vd,Vd.conj()))/2
#     CCm=la.matrix_power(Cm,ts)
#     PPm=la.matrix_power(Pm,ts)
#     left=cstate@BBm
#     rc=AAm@cstate
#     rp=np.copy(rc)
#     retc,retp=[],[]
#     for _ in range(tm):
#         rc=CCm@rc
#         rp=PPm@rp
#         retc.append((left@rc)/2**h.shape[0])
#         retp.append((left@rp)/2**h.shape[0])
#     return retp,retc
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

def calculate_random_sample(J,g,h,S,T):
    return calculate_influence(J,g,h,generate_random_conf(T,S))

def calculate_random_sample_par(J,g,h,S,T,par):
    return calculate_influence(J,g,h,generate_random_conf_par(T,S,par))

def calculate_periodic_sample(J,g,h,S,d,ts,tm):
    if S>ts:
        raise ValueError("S<ts")
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    Tr=W@np.diag(Uhdiag*Ujdiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    Fw=la.matrix_power(Tr@Vu,S)@la.matrix_power(Tr@Vd,ts-S)
    Bw=la.matrix_power(Tr@Vd,d)@la.matrix_power(Tr@Vu,S)@la.matrix_power(Tr@Vd,ts-S-d)
    Bw=Bw.conj().T
    acc=np.eye(2**h.shape[0])
    ret=[1.0]
    left=np.eye(2**h.shape[0])
    right=np.eye(2**h.shape[0])
    for _ in range(tm):
        left=Fw@left
        right=Bw@right
        ret.append(np.trace(right@left)/2**h.shape[0])
    return ret
def calculate_quasiperiodic_sample(J,g,h,S,d,per,prob,T):
    if S>ts:
        raise ValueError("S<ts")
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    Uhdiag=np.ravel(functools.reduce(np.outer,[np.array([np.exp(1.0j*hc),np.exp(-1.0j*hc)]) for hc in h]))
    Ujdiag=np.array([np.exp(1.0j*J*(h.shape[0]-1-2*count_domvals(h.shape[0],i))) for i in range(2**h.shape[0])])
    Tr=W@np.diag(Uhdiag*Ujdiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*J),np.exp(-1.0j*J)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    Fu=Tr@Vu
    Fd=Tr@Vd
    Bu=Fu.conj().T
    Bd=Fd.conj().T
    Fw=la.matrix_power(Tr@Vu,S)@la.matrix_power(Tr@Vd,ts-S)
    Bw=la.matrix_power(Tr@Vd,d)@la.matrix_power(Tr@Vu,S)@la.matrix_power(Tr@Vd,ts-S-d)
    Bw=Bw.conj().T
    left=np.eye(2**h.shape[0])
    right=np.eye(2**h.shape[0])
    for c in fwstr:
        if c=="p":
            left=Fw@left
        elif c=="+":
            left=Fu@left
        else:
            left=Fd@left
    for c in bwstr:
        if c=="p":
            right=Bw@right
        elif c=="+":
            right=Bu@right
        else:
            right=Bd@right
    return np.trace(right@left)/2**h.shape[0]
# N=5
# ts=6
# s=1
#
# plt.plot(calculate_periodic_sample(0.1,0.1,np.random.random((N,))*2*np.pi,s,1,ts,1000))
# plt.plot([calculate_random_sample(0.1,0.1,np.random.random((N,))*2*np.pi,int(T),6*T) for T in range(1000)])


# N=1
# samples=4000
# bins=np.linspace(-1,1,401,endpoint=True)
# ret1=[]
# ret2=[]
# Ts=[]
# sds=[]
# gs=[]
# calculate_random_sample(0.2,0.2,0.2,10,20)
# N=6
# for T in np.arange(100,1100,100):
#     print("\t%i"%T)
#     for sd in np.arange(0.1,0.6,0.1):
#         print("%i"%int(10*sd),end="")
#         for g in np.linspace(0,np.pi/4,21,endpoint=True):
#             iret1=[calculate_random_sample_par(g,g,np.random.random((N,))*np.pi*2,int(sd*T),T,True) for _ in range(samples)]
#             iret2=[calculate_random_sample_par(g,g,np.random.random((N,))*np.pi*2,int(sd*T),T,False) for _ in range(samples)]
#             ret1.append(np.histogram(np.array(iret1).real,bins)[0])
#             ret2.append(np.histogram(np.array(iret2).real,bins)[0])
#             Ts.append(T)
#             sds.append(sd)
#             gs.append(g)
# rest1=np.hstack([np.array(Ts)[:,np.newaxis],np.array(sds)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret1)])
# rest2=np.hstack([np.array(Ts)[:,np.newaxis],np.array(sds)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret2)])
# np.savetxt("shallow/random_ev_N_%i.csv"%N,np.vstack([np.array([np.nan,np.nan]+list(bins)),rest1]))
# np.savetxt("shallow/random_od_N_%i.csv"%N,np.vstack([np.array([np.nan,np.nan]+list(bins)),rest2]))


# N=1
# samples=4000
# bins=np.linspace(-1,1,401,endpoint=True)
# ret=[]
# sds=[]
# gs=[]
# Ns=[]
# pers=[]
# Ts=[]
# for N in range(1,8):
#     print("N: %i"%N)
#     for sd in np.arange(0.1,0.6,0.1):
#         print("\t%i"%int(10*sd))
#         for g in np.linspace(0,np.pi/4,21,endpoint=True):
#             for per in np.arange(10,110,10):
#                 cTs=np.arange(0,1000,per)
#                 iret=np.array([calculate_periodic_sample(g,g,np.random.random((N,))*np.pi*2,int(sd*per),1,per,len(cTs)) for _ in range(samples)])
#                 for i,T in enumerate(cTs):
#                     ret.append(np.histogram(iret[:,i].real,bins)[0])
#                     Ts.append(T)
#                     sds.append(sd)
#                     pers.append(per)
#                     gs.append(g)
#                     Ns.append(N)
# g=0.11780972450961724
# ret=[]
# Ts=[]
# sds=[]
# gs=[]
# Ns=[]
# # for N in range(1,8):
# N=5
# print("N: %i"%N)
# for T in np.arange(1000,11000,1000):
#     print("\t%i"%T)
#     for sd in np.arange(0.1,0.6,0.1):
#         print("%i"%int(10*sd),end="")
#         for g in np.linspace(0,np.pi/4,21,endpoint=True):
#             iret=[calculate_random_sample(g,g,np.random.random((N,))*np.pi*2,int(sd*T),T) for _ in range(samples)]
#             ret.append(np.histogram(np.array(iret).real,bins)[0])
#             Ts.append(T)
#             sds.append(sd)
#             gs.append(g)
#             Ns.append(N)
# ret_e=[]
# ret_o=[]
# Ts=[]
# sd=0.2
# inner_samp=200
# outer_samp=1000
# gs=[]
# N=5
# bins=np.linspace(-1,1,401,endpoint=True)
#
# for T in np.arange(100,1100,100):
#     print("\t%i"%T)
#     S=int(sd*T)
#     for g in np.linspace(0,np.pi/4,21,endpoint=True):
#         iret_e=[]
#         iret_o=[]
#         for _ in range(outer_samp):
#             samp_e=generate_random_conf_par(T,S,True)
#             iret_e.append(np.mean([calculate_influence(g,g,np.random.random((N,))*np.pi*2,samp_e) for _ in range(inner_samp)]))
#             samp_o=generate_random_conf_par(T,S,True)
#             iret_o.append(np.mean([calculate_influence(g,g,np.random.random((N,))*np.pi*2,samp_o) for _ in range(inner_samp)]))
#         ret_e.append(np.histogram(np.array(iret_e).real,bins)[0])
#         ret_o.append(np.histogram(np.array(iret_o).real,bins)[0])
#         Ts.append(T)
#         gs.append(g)
#
# rest1=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_e)])
# rest2=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_o)])
# np.savetxt("shallow/randis_ev_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest1]))
# np.savetxt("shallow/randis_od_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest2]))

ret_p=[]
ret_ps=[]
Ts=[]
sd=0.2
inner_samp=200
outer_samp=1000
gs=[]
N=1
bins=np.linspace(-1,1,401,endpoint=True)

for T in np.arange(100,1100,100):
    print("\t%i"%T)
    S=int(sd*T)
    for g in np.linspace(0,np.pi/4,21,endpoint=True):
        iret_p,iret_ps=[],[]
        for _ in range(outer_samp):
            samp_p=generate_random_conf_perm(T,S)
            samp_ps=generate_random_conf_perm_sym(T,S)
            iret_p.append(np.mean([calculate_influence(g,g,np.random.random((N,))*np.pi*2,samp_p) for _ in range(inner_samp)]))
            iret_ps.append(np.mean([calculate_influence(g,g,np.random.random((N,))*np.pi*2,samp_ps) for _ in range(inner_samp)]))
        ret_p.append(np.histogram(np.array(iret_p).real,bins)[0])
        ret_ps.append(np.histogram(np.array(iret_ps).real,bins)[0])
        Ts.append(T)
        gs.append(g)

rest_p=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_p)])
np.savetxt("shallow/randis_pe_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_p]))
rest_ps=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_ps)])
np.savetxt("shallow/randis_ps_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_ps]))

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
# !rsync -zhv fiteo:shallow_jr/* /home/michael/floquet_mbl/data_mbl/shallow_jr/
# !mkdir /home/michael/floquet_mbl/data_mbl/shallow_jr
# prefix="/home/michael/floquet_mbl/shallow"
prefix="shallow"
# !mkdir shallow
def count_domvals(L,i):
    return gmpy.popcount((i>>1)^i)-(i>>(L-1))
def fold_state(fw,bw):
    dicti={("u","u"):"+",("u","d"):"b",("d","u"):"a",("d","d"):"-"}
    return "".join([dicti[cs] for cs in zip(fw,bw[::-1])])

def unfold_state(state):
    acf="".join(["u" if c=="b" or c=="+" else "d" for c in state])
    acb="".join(["u" if c=="a" or c=="+" else "d" for c in state])
    return acf,acb[::-1]

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
    J=np.array(J)
    if len(J.shape)==0:
        Jz=J
        J=np.array([J]*(h.shape[0]-1))
    else:
        Jz=J[0]
        J=J[1:]
    Udiag=np.exp(1.0j*tdk.get_imbrie_diag(h*2,list(4*J)+[0.0]))
    Tr=W@np.diag(Udiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*Jz),np.exp(-1.0j*Jz)],np.ones((2**(h.shape[0]-1))))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    acc=np.eye(2**h.shape[0])
    for s in fw:
        acc=mats[s]@acc
    for s in bw:
        acc=mats[s].conj().T@acc
    return np.trace(acc)/2**h.shape[0]


def calculate_loschmidt(J,g,h,ts,tm):
    h=np.array(h)
    if len(h.shape)==0:
        h=h[np.newaxis]
    cstate=np.ravel(np.eye(2**len(h)))
    W=np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]])
    W=functools.reduce(np.kron,[W]*h.shape[0])
    J=np.array(J)
    if len(J.shape)==0:
        Jz=J
        J=np.array([J]*(h.shape[0]-1))
    else:
        Jz=J[0]
        J=J[1:]
    Udiag=np.exp(1.0j*tdk.get_imbrie_diag(h*2,list(4*J)+[0.0]))
    Tr=W@np.diag(Udiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*Jz),np.exp(-1.0j*Jz)],np.ones((2**(h.shape[0]-1))))))
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


def verify():
    data=np.loadtxt("/home/michael/floquet_mbl/data_erg/ls_256.csv",dtype=complex)
    Ts=data[0,3:]
    data=data[1:].reshape((11,11,11,data.shape[1]))
    J,g,h=data[5,5,5][:3]
    ct=np.array(calculate_loschmidt(J,g,[h]*5,1,10))[2:]
    assert np.isclose(ct.conj(),data[5,5,5][4:13]).all()

def compute_perm(N,inner_samp=200,outer_samp=1000):
    ret_p=[]
    ret_ps=[]
    Ts=[]
    sd=0.2
    gs=[]
    bins=np.linspace(-1,1,401,endpoint=True)
    for T in np.arange(100,1100,100):
        print("\t%i"%T)
        S=int(sd*T)
        for g in np.linspace(0,np.pi/4,21,endpoint=True):
            iret_p,iret_ps=[],[]
            for _ in range(outer_samp):
                samp_p=generate_random_conf_perm(T,S)
                samp_ps=generate_random_conf_perm_sym(T,S)
                iret_p.append(np.mean([calculate_influence(np.random.random((N,))*np.pi*2,g,np.random.random((N,))*np.pi*2,samp_p) for _ in range(inner_samp)]))
                iret_ps.append(np.mean([calculate_influence(np.random.random((N,))*np.pi*2,g,np.random.random((N,))*np.pi*2,samp_ps) for _ in range(inner_samp)]))
            ret_p.append(np.histogram(np.array(iret_p).real,bins)[0])
            ret_ps.append(np.histogram(np.array(iret_ps).real,bins)[0])
            Ts.append(T)
            gs.append(g)
    rest_p=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_p)])
    np.savetxt("shallow_jr/jr_pe_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_p]))
    rest_ps=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_ps)])
    np.savetxt("shallow_jr/jr_ps_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_ps]))

def compute_sym(N,inner_samp=200,outer_samp=1000):
    ret_s=[]
    Ts=[]
    sd=0.2
    gs=[]
    bins=np.linspace(-1,1,401,endpoint=True)
    for T in np.arange(100,1100,100):
        print("\t%i"%T)
        S=int(sd*T)
        for g in np.linspace(0,np.pi/4,21,endpoint=True):
            iret_s=[]
            for _ in range(outer_samp):
                samp_s=generate_random_conf_sym(T,S)
                iret_s.append(np.mean([calculate_influence(np.random.random((N,))*np.pi*2,g,np.random.random((N,))*np.pi*2,samp_s) for _ in range(inner_samp)]))
            ret_s.append(np.histogram(np.array(iret_s).real,bins)[0])
            Ts.append(T)
            gs.append(g)
    rest_s=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_s)])
    np.savetxt("shallow_jr/jr_sy_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_s]))

def compute_rand(N,inner_samp=200,outer_samp=1000):
    ret_r=[]
    Ts=[]
    sd=0.2
    gs=[]
    bins=np.linspace(-1,1,401,endpoint=True)
    for T in np.arange(100,1100,100):
        print("\t%i"%T)
        S=int(sd*T)
        for g in np.linspace(0,np.pi/4,21,endpoint=True):
            iret_r=[]
            for _ in range(outer_samp):
                samp_r=generate_random_conf(T,S)
                iret_r.append(np.mean([calculate_influence(np.random.random((N,))*np.pi*2,g,np.random.random((N,))*np.pi*2,samp_r) for _ in range(inner_samp)]))
            ret_r.append(np.histogram(np.array(iret_r).real,bins)[0])
            Ts.append(T)
            gs.append(g)
    rest_r=np.hstack([np.array(Ts)[:,np.newaxis],np.array(gs)[:,np.newaxis],np.array(ret_r)])
    np.savetxt("shallow_jr/jr_rd_N_%i.csv"%N,np.vstack([np.array([np.nan]+list(bins)),rest_r]))

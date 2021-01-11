import numpy as np
from ..sparse import ising_diag
import functools

def im_element(Js,gs,hs,state):
    '''
        Calculate a single influence matrix element using the quantum echo
        representation
    '''
    if state[0]=="u" or state[0]=="d":#unfolded state
        fw,bw=state[1:len(state)//2],state[len(state)//2+1:]
    elif state[0]=="+" or state[0]=="-" or state[0]=="b" or state[0]=="a": #folded state
        fw,bw=unfold_state(state)
    else: # unfolded tuple
        fw,bw=state
    L=len(hs)
    gs=gs[::-1]
    hs=hs[::-1]
    Ws=[np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]]) for g in gs]
    W=functools.reduce(np.kron,Ws)
    Udiag=np.exp(1.0j*ising_diag(Js[:-1],gs))
    Tr=W@np.diag(Udiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*Js[-1]),np.exp(-1.0j*Js[-1])],np.ones((2**(L-1))))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    acc=np.eye(2**L)
    for s in fw:
        acc=mats[s]@acc
    for s in bw:
        acc=mats[s].conj().T@acc
    return np.trace(acc)/2**L

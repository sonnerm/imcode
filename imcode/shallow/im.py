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
    else:
        fw,bw=unfold_state(state)
    L=len(hs)
    Ws=[np.array([[np.cos(g),1.0j*np.sin(g)],[1.0j*np.sin(g),np.cos(g)]]) for g in gs]
    W=functools.reduce(np.kron,Ws)
    Udiag=ising_diag(Js[:-1],gs)
    Tr=W@np.diag(Udiag)
    Vu=np.diag(np.ravel(np.outer([np.exp(1.0j*Js[-1]),np.exp(-1.0j*Js[-1])],np.ones((2**L)))))
    Vd=Vu.conj()
    mats={"u":Tr@Vu,"d":Tr@Vd}
    acc=np.eye(2**L)
    for s in fw:
        acc=mats[s]@acc
    for s in bw:
        acc=mats[s].conj().T@acc
    return np.trace(acc)/2**L

from .utils import multiply_mpos,wrap_ndarray,BlipSite
from .ising import ising_W
from functools import lru_cache
from tenpy.networks.mpo import MPO
import numpy as np
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc
@lru_cache(None)
def Jr_operator(t):
    sites=[BlipSite() for t in range(t+1)]
    Iprim=np.ones((4,4))#/np.sqrt(2)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    inc=[wrap_ndarray(_get_jr_inc(c)) for c in range(t//2)]
    dec=[wrap_ndarray(_get_jr_dec(c)) for c in range(t//2,0,-1)]
    return MPO(sites,[Id]+inc+dec+[Id])

def _get_jr_inc(c):
    ret=np.zeros((2*c+1,2*c+3,4,4))
    for i in range(2*c+1):
        #No flip
        ret[i,i+1,0,0]=1.0
        ret[i,i+1,1,1]=1.0
        ret[i,i+1,2,2]=1.0
        ret[i,i+1,3,3]=1.0
        #flip on fw and bw
        ret[i,i+1,0,1]=1.0
        ret[i,i+1,1,0]=1.0
        ret[i,i+1,2,3]=1.0
        ret[i,i+1,3,2]=1.0
        #flip on fw
        ret[i,i+2,0,2]=1.0
        ret[i,i+2,1,3]=1.0
        ret[i,i+2,2,1]=1.0
        ret[i,i+2,3,0]=1.0
        #flip on bw
        ret[i,i,0,3]=1.0
        ret[i,i,1,2]=1.0
        ret[i,i,3,1]=1.0
        ret[i,i,2,0]=1.0
    return ret

def _get_jr_dec(c):
    ret=np.zeros((2*c+1,2*c-1,4,4))
    for i in range(2*c-1):
        ret[i+1,i,0,0]=1.0
        ret[i+1,i,1,1]=1.0
        ret[i+1,i,2,2]=1.0
        ret[i+1,i,3,3]=1.0
        #flip on fw and bw
        ret[i+1,i,0,1]=1.0
        ret[i+1,i,1,0]=1.0
        ret[i+1,i,2,3]=1.0
        ret[i+1,i,3,2]=1.0

        #flip on fw
        ret[i+2,i,0,2]=1.0
        ret[i+2,i,1,3]=1.0
        ret[i+2,i,2,1]=1.0
        ret[i+2,i,3,0]=1.0
        #flip on bw
        ret[i,i,0,3]=1.0
        ret[i,i,1,2]=1.0
        ret[i,i,3,1]=1.0
        ret[i,i,2,0]=1.0
    return ret

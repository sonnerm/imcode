from ..utils import multiply_mpos,wrap_ndarray
from .utils import FoldSite
from .ising import ising_W,ising_J
from functools import lru_cache
from tenpy.networks.mpo import MPO
import numpy as np
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc
@lru_cache(None)
def hr_operator(t):
    sites=[FoldSite() for t in range(t+1)]
    Iprim=np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    inc=[wrap_ndarray(_get_proj_inc(c)) for c in range((t-1)//2)]
    dec=[wrap_ndarray(_get_proj_dec(c)) for c in range((t-1)//2,0,-1)]
    if (t-1)%2:
        return MPO(sites,[Id]+inc+[wrap_ndarray(_get_proj_cen(t//2-1))]+dec+[Id])
    else:
        return MPO(sites,[Id]+inc+dec+[Id])
def _get_proj_cen(c):
    ret=np.zeros((2*c+1,2*c+1,4,4))
    for i in range(0,2*c+1):
        ret[i,i,0,0]=1.0
        ret[i,i,1,1]=1.0
        if i<2*c:
            ret[i,i+1,2,2]=1.0
        if i>0:
            ret[i,i-1,3,3]=1.0
    return ret
def _get_proj_inc(c):
    ret=np.zeros((2*c+1,2*c+3,4,4))
    for i in range(2*c+1):
        ret[i,i+1,0,0]=1.0
        ret[i,i+1,1,1]=1.0
        ret[i,i+2,2,2]=1.0
        ret[i,i,3,3]=1.0
    return ret

def _get_proj_dec(c):
    ret=np.zeros((2*c+1,2*c-1,4,4))
    for i in range(2*c-1):
        ret[i+1,i,0,0]=1.0
        ret[i+1,i,1,1]=1.0
        ret[i+2,i,3,3]=1.0
        ret[i,i,2,2]=1.0
    return ret
def ising_hr_T(t,J,g):
    return multiply_mpos([ising_J(t,J),ising_W(t,g),hr_operator(t)])

def ising_hr_Tp(t,J,g):
    return multiply_mpos([hr_operator(t),ising_J(t,J),ising_W(t,g)])

from .. import MPO
from .ising import ising_W,ising_J
from functools import lru_cache
import numpy as np
@lru_cache(None)
def hr_operator(t):
    Iprim=np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    inc=[_get_proj_inc(c) for c in range((t-1)//2)]
    dec=[_get_proj_dec(c) for c in range((t-1)//2,0,-1)]
    if (t-1)%2:
        return MPO.from_matrices([Ida]+inc+[wrap_ndarray(_get_proj_cen(t//2-1))]+dec+[Ida])
    else:
        return MPO.from_matrices([Ida]+inc+dec+[Ida])
def _get_proj_cen(c):
    ret=np.zeros((2*c+1,2*c+1,4,4))
    for i in range(0,2*c+1):
        ret[i,i,0,0]=1.0
        ret[i,i,2,2]=1.0
        if i<2*c:
            ret[i,i+1,1,1]=1.0
        if i>0:
            ret[i,i-1,2,2]=1.0
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
    return (ising_J(t,J)@ising_W(t,g)@hr_operator(t)).contract()

def ising_hr_Tp(t,J,g):
    return (hr_operator(t)@ising_J(t,J)@ising_W(t,g)).contract()

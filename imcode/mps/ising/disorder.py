import numpy as np
from ..import MPO
import functools
def hr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    weights=weights[:2*t+1]
    Ms=[_hr_inc(i) for i in range(1,t//2+1+t%2)]
    Ms[-1]=np.einsum("abcd,be->aecd",Ms[-1],_dis_center(t//2+t%2,t//2,weights))
    Ms+=[_hr_dec(i) for i in range(t//2,0,-1)]
    return MPO.from_matrices(Ms)
@functools.lru_cache(None)
def _hr_inc(i):
    ret=np.zeros((2*i-1,2*i+1,4,4))
    for k in range(2*i-1):
        ret[k,k+1,3,3]=1.0
        ret[k,k+1,0,0]=1.0
        ret[k,k,2,2]=1.0
        ret[k,k+2,1,1]=1.0
    return ret
@functools.lru_cache(None)
def _hr_dec(i):
    ret=np.zeros((2*i+1,2*i-1,4,4))
    for k in range(2*i-1):
        ret[k+1,k,3,3]=1.0
        ret[k+1,k,0,0]=1.0
        ret[k,k,2,2]=1.0
        ret[k+2,k,1,1]=1.0
    return ret

def _dis_center(i,j,weights):
    inds=list(range(0,len(weights),2))[::-1]+list(range(1,len(weights),2))
    weights=np.array(weights)[inds]
    ret=np.zeros((2*i+1,2*j+1),dtype=np.common_type(weights,np.array(1.0)))
    for k in range(2*i+1):
        for l in range(2*j+1):
            ret[k,l]=weights[k+l]
    return ret

def Jr_operator(t,weights=None):
    if weights is None:
        weights=[1.0]+[0.0,0.0]*t
    weights=weights[:2*t+1]
    Ms=[_Jr_inc(i) for i in range(1,t//2+1+t%2)]
    Ms[-1]=np.einsum("abcd,be->aecd",Ms[-1],_dis_center(t//2+t%2,t//2,weights))
    Ms+=[_Jr_dec(i) for i in range(t//2,0,-1)]
    return MPO.from_matrices(Ms)

@functools.lru_cache(None)
def _Jr_inc(i):
    ret=np.zeros((2*i-1,2*i+1,4,4))
    for k in range(2*i-1):
        ret[k,k+1,0,0]=1.0
        ret[k,k+1,1,1]=1.0
        ret[k,k+1,2,2]=1.0
        ret[k,k+1,3,3]=1.0
        ret[k,k+1,1,2]=1.0
        ret[k,k+1,2,1]=1.0
        ret[k,k+1,3,0]=1.0
        ret[k,k+1,0,3]=1.0

        ret[k,k,2,0]=1.0
        ret[k,k,0,2]=1.0
        ret[k,k,1,3]=1.0
        ret[k,k,3,1]=1.0

        ret[k,k+2,2,3]=1.0
        ret[k,k+2,3,2]=1.0
        ret[k,k+2,1,0]=1.0
        ret[k,k+2,0,1]=1.0
    return ret
@functools.lru_cache(None)
def _Jr_dec(i):
    ret=np.zeros((2*i+1,2*i-1,4,4))
    for k in range(2*i-1):
        ret[k+1,k,0,0]=1.0
        ret[k+1,k,1,1]=1.0
        ret[k+1,k,2,2]=1.0
        ret[k+1,k,3,3]=1.0
        ret[k+1,k,1,2]=1.0
        ret[k+1,k,2,1]=1.0
        ret[k+1,k,3,0]=1.0
        ret[k+1,k,0,3]=1.0

        ret[k,k,2,0]=1.0
        ret[k,k,0,2]=1.0
        ret[k,k,1,3]=1.0
        ret[k,k,3,1]=1.0

        ret[k+2,k,2,3]=1.0
        ret[k+2,k,3,2]=1.0
        ret[k+2,k,1,0]=1.0
        ret[k+2,k,0,1]=1.0
    return ret

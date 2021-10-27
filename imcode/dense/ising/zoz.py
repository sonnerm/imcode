from .. import kron,SX,SZ,ID,outer
from ..channel import unitary_channel
from functools import lru_cache
import numpy as np
import scipy.linalg as scla

def zoz_H(L,gates):
    if isinstance(gates,np.ndarray) and len(gates.shape)==4:
        gates=[gates]*(L-2)
    ret=np.zeros((2**L,2**L))
    for i,g in enumerate(gates):
        ret+=np.einsum("ab,cd,defg,gh,ij->abcdefghij",np.eye(2**i),np.eye(2),g,np.eye(2),np.eye(2**(L-3-i))).reshape((2**L,2**L))
    return ret


def zoz_F(L,gates,reversed=False):
    if isinstance(gates,np.ndarray) and len(gates.shape)==4:
        gates=[gates]*(L-2)
    ev=np.eye(2).reshape((1,2,1,2))
    for i,g in enumerate(gates[::2]):
        ev=np.einsum("abcd,befg,gh->abegcdfh",ev,g,np.eye(2)).reshape((2**(2*i),2,2**(2*i)))
    od=np.eye(4).reshape((2,2,2,2))
    for i,g in enumerate(gates[1::2]):
        od=np.einsum("abcd,befg,gh->abegcdfh",od,g,np.eye(2)).reshape((2**(2*i),2,2**(2*i)))
    if len(gates)%2==0:
        ev=np.kron(ev,np.eye(2))
    if reversed:
        return ev@od
    else:
        return od@ev

def zoz_T1(T,gates):
    pass
def zoz_T2(T,gates):
    pass
def zoz_L1(T,lch):
    pass
def zoz_L2(T,lch):
    pass

def pxp_H(L,alpha):
    return zoz_H(L,np.einsum("a,bc,d->abcd",np.array([alpha,0]),SX,np.array([1,0])))
def zxz_H(L,alpha):
    return zoz_H(L,np.einsum("a,bc,d->abcd",SZ*alpha,SX,SZ))
def pxp_F(L,alpha):
    return zoz_F(L,np.einsum("a,bc,d->abcd",np.array([alpha,0]),SX,np.array([1,0])))
def zxz_F(L,alpha):
    return zoz_F(L,np.einsum("a,bc,d->abcd",SZ*alpha,SX,SZ))

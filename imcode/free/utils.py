import numpy as np
import scipy.linalg as la
from ..dense import dense_kron
from ..dense import SX,SZ,SY,ID
def me(L,i):
    return dense_kron([SX]*i+[SZ]+[ID]*(L-i-1))
def mo(L,i):
    return dense_kron([SX]*i+[SY]+[ID]*(L-i-1))
def pe(L):
    sxp=dense_kron([SX]*L)
    return np.eye(2**L)/2-sxp/2
    # return ret/2+np.ones((2**L))/2
def po(L):
    sxp=dense_kron([SX]*L)
    return np.eye(2**L)/2+sxp/2

def _single_maj_to_quad(M):
    assert np.allclose(M,-M.T)
    L=M.shape[0]//2
    assert M.shape[0]%2==0
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret+=me(L,i)@me(L,j)*M[2*i,2*j]
            ret+=me(L,i)@mo(L,j)*M[2*i,2*j+1]
            ret+=mo(L,i)@me(L,j)*M[2*i+1,2*j]
            ret+=mo(L,i)@mo(L,j)*M[2*i+1,2*j+1]
    return ret/4

def maj_to_quad(maj):
    L=maj[0].shape[0]//2
    return _single_maj_to_quad(maj[0])@pe(L)+_single_maj_to_quad(maj[1])@po(L)

def _single_maj_to_trans(M):
    return la.expm(_single_maj_to_quad(la.logm(M)))
    # L=maj[0].shape[0]//2
    # return _single_maj_to_trans(maj[0])@pe(L)+_single_maj_to_trans(maj[1])@po(L)
def _single_quad_to_maj(M,L):
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j))/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j))/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j))/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j))/2**L
    ret-=np.diag(np.diag(ret))
    return ret*4#-np.diag(np.diag(ret))*(2*L-1)/(2*L)

def quad_to_maj(H):
    L=int(np.log2(H[0].shape[0]))
    return (_single_quad_to_maj(H@pe(L),L),_single_quad_to_maj(H@po(L),L))
def trans_to_maj(O):
    L=int(np.log2(O[0].shape[0]))
    He=la.logm(O@pe(L)+po(L))
    Ho=la.logm(O@po(L)+pe(L))
    return la.expm(_single_quad_to_maj(He,L)/2),la.expm(_single_quad_to_maj(Ho,L)/2)

def maj_to_trans(maj):
    return la.expm(maj_to_quad((la.logm(maj[0]),la.logm(maj[1])))*2)
def simple_to_maj(H):
    L=int(np.log2(H[0].shape[0]))
    Hi=la.inv(H)
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@H@me(L,j)@Hi)/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@H@me(L,j)@Hi)/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@H@mo(L,j)@Hi)/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@H@mo(L,j)@Hi)/2**L
    return ret

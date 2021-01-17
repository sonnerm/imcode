import numpy as np
from ..dense import dense_kron
from ..dense import SX,SZ,SY,ID
def me(L,i):
    return dense_kron([SX]*i+[SZ]+[ID]*(L-i-1))
def mo(L,i):
    return dense_kron([SX]*i+[SY]+[ID]*(L-i-1))
def pe(L):
    ret=np.ones((2**L,2**L))
    for i in range(L):
        ret=ret.reshape((2**i,2,2**(L-i-1)))
        ret[:,0,:]*=-1
    return ret
    # return ret/2+np.ones((2**L))/2
def po(L):
    ret=np.ones((2**L,2**L))
    for i in range(L):
        ret=ret.reshape((2**i,2,2**(L-i-1)))
        ret[:,0,:]*=-1
    return np.ravel(ret)
    # return np.ones((2**L))/2-ret/2
def _single_maj_to_quad(M):
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
    L=M.shape[0]//2
    assert M.shape[0]%2==0
    ret=np.eye(2**L,dtype=complex)
    for i in range(L):
        for j in range(L):
            ret=ret@me(L,i)@me(L,j)*M[2*i,2*j]
            ret=ret@me(L,i)@mo(L,j)*M[2*i,2*j+1]
            ret=ret@mo(L,i)@me(L,j)*M[2*i+1,2*j]
            ret=ret@mo(L,i)@mo(L,j)*M[2*i+1,2*j+1]
    return ret/4
def maj_to_trans(maj):
    L=maj[0].shape[0]//2
    return _single_maj_to_trans(maj[0])@pe(L)+_single_maj_to_quad(maj[1])@po(L)
def _manybody_to_maj(M,L):
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j))/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j))/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j))/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j))/2**L
    return ret*2#-np.diag(np.diag(ret))*(2*L-1)/(2*L)
def quad_to_maj(H):
    L=int(np.log2(H[0].shape[0]))
    return (_manybody_to_maj(H@pe(L),L),_manybody_to_maj(H@po(L),L))
def trans_to_maj(H):
    L=int(np.log2(H[0].shape[0]))
    return (_manybody_to_maj(H@pe(L),L),_manybody_to_maj(H@po(L),L))

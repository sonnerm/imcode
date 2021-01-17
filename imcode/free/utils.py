import numpy as np
from ..dense import dense_kron
def me(L,i):
    return dense_kron([sx]*i+[sz]+[id]*(L-i-1))
def mo(L,i):
    return dense_kron([sx]*i+[sy]+[id]*(L-i-1))
def pe(L):
    pass
def po(L):
    pass
def maj_to_quad(maj):
    He,Ho=maj
    L=He.shape[0]//2
    assert He.shape[0]%2==0
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret+=me(L,i)@me(L,j)*M[2*i,2*j]
            ret+=me(L,i)@mo(L,j)*M[2*i,2*j+1]
            ret+=mo(L,i)@me(L,j)*M[2*i+1,2*j]
            ret+=mo(L,i)@mo(L,j)*M[2*i+1,2*j+1]
    return ret/4
def maj_to_trans(maj):
    pass
def quad_to_maj(H):
    L=int(round(np.log(M.shape[0])/np.log(2)))
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j))/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j))/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j))/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j))/2**L
    return ret*2#-np.diag(np.diag(ret))*(2*L-1)/(2*L)
def trans_to_maj(U):

    L=int(round(np.log(M.shape[0])/np.log(2)))
    ret=np.zeros((2*L,2*L),dtype=complex)
    Mi=la.inv(M)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j)@Mi)/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j)@Mi)/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j)@Mi)/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j)@Mi)/2**L
    return ret

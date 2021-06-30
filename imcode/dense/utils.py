import numpy as np
import numpy.linalg as la
from functools import reduce
def kron(Ms):
    '''
        Calculates the dense kronecker product over a list of matrices
    '''
    return reduce(np.kron,Ms)

def outer(vs):
    '''
        Calculates the outer product over a list of vectors
    '''
    return reduce(np.outer,vs).ravel()
#Defining primitive pauli matrices
SY=np.array([[0,-1.0j],[1.0j,0]])
SM=np.array([[0,0],[1.0,0]])
SP=np.array([[0,1.0],[0,0]])
SX=np.array([[0,1.0],[1.0,0]])
SZ=np.array([[1.0,0],[0,-1.0]])
ID=np.array([[1.0,0],[0,1.0]])

def sx(L,i):
    return kron([ID]*i+[SX]+[ID]*(L-i-1))
def sz(L,i):
    return kron([ID]*i+[SZ]+[ID]*(L-i-1))
def sy(L,i):
    return kron([ID]*i+[SY]+[ID]*(L-i-1))


def sm(L,i):
    return kron([ID]*i+[SM]+[ID]*(L-i-1))
def sp(L,i):
    return kron([ID]*i+[SP]+[ID]*(L-i-1))
def one(L):
    return kron([ID]*L)

def normalize_im(v):
    return v/v[0] #TODO: good for now

def rdm(vec,sites):
    '''
        Calculates the reduced density matrix of the subsystem defined by ``sites``
    '''
    L=int(np.log2(len(vec)))
    complement=sorted(list(set(range(L))-set(sites)))
    vec=vec/np.sqrt(np.sum(vec.conj()*vec))
    vec=vec.reshape((2,)*L)
    vec=np.transpose(vec,sites+complement)
    vec=vec.reshape((2**len(sites),2**(len(complement))))
    ret=np.einsum("ij,kj->ik",vec.conj(),vec)
    return ret
def rdm_entropy(rdm):
    '''
        Calculates the entropy of a reduced density matrix
    '''
    ev=la.eigvalsh(rdm) #TODO: check
    return -np.sum(ev*np.log(np.clip(ev,1e-30,1.0)))

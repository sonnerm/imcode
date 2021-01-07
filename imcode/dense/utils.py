import numpy as np
import numpy.linalg as la
from functools import reduce
def dense_kron(Ms):
    '''
        Calculates the dense kronecker product over a list of matrices
    '''
    return reduce(np.kron,Ms)
#Defining primitive pauli matrices
SY=np.array([[0,-1.0j],[1.0j,0]])
SM=np.array([[0,0],[1.0,0]])
SP=np.array([[0,1.0],[0,0]])
SX=np.array([[0,1.0],[1.0,0]])
SZ=np.array([[1.0,0],[0,-1.0]])
ID=np.array([[1.0,0],[0,1.0]])

def sx(L,i):
    return dense_kron([ID]*i+[SX]+[ID]*(L-i-1))
def sz(L,i):
    return dense_kron([ID]*i+[SZ]+[ID]*(L-i-1))
def sy(L,i):
    return dense_kron([ID]*i+[SY]+[ID]*(L-i-1))


def sm(L,i):
    return dense_kron([ID]*i+[SM]+[ID]*(L-i-1))
def sp(L,i):
    return dense_kron([ID]*i+[SP]+[ID]*(L-i-1))
def one(L):
    return dense_kron([ID]*L)

def normalize_im(v):
    return v/v[0] #TODO: good for now

def disorder_sector(L):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**(2*L)):
        if gmpy.popcount((i>>L)&(~(1<<(L-1))))==gmpy.popcount((i^((i>>L)<<L))&(~(1<<(L-1)))):
            sec[i]=cn
            invsec.append(i)
            cn+=1
    return (2*L,sec,invsec)
def reduced_density_matrix(sites,vec):
    '''
        Calculates the reduced density matrix of the subsystem defined by ``sites``
    '''
    L=int(np.log2(len(vec)))
    complement=sorted(list(set(range(L))-set(sites)))
    vec=vec.reshape((2,)*L)
    vec=np.transpose(vec,sites+complement)
    return vec@vec.T #TODO: check
def rdm_entropy(rdm):
    '''
        Calculates the entropy of a reduced density matrix
    '''
    ev=la.eigvalsh(rdm) #TODO: check
    return np.sum(ev*np.log(ev+1e-30))

from .. import kron,SX,SZ,ID,outer
from functools import lru_cache
import numpy as np
import scipy.linalg as scla

def ising_H(L,J,g,h):
    r'''
        Construct a dense Hamiltonian of a spin 1/2 Ising ring with parameters given by the arrays J,g,h.
        H=\sum_i J_i s^z_{i+1}s^z_{i} + \sum_i h_i s^z_i + \sum_i g_i s^x_i
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    ret=np.zeros((2**L,2**L),dtype=np.common_type(J,g,h,np.array(1.0)))
    for i,Jv in enumerate(J[:L-1]):
        ret+=Jv*kron([ID]*i+[SZ]+[SZ]+[ID]*(L-i-2))
    if len(J)==L and L>1:
        ret+=J[-1]*kron([SZ]+[ID]*(L-2)+[SZ])
    for i,hv in enumerate(h):
        ret+=hv*kron([ID]*i+[SZ]+[ID]*(L-i-1))
    for i,gv in enumerate(g):
        ret+=gv*kron([ID]*i+[SX]+[ID]*(L-i-1))
    return ret
def ising_F(L,J,g,h):
    r'''
        Constructs a dense one period time evolution operator for the kicked ising chain model
        F = \exp(i\sum_i J_i s_i^zs_{i+1}^z + h_i s_i^z)\exp(i \sum_i g_i s^x_i)
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    return scla.expm(1.0j*ising_H(L,J,[0.0]*L,h))@scla.expm(1.0j*ising_H(L,[0.0]*L,g,[0.0]*L))


def ising_J(T,J):
    J=np.array(J)
    K=-J.conj()
    Tm=np.array([[+J+K,+J-K,-J+K,-J-K],[+J-K,+J+K,-J-K,-J+K],[-J+K,-J-K,+J+K,+J-K],[-J-K,-J+K,+J-K,+J+K]])
    Tm=np.exp(1.0j*Tm)
    return kron([Tm for _ in range(T)])

def ising_h(T,h):
    h=np.array(h)
    k=-h.conj()
    elem=np.diag(np.exp(1.0j*np.array([h+k,h-k,-h+k,-h-k])))
    return kron([elem for _ in range(T)])
def ising_W(T,g,init=np.eye(2)/2,final=np.eye(2)):
    gate=np.cos(g)*ID+1.0j*np.sin(g)*SX
    ret=np.einsum("ab,bc,dc->ad",gate,init,gate.conj())
    gate=np.einsum("ab,cd->acbd",gate,gate.conj()).reshape((4,4))
    for i in range(T-1):
        ret=ret.reshape((4**i,4))
        ret=np.einsum("ab,bc->abc",ret,gate)
    ret=np.einsum("ab,b->ab",ret.reshape(4**(T-1),4),final.ravel())
    return np.diag(np.ravel(ret))

def ising_T(T,J,g,h,init=np.eye(2)/2,final=np.eye(2)):
    r'''
        Calculate a dense IM approach transfer matrix for the kicked ising
        chain. See arXiv:2009.10105 for details. The sites are ordered according
        to the folded picture.
    '''
    U1=ising_h(T,h)*ising_W(T,g,init,final)
    U2=ising_J(T,J)
    return U2@U1


def ising_SFF(T,J,g,h):
    pass
def ising_hr_SFF(T,J,g):
    pass

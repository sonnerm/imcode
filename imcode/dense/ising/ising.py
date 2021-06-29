from .. import dense_kron,SX,SZ,ID
from functools import lru_cache
import numpy as np
import scipy.linalg as scla

def ising_H(J,g,h):
    r'''
        Construct a dense Hamiltonian of a spin 1/2 Ising ring with parameters given by the arrays J,g,h.
        H=\sum_i J_i s^z_{i+1}s^z_{i} + \sum_i h_i s^z_i + \sum_i g_i s^x_i
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    L=len(h) # maybe change to explicit length?
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    ret=np.zeros((2**L,2**L),dtype=np.common_type(J,g,h,np.array(1.0)))
    for i,Jv in enumerate(J[:L-1]):
        ret+=Jv*dense_kron([ID]*i+[SZ]+[SZ]+[ID]*(L-i-2))
    if len(J)==L and L>1:
        ret+=J[-1]*dense_kron([SZ]+[ID]*(L-2)+[SZ])
    for i,hv in enumerate(h):
        ret+=hv*dense_kron([ID]*i+[SZ]+[ID]*(L-i-1))
    for i,gv in enumerate(g):
        ret+=gv*dense_kron([ID]*i+[SX]+[ID]*(L-i-1))
    return ret
def ising_F(J,g,h):
    r'''
        Constructs a dense one period time evolution operator for the kicked ising chain model
        F = \exp(i\sum_i J_i s_i^zs_{i+1}^z + h_i s_i^z)\exp(i \sum_i g_i s^x_i)
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    L=len(h) #maybe change to explicit length
    return scla.expm(1.0j*ising_H(J,[0.0]*L,h))@scla.expm(1.0j*ising_H([0.0]*L,g,[0.0]*L))


def ising_J(T,J):
    Pm=np.array([[1,1],[1,1]])#/np.sqrt(2)
    Tm1=np.array([[np.exp(1.0j*J),np.exp(-1.0j*np.conj(J))],[np.exp(-1.0j*np.conj(J)),np.exp(1.0j*J)]])
    Tm2=Tm1.conj()
    return dense_kron([Pm]+[Tm1]*(T-1)+[Pm]+[Tm2]*(T-1))

def ising_h(T,h):
    h=np.array([0]+[h]*(T-1)+[0]+[-np.array(h).conj()]*(T-1))
    U1=np.diag(np.exp(1.0j*np.array(ising_H(np.zeros_like(h),np.zeros_like(h),h).diagonal())))
    return U1

def ising_W(T,g,init=(0.5,0.5)):
    ret=np.ones((2**(2*T)),dtype=complex)
    for i in range(T):
        ret=ret.reshape((2**i,2,2,2**(2*T-2-i)))
        ret[:,1,1,:]*=np.cos(g)
        ret[:,0,0,:]*=np.cos(g)
        ret[:,1,0,:]*=np.sin(g)*1.0j
        ret[:,0,1,:]*=np.sin(g)*1.0j

    for i in range(T-1):
        ret=ret.reshape((2**(T+i),2,2,2**(T-2-i)))
        ret[:,1,1,:]*=np.conj(np.cos(g))
        ret[:,0,0,:]*=np.conj(np.cos(g))
        ret[:,1,0,:]*=np.conj(np.sin(g)*1.0j)
        ret[:,0,1,:]*=np.conj(np.sin(g)*1.0j)

    ret=ret.reshape((2,2**(2*T-2),2))
    ret[1,:,1]*=np.conj(np.cos(g))*init[1]
    ret[0,:,0]*=np.conj(np.cos(g))*init[0]
    ret[1,:,0]*=np.conj(np.sin(g)*1.0j)*init[1]
    ret[0,:,1]*=np.conj(np.sin(g)*1.0j)*init[0]
    return np.diag(np.ravel(ret))

def ising_T(T,J,g,h,init=(0.5,0.5)):
    r'''
        Calculate a dense IM approach transfer matrix for the kicked ising
        chain. See arXiv:2009.10105 for details. The sites are ordered as
        follows: [s_0 forward s_T backward].
    '''
    U1=ising_h(T,h)*ising_W(T,g,init)
    U2=ising_J(T,J)
    return U2@U1


def ising_SFF(T,J,g,h):
    pass
def ising_hr_SFF(T,J,g,h):
    pass

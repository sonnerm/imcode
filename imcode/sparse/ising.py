import numpy as np
from functools import lru_cache
from .utils import fwht,DiagonalLinearOperator,SxDiagonalLinearOperator
from ..utils import popcount
import scipy.sparse.linalg as spla

def ising_diag(J,h):
    L=len(h)
    ret=np.zeros((2**L),dtype=np.common_type(J,h))
    for i,hv in enumerate(h):
        cret=np.ones((2**i,2,2**(L-1-i)))
        cret[:,1,:]=-1
        ret+=np.ravel(cret*hv)
    for i,Jv in enumerate(J[:L-1]):
        cret=np.ones((2**i,2,2,2**(L-2-i)))
        cret[:,1,0,:]=-1
        cret[:,0,1,:]=-1
        ret+=np.ravel(cret)*Jv
    if len(J)==L:
        cret=np.ones((2,2**(L-2),2))
        cret[1,:,0]=-1
        cret[0,:,1]=-1
        ret+=np.ravel(cret)*J[-1]
    return ret
def ising_H(J,g,h):
    '''
    '''
    opsz=DiagonalLinearOperator(ising_diag(J,h))
    opsx=SxDiagonalLinearOperator(ising_diag(np.zeros_like(J),g))
    return opsz+opsx
def ising_F(J,g,h):
    opsz=DiagonalLinearOperator(np.exp(1.0j*ising_diag(J,h)))
    opsx=SxDiagonalLinearOperator(np.exp(1.0j*ising_diag(np.zeros_like(J),g)))
    return opsz@opsx

def ising_W(T,g):
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
    ret[1,:,1]*=np.conj(np.cos(g))
    ret[0,:,0]*=np.conj(np.cos(g))
    ret[1,:,0]*=np.conj(np.sin(g)*1.0j)
    ret[0,:,1]*=np.conj(np.sin(g)*1.0j)
    return DiagonalLinearOperator(np.ravel(ret))
def ising_J(T,J):
    D=np.ones(2**(2*T),dtype=complex)
    for i in range(1,T):
        D=D.reshape((2**i,2,2**(2*T-i-1)))
        D[:,1,:]*=np.exp(1.0j*J)-np.exp(-1.0j*np.conj(J))
        D[:,0,:]*=np.exp(1.0j*J)+np.exp(-1.0j*np.conj(J))
        D=D.reshape((2**(T+i),2,2**(T-i-1)))
        D[:,1,:]*=-np.exp(1.0j*J)+np.exp(-1.0j*np.conj(J))
        D[:,0,:]*=np.exp(1.0j*J)+np.exp(-1.0j*np.conj(J))
    D=D.reshape((2,2**(2*T-1)))
    D[1,:]*=0
    D[0,:]*=2
    D=D.reshape((2**(T),2,2**(T-1)))
    D[:,1,:]*=0
    return SxDiagonalLinearOperator(np.ravel(D))
def ising_h(T,h):
    h=np.array([0.0]+[h]*(T-1)+[0.0]+[-np.array(h).conj()]*(T-1))
    D1=np.exp(1.0j*ising_diag(np.zeros_like(h),h))
    return DiagonalLinearOperator(D1)

def ising_T(t,J,g,h):
    '''
    '''
    U1=DiagonalLinearOperator(ising_h(t,h).diag*ising_W(t,g).diag)#slight optimization
    U2=ising_J(t,J)
    return U2@U1
@lru_cache(None)
def _hr_diagonal(T):
    ret=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        if popcount((i>>T)&(~(1<<(T-1))))==popcount((i^((i>>T)<<T))&(~(1<<(T-1)))):
            ret[i]=1
    return ret
def hr_operator(T):
    return DiagonalLinearOperator(_hr_diagonal(T))
@lru_cache(None)
def Jr_operator(T):
    # D=np.ones(2**(2*T))
    # for i in range(1,T):
    #     D=D.reshape((2**i,2,2**(2*T-i-1)))
    #     D[:,0,:]*=-1
    #     D=D.reshape((2**(T+i),2,2**(T-i-1)))
    #     D[:,0,:]*=-1
    # D[D<0]=0
    # for i in range(1,T):
    #     D=D.reshape((2**i,2,2**(2*T-i-1)))
    #     D[:,1,:]*=-1
    #     D=D.reshape((2**(T+i),2,2**(T-i-1)))
    #     D[:,1,:]*=-1
    # D[D<0]=0
    # for i in range(1,T):
    #     D=D.reshape((2**i,2,2**(2*T-i-1)))
    #     D[:,1,:]*=1
    #     D=D.reshape((2**(T+i),2,2**(T-i-1)))
    #     D[:,1,:]*=-1
    # D=D.reshape((2,2**(2*T-1)))
    # D[1,:]*=0
    # D[0,:]*=2
    # D=D.reshape((2**(T),2,2**(T-1)))
    # D[:,1,:]*=0
    # return SxDiagonalLinearOperator(np.ravel(D))
    return None

def ising_hr_T(T,J,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism described in arXiv:2012.00777. The averaging
        is performed over parameter h. Site ordering as in ising_T.
    '''
    U1=DiagonalLinearOperator(hr_operator(T).diag*ising_W(T,g).diag)
    U2=ising_J(T,J)
    return U2@U1

def ising_hr_Tp(T,J,g):
    U1=ising_W(T,g)
    U2=ising_J(T,J)
    Up=hr_operator(T)
    return Up@U2@U1
def ising_Jr_T(T,g,h):

    '''
        Calculate a dense spatial transfer matrix for the J disorder averaged
        influence matrix formalism similar to arXiv:2012.00777
        Site ordering as in ising_T.
    '''
    U1=DiagonalLinearOperator(ising_h(T,h).diag*ising_W(T,g).diag)
    U2=Jr_operator(T)
    return U2@U1
#TODO add new ref if available
def ising_Jhr_T(T,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism with averaging over both J and h.
        Site ordering as in ising_T.
    '''
    #TODO add new ref if available
    U1=DiagonalLinearOperator(hr_operator(T).diag*ising_W(T,g).diag)
    U2=Jr_operator(T)
    return U2@U1
def ising_Jhr_Tp(T,g):
    U1=ising_W(T,g)
    U2=Jr_operator(T)
    Up=hr_operator(T)
    return Up@U2@U1

def ising_SFF(T,J,g,h):
    pass

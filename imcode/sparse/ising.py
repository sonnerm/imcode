import numpy as np
from functools import lru_cache
from scipy.sparse.linalg import LinearOperator
from .utils import fwht
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
class SxDiagonalLinearOperator(LinearOperator):
    '''
        Linear operator
    '''
    def __init__(self, D):
        self.diag=D
        shape=(D.shape[0],D.shape[0])
        dtype=D.dtype
        super().__init__(dtype,shape)
    def _matvec(self,vec):
        v=np.array(np.ravel(vec),dtype=np.common_type(vec,self.diag))
        fwht(v)
        v*=self.diag/v.shape[0]
        fwht(v)
        return v
    def _adjoint(self):
        return SxDiagonalLinearOperator(self.diag.conj())

class DiagonalLinearOperator(LinearOperator):
    '''
        Linear operator
    '''
    def __init__(self, D):
        self.diag=D
        shape=(D.shape[0],D.shape[0])
        dtype=D.dtype
        super().__init__(dtype,shape)
    def _matvec(self,vec):
        return np.ravel(vec)*self.diag
    def _adjoint(self):
        return DiagonalLinearOperator(self.diag.conj())
def ising_H(J,g,h):
    '''
    '''
    opsz=DiagonalLinearOperator(ising_diag(J,h))
    opsx=SxDiagonalLinearOperator(ising_diag(np.zeros_like(J),h))
    return opsz+opsx
def ising_F(J,g,h):
    opsz=DiagonalLinearOperator(np.exp(1.0j*ising_diag(J,h)))
    opsx=SxDiagonalLinearOperator(np.exp(1.0j*ising_diag(np.zeros_like(J),h)))
    return opsz@opsx

def ising_W(T,g):
    Jt=-np.pi/4-np.log(np.tan(g))*0.5j
    eta=np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
    Jt=np.array([Jt]*(T)+[-Jt.conj()]*T)
    D1=np.exp(1.0j*ising_diag(Jt,np.zeros_like(Jt)))
    D1*=np.exp(2*T*eta.real)
    return DiagonalLinearOperator(D1)
def ising_J(T,J):
    gt=np.array([0.0]+[2*gt.conj()]*(T-1)+[0.0]+[-2*gt]*(T-1))
    D2=np.exp(1.0j*ising_diag(gt,np.zeros_like(gt)))
    D2*=np.exp(-(2*T-2)*eta2.real)
    D2/=2
    D2*=get_keldysh_boundary(T)
def ising_h(T,h):
    h=np.array([0.0]+[h]*(T-1)+[0.0]+[-np.array(h).conj()]*(T-1))
    D1=np.exp(1.0j*ising_diag(np.zeros_like(h),h))
    return DiagonalLinearOperator(D1)
def _get_weights(g,T):
    ws=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        dm=count_dm(i,T)
        ws[i]=np.cos(g)**(2*T-dm)*np.sin(g)**dm*(-1)**(count_diff(i,T)//2)
    return ws
def _get_keldysh_boundary(T):
    ret=np.zeros((2,2**(T-1),2,2**(T-1)))
    ret[0,:,0,:]=4
    return np.ravel(ret)

def ising_T(J,g,h):
    '''
    '''
    U1=DiagonalLinearOperator(ising_h(T,h).diag*ising_W(T,g).diag)#slight optimization
    U2=ising_J(T,J)
    return U2@U1
@lru_cache(None)
def _hr_diagonal(T):
    ret=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        if popcount((i>>T)&(~(1<<(T-1))))==popcount((i^((i>>T)<<T))&(~(1<<(T-1)))):
            ret=1
    return ret
def hr_operator(T):
    return DiagonalLinearOperator(_hr_diagonal)
def Jr_operator(T):
    return SxDiagonalLinearOperator(_hr_diagonal)#TODO: Fingers crossed

def ising_hr_T(T,J,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism described in arXiv:2012.00777. The averaging
        is performed over parameter h. Site ordering as in ising_T.
    '''
    U1=DiagonalLinearOperator(ising_hr(T).diag*ising_W(T,g).diag)
    U2=ising_J(T,J)
    return U2@U1
def ising_Jr_T(T,g,h):

    '''
        Calculate a dense spatial transfer matrix for the J disorder averaged
        influence matrix formalism similar to arXiv:2012.00777
        Site ordering as in ising_T.
    '''
    U1=DiagonalLinearOperator(ising_h(T,h).diag*ising_W(T,g).diag)
    U2=ising_Jr(T)
    return U2@U1
#TODO add new ref if available
def ising_Jhr_T(T,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism with averaging over both J and h.
        Site ordering as in ising_T.
    '''
    #TODO add new ref if available
    U1=DiagonalLinearOperator(ising_hr(T).diag*ising_W(T,g).diag)
    U2=ising_Jr(T)
    return U2@U1

def ising_SFF(T,J,g,h):
    pass

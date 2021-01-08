import numpy as np
from ..utils import popcount
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator

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
def fwht(a):
    '''
        Performs an inplace Fast-Walsh-Hadamard transformation on array a.
    '''
    h = 1
    slen=len(a)
    while h < slen:
        a=a.reshape((slen//h,h))
        a[::2,:],a[1::2,:]=a[::2,:]+a[1::2,:],a[::2,:]-a[1::2,:]
        a=a.reshape((slen,))
        h *= 2
def disorder_sector(L):
    cn=0
    sec={}
    invsec=[]
    for i in range(2**(2*L)):
        if popcount((i>>L)&(~(1<<(L-1))))==popcount((i^((i>>L)<<L))&(~(1<<(L-1)))):
            sec[i]=cn
            invsec.append(i)
            cn+=1
    return (2*L,sec,invsec)
def sparse_to_dense(spop):
    return spop@np.eye(spop.shape[0]) # For now
# def FWHTOperator(spla.LinearOperator):
#     def __init__(self):
#         pass
# def FloquetIsingLinearOperator(spla.LinearOperator):
#     def __init__(self,D1,D2):
#         pass
#     def matvec(self,a):
#         fwht(a)
#         v=self.D2*a/a.shape[0]
#         fwht(a)
#         v=self.D1*a
#     def rmatvec(self,a):
#         pass
#     def adjoint(self):
#         pass
#     def to_dense(self):
#         pass

import numpy as np
import scipy.sparse.linalg as spla

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

def IsingLinearOperator(spla.LinearOperator):
    def __init__(self,D1,D2):
        pass
    def matvec(self,a):
        fwht(a)
        v=self.D2*a/a.shape[0]
        fwht(a)
        v=self.D1*a
    def rmatvec(self,a):
        pass
    def adjoint(self):
        pass
    def to_dense(self):
        pass

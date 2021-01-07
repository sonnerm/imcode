import numpy as np
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
def ising_H(J,g,h):
    '''
    '''
    pass


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
def _apply_F_dual(sec,T,D1,D2,v):
    v=embed(v,sec)
    fwht(v)
    v=D2*v/v.shape[0]
    fwht(v)
    v=D1*v
    return v[sec[2]]

def _apply_F_dual_ad(sec,T,D1,D2,v):
    v=embed(v,sec)
    v=D1.conj()*v
    fwht(v)
    v=D2.conj()*v/v.shape[0]
    fwht(v)
    return v[sec[2]]
def ising_F(J,g,h):
    '''
    '''
    Jt,gt,eta1,eta2=dualU(J,g)
    gt=np.array([0.0]+[2*gt.conj()]*(T-1)+[0.0]+[-2*gt]*(T-1))
    D2=np.exp(1.0j*get_imbrie_diag(gt,np.zeros_like(gt)))
    D2*=np.exp(-(2*T-2)*eta2.real)
    D2/=2
    D2*=get_keldysh_boundary(T)
    h=np.array([0.0]+[2*h]*(T-1)+[0.0]+[-2*np.array(h).conj()]*(T-1))
    Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
    D1=np.exp(-1.0j*get_imbrie_diag(h,Jt))
    D1*=np.exp(2*T*eta1.real)
    return spla.LinearOperator((len(sec[2]),len(sec[2])),lambda v:_apply_F_dual(sec,T,D1,D2,v),lambda v:_apply_F_dual_ad(sec,T,D1,D2,v))

def ising_T(T,J,g,h):
    pass

def ising_SFF(T,J,g,h):
    pass

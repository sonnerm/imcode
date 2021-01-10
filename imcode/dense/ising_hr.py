from functools import lru_cache
from .ising import ising_W,ising_J
from ..utils import popcount
import numpy as np

@lru_cache(None)
def hr_operator(T):
    ret=np.zeros(2**(2*T))
    for i in range(2**(2*T)):
        if popcount((i>>T)&(~(1<<(T-1))))==popcount((i^((i>>T)<<T))&(~(1<<(T-1)))):
            ret[i]=1
    return np.diag(ret)

def ising_hr_T(T,J,g):
    r'''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism described in arXiv:2012.00777. The averaging
        is performed over parameter h. Site ordering as in ising_T.
    '''
    U1=hr_operator(T)*ising_W(T,g)
    U2=ising_J(T,J)
    return U2@U1

def ising_hr_Tp(T,J,g):
    U1=ising_W(T,g)
    U2=ising_J(T,J)
    Up=hr_operator(T)
    return Up@U2@U1

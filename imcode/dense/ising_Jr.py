from functools import lru_cache
from .ising import ising_W,ising_h
from .ising_hr import hr_operator
from ..utils import popcount
import numpy as np


@lru_cache(None)
def Jr_operator(T):
    ret=np.zeros((2**(2*T),2**(2*T)))
    for i in range(2**(2*T)):
        if popcount((i>>T)&(~(1<<(T-1))))==popcount((i^((i>>T)<<T))&(~(1<<(T-1)))):
            for j in range(2**(2*T)):
                ret[j,j^i]=1
    return ret/2
def ising_Jr_T(T,g,h):

    r'''
        Calculate a dense spatial transfer matrix for the J disorder averaged
        influence matrix formalism similar to arXiv:2012.00777
        Site ordering as in ising_T.
    '''
    U1=ising_h(T,h)*ising_W(T,g)
    U2=Jr_operator(T)
    return U2@U1
#TODO add new ref if available
def ising_Jhr_T(T,g):
    r'''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism with averaging over both J and h.
        Site ordering as in ising_T.
    '''
    #TODO add new ref if available
    U1=hr_operator(T)*ising_W(T,g)
    U2=Jr_operator(T)
    return U2@U1

def ising_Jhr_Tp(T,g):
    U1=ising_W(T,g)
    U2=Jr_operator(T)
    Up=hr_operator(T)
    return Up@U2@U1

from ..utils import multiply_mpos,wrap_ndarray
from .utils import FoldSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import numpy as np
def dephase_operator(t,gamma):
    '''
        Returns a dephase operator, gamma = 0 no dephasing, gamma =1 perfect dephaser
    '''
    return
def dephase_im(t,gamma):
    return
def depolarize_operator(t,p):
    '''
        Returns a depolarization operator, p = 0 no depolarization, p =1 full depolarization
    '''
    return

from ..utils import multiply_mpos,wrap_ndarray
from .utils import FoldSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import numpy as np
def dephaser_operator(t,gamma):
    '''
        Returns a dephase operator, gamma = 0 no dephasing, gamma =1 perfect dephaser
    '''
    sites=[FoldSite() for t in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    Ceprim=np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0-gamma,0.0],[0.0,0.0,0.0,1.0-gamma]])
    Cea=np.einsum("ab,cd->abcd",np.eye(1),Ceprim)
    Ce=npc.Array.from_ndarray(Cea,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO(sites,[Ce]*(t+1))
def depolarizer_operator(t,p):
    '''
        Returns a depolarization operator, p = 0 no depolarization, p =1 full depolarization
    '''
    sites=[FoldSite() for t in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    Ceprim=np.array([[1.0-p/2,p/2,0.0,0.0],[p/2,1.0-p/2,0.0,0.0],[0.0,0.0,1.0-p,0.0],[0.0,0.0,0.0,1.0-p]])
    Cea=np.einsum("ab,cd->abcd",np.eye(1),Ceprim)
    Ce=npc.Array.from_ndarray(Cea,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO(sites,[Ce]*(t+1))
def ising_hr_dephase_T(t,J,g,gamma):
    return multiply_mpos([ising_hr_T(t,J,g),dephaser_operator(t,gamma)])
def ising_hr_dephase_Tp(t,J,g,gamma):
    return multiply_mpos([ising_hr_Tp(t,J,g),dephaser_operator(t,gamma)])
def ising_hr_depolarize_T(t,J,g,gamma):
    return multiply_mpos([ising_hr_T(t,J,g),depolarizer_operator(t,gamma)])
def ising_hr_depolarize_Tp(t,J,g,gamma):
    return multiply_mpos([ising_hr_Tp(t,J,g),depolarizer_operator(t,gamma)])

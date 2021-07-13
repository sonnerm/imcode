from .. import MPO,MPS
import numpy as np
def dephaser_operator(t,gamma):
    '''
        Returns a dephase operator, gamma = 0 no dephasing, gamma =1 perfect dephaser
    '''
    Ceprim=np.array([[1.0,0.0,0.0,0.0],[0.0,1.0-gamma,0.0,0.0],[0.0,0.0,1.0-gamma,0.0],[0.0,0.0,0.0,1.0]])
    Cea=np.einsum("ab,cd->abcd",np.eye(1),Ceprim)
    return MPO.from_matrices([Cea]*t)
def depolarizer_operator(t,p):
    '''
        Returns a depolarization operator, p = 0 no depolarization, p =1 full depolarization
    '''
    Ceprim=np.array([[1.0-p/2,0.0,0.0,p/2],[0,1.0-p,0.0,0.0],[0.0,0.0,1.0-p,0.0],[p/2,0.0,0.0,1.0-p/2]])
    Cea=np.einsum("ab,cd->abcd",np.eye(1),Ceprim)
    return MPO.from_matrices([Cea]*t)

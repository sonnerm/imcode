import numpy as np
from .. import outer
def open_boundary_im(t):
    return np.ones((4**(t)))
def perfect_dephaser_im(t):
    return dephaser_im(t,1.0)
def dephaser_im(t,gamma):
    return outer([[1,1-gamma,1-gamma,1]]*t)

def im_diag(T):
    '''
        Obtain the semi-infinite chain influence matrix by fully diagonalizing
        the transfer matrix `T` and taking the eigenvector to the largest eigenvalue
        normalized such that classical trajectories are one, fd results are returned
    '''
    ev,evv=la.eig(T)
    return (evv[:,np.argmax(ev)]/evv[0,np.argmax(ev)],(ev,evv))

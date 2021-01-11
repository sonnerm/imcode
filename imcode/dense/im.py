import numpy as np
import numpy.linalg as la
def open_boundary_im(t):
    return np.ones((2**(2*t)))
def perfect_dephaser_im(t):
    pass

def im_iterative(T):
    '''
        Obtain the semi-infinite chain influence matrix by iterating the transfer matrix `T` 2T times.
    '''
    im_finite([T]*(2*T))
def im_finite(Ts,boundary=None):
    if boundary is None:
        t2=int(np.log2(Ts[0].shape[0]))
        vec=np.ones(Ts[0].shape[0])
    else:
        vec=boundary
    for T in Ts:
        vec=T@vec
    return vec


def im_diag(T):
    '''
        Obtain the semi-infinite chain influence matrix by fully diagonalizing
        the transfer matrix `T` and taking the eigenvector to the largest eigenvalue
        normalized such that classical trajectories are one, fd results are returned
    '''
    ev,evv=la.eig(T)
    return (evv[:,np.argmax(ev)]/evv[0,np.argmax(ev)],(ev,evv))

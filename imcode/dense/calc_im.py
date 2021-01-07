import numpy as np
import numpy.linalg as la
def calc_im_iterative(T):
    '''
        Obtain the semi-infinite chain influence matrix by iterating the transfer matrix `T` 2T times.
    '''
    calc_im_finite(T,2*T)
def calc_im_finite(T,L):
    t2=int(np.log2(T.shape[0]))
    vec=np.ones(T.shape[0])
    for _ in range(L):
        vec=T@vec
    return vec


def calc_im_diag(T):
    '''
        Obtain the semi-infinite chain influence matrix by fully diagonalizing
        the transfer matrix `T` and taking the eigenvector to the largest eigenvalue
        normalized such that classical trajectories are one, fd results are returned
    '''
    ev,evv=la.eig(T)
    return (evv[:,np.argmax(ev)]/evv[0,np.argmax(ev)],(ev,evv))

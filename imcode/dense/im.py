import numpy as np
import numpy.linalg as la
from .utils import popcount
def open_boundary_im(t):
    return np.ones((2**(2*t)))
def perfect_dephaser_im(t):
    ret=np.zeros((2**(2*t)))
    L=1<<(2*t-1)
    R=1<<(t-1)
    for i in range(2**(t-1)):
        ret[L|(i<<t)|R|i]=1
        ret[(i<<t)|R|i]=1
        ret[L|(i<<t)|i]=1
        ret[(i<<t)|i]=1
    return ret
def dephaser_im(t,gamma):
    ret=np.zeros((2**(2*t)))
    mask_bw=(1<<(t-1))-1
    mask_fw=mask_bw<<t
    for i in range(2**(2*t)):
        ret[i]=(1-gamma)**(popcount(i&mask_bw)-popcount(i&mask_fw))
    return ret
def im_iterative(T):
    '''
        Obtain the semi-infinite chain influence matrix by iterating the transfer matrix `T` 2T times.
    '''
    t=int(np.log2(T.shape[0]))
    return im_finite([T]*(2*t))
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

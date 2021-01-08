import numpy as np
import numpy.linalg as la
def im_iterative(T):
    '''
        Obtain the semi-infinite chain influence matrix by iterating the transfer matrix `T`
    '''
    t2=int(np.log2(T.shape[0]))
    vec=np.ones(T.shape[0])
    for _ in range(t2):
        vec=T@vec
    return vec

def im_lanczos(T):
    '''
        Obtain the semi-infinite chain influence matrix by fully diagonalizing
        the transfer matrix `T` and taking the eigenvector to the largest eigenvalue
        normalized such that classical trajectories are one, fd results are returned
    '''
    ev,evv=la.eig(T)
    return (evv[:,np.argmax(ev)]/evv[0,np.argmax(ev)],(ev,evv))

def im_finite(Ts,boundary=None):
    if boundary is None:
        t2=int(np.log2(Ts[0].shape[0]))
        vec=np.ones(Ts[0].shape[0])
    else:
        vec=boundary
    for T in Ts:
        vec=T@vec
    return vec

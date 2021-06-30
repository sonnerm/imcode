import numpy as np
import numpy.linalg as la
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

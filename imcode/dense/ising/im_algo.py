from . import open_boundary_im
from .. import outer
import numpy as np
import numpy.linalg as la
def im_diag(T):
    ev,evv=la.eig(T)
    oev=evv[:,np.argmax(np.abs(ev))]
    oev/=oev[0]
    return oev
def im_rectangle(Ts,boundary=None):
    if isinstance(Ts,np.ndarray) and len(Ts.shape)==2:
        Ts=[Ts]*int(np.log2(Ts.shape[0]))
    if boundary is None:
        boundary=open_boundary_im(int(np.log2(Ts[0].shape[0]))//2)
    im=boundary
    yield im
    for T in Ts:
        im=T@im
        yield im
def im_diamond(Ts):
    imit=[1,0,0,1]
    im=None
    for T in Ts:
        if im is None:
            im=imit
        else:
            im=outer([imit,im,imit])
        im=T@im
        yield im
def im_triangle(Ts):
    imit=[1,0,0,1]
    im=None
    for T in Ts:
        if im is None:
            im=imit
        else:
            im=outer([im,imit])
        im=T@im
        yield im
def im_direct(Fs, init=None):
    state = init.reshape(init.shape[0], 1, init.shape[1])#reshape density matrix to bring it into general form with open legs "in the middle"
    for F in Fs:
        F_apply_shape = (2, F.shape[0]//2, 2, F.shape[1]//2)#shape needed for layer to multiply it to state
        F = F.reshape(F_apply_shape)
        #apply F
        state = np.einsum('aibj, jqk, kdle  -> aqdil', F, state, F.T.conj())

        #reshape state such that last spin can be traced out
        trace_shape = (dim_open_legs_per_branch**2, 2**(L - 1 - n_traced), 4)#replace by einsum
        state = np.reshape(state, trace_shape)
        #trace out the two last spins
        yield np.einsum('abb', state)





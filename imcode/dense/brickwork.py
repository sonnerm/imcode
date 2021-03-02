import numpy as np
from .utils import dense_kron
def brickwork_T(t,gate,lop,init=(0.5,0.5)):
    bs=brickwork_S(t,gate)
    return fuse_final(bs)@brickwork_L(t,lop,init)@fuse_initial(bs)@brickwork_L(t,lop,init).T

def brickwork_S(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    return dense_kron([gate.reshape(4,4)]*t+[gate.reshape(4,4).conj()]*t)
def fuse_initial(S):
    '''
        fuse transfer matrix at inital time (impose diagonal initial density matrix, observable)
    '''
    ret=S.reshape((2,S.shape[0]//4,2,S.shape[1]))
    ret=ret[(0,1),:,(0,1),:]
    ret=ret.reshape((S.shape[0]//2,2,S.shape[1]//4,2))
    ret=ret[:,(0,1),:,(0,1)]
    return ret.reshape((S.shape[0]//2,S.shape[1]//2))

def fuse_final(S):
    '''
        fuse transfer matrix at final time (impose diagonal observables)
    '''
    t=int(np.sqrt(S.shape[0]))//2
    ret=S.reshape((t,2,2,t*S.shape[1]))
    ret=ret[:,(0,1),(0,1),:]
    t=int(np.sqrt(S.shape[1]))//2
    ret=ret.reshape((S.shape[0]//2*t,2,2,t))
    ret=ret[:,(0,1),(0,1),:]
    return ret.reshape((S.shape[0]//2,S.shape[1]//2))

def brickwork_L(t,lop,init=(0.5,0.5)):
    '''
        brickwork "local" operator
    '''
    ret=np.array(init).reshape((2,1))
    for i in range(1,2*t):
        ret=np.einsum("xy,ab->xayb",ret,lop).reshape((ret.shape[0]*lop.shape[0],ret.shape[1]*lop.shape[1]))
    ret=np.einsum("xy,a->xya",ret,np.array((1.0,1.0))).reshape(ret.shape[0],ret.shape[1]*2)
    for i in range(1,2*t):
        ret=np.einsum("xy,ab->xayb",ret,lop.conj()).reshape((ret.shape[0]*lop.shape[0],ret.shape[1]*lop.shape[1]))
    return ret
def brickwork_open_boundary_im(t):
    ret=dense_kron([np.eye(2)]*(2*t)).ravel()
    ret=ret.reshape([2**(2*t-1),2,2,2**(2*t-1)])[:,(0,1),(0,1),:].ravel()
    return ret

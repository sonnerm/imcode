import numpy as np
from .utils import dense_kron
def brickwork_F(gates,lops):
    assert len(lops)%2==0
    ls=dense_kron([l for l in lops])
    evs=dense_kron([g for g in gates[::2]])
    if len(gates)==len(lops)-1:
        gates=list(gates)+[np.eye(4)]
    ods=dense_kron([g for g in gates[1::2]])
    ods=ods.reshape((ods.shape[0]//4,2,2,ods.shape[1]//4,2,2))
    ods=ods.transpose([2,0,1,5,3,4]).reshape((evs.shape[0],evs.shape[1]))
    return ods@ls@evs@ls
def brickwork_T(t,gate,lop,init=(0.5,0.5)):
    bs=brickwork_S(t,gate)
    bl=brickwork_L(t,lop,init)
    return bl@bs@bl.T@bs

def brickwork_S(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    gate=gate.reshape(2,2,2,2).transpose([0,2,1,3]).reshape(4,4)#dual gate
    gate=np.kron(gate,gate.conj())
    return dense_kron([gate.reshape(16,16)]*t)

def brickwork_L(t,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
    '''
        brickwork "local" operator
    '''
    infi=np.outer(np.array(init),np.array(final))
    lop=np.kron(lop,lop.conj()).reshape(4,4)
    ret=dense_kron([infi]+[lop]*(2*t-1))
    ret=ret.reshape((4,ret.shape[0]//4,4,ret.shape[1]//4)).transpose([0,1,3,2]).reshape((ret.shape[0],ret.shape[1]))
    return ret

# def fuse_initial(S):
#     '''
#         fuse transfer matrix at inital time (impose diagonal initial density matrix, observable)
#     '''
#     ret=S.reshape((2,S.shape[0]//4,2,S.shape[1]))
#     ret=ret[(0,1),:,(0,1),:]
#     ret=ret.reshape((S.shape[0]//2,2,S.shape[1]//4,2))
#     ret=ret[:,(0,1),:,(0,1)]
#     return ret.reshape((S.shape[0]//2,S.shape[1]//2))
#
# def fuse_final(S):
#     '''
#         fuse transfer matrix at final time (impose diagonal observables)
#     '''
#     t=int(np.sqrt(S.shape[0]))//2
#     ret=S.reshape((t,2,2,t*S.shape[1]))
#     ret=ret[:,(0,1),(0,1),:]
#     t=int(np.sqrt(S.shape[1]))//2
#     ret=ret.reshape((S.shape[0]//2*t,2,2,t))
#     ret=ret[:,(0,1),(0,1),:]
#     return ret.reshape((S.shape[0]//2,S.shape[1]//2))

def brickwork_zz_operator(t):
    ret=dense_kron([np.diag([1,0,0,-1])]+[np.eye(4)]*(2*t-2)+[np.diag([1,0,0,-1])])
    return ret
def brickwork_open_boundary_im(t):
    ret=dense_kron([np.eye(2)]*(2*t)).ravel()
    return ret

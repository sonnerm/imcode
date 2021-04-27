import numpy as np
from .utils import dense_kron,dense_outer
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
# def brickwork_T(t,gate,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
#     bs=brickwork_S(t,gate)
#     bl=brickwork_L(t,lop,init,final)
#     return bs@bl.T@bs@bl
def brickwork_T(t,gate,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
    #TODO remove cheat
    bs=brickwork_S(t,gate@np.kron(lop,lop))
    bl=brickwork_L(t,np.eye(2),init,final)
    return bs@bl@bs@bl


def brickwork_S(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    gate=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2)))
    return dense_kron([gate.reshape(16,16)]*t)

# def brickwork_L(t,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
#     '''
#         brickwork "local" operator
#     '''
#     infi=np.outer(np.array(init),np.array(final))
#     lop=np.kron(lop,lop.conj()).reshape(4,4)
#     ret=dense_kron([infi]+[lop]*(2*t-1))
#     ret=ret.reshape((4,ret.shape[0]//4,4,ret.shape[1]//4)).transpose([0,1,3,2]).reshape((ret.shape[0],ret.shape[1]))
#     return ret

def brickwork_Lt(t,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
    '''
        brickwork "local" operator
    '''
    init=np.array(init).reshape((1,4))
    final=np.array(final).reshape((4,1))
    lop=np.kron(lop,lop.conj()).reshape((4,4))
    ret=dense_kron([init]+([lop,np.eye(4)]*t)[:-1]+[final])
    return ret

def brickwork_L(t,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
    '''
        brickwork "local" operator
    '''
    init=np.array(init).reshape((4,1))
    final=np.array(final).reshape((1,4))
    lop=np.kron(lop,lop.conj()).reshape((4,4))
    ret=dense_kron([init]+([lop,np.eye(4)]*t)[:-1]+[final])
    return ret

# def brickwork_zz_operator(t):
#     ret=dense_kron([np.diag([1,0,0,-1])]+[np.eye(4)]*(2*t-2)+[np.diag([1,0,0,-1])])
#     return ret

def brickwork_zz_operator(t):
    return brickwork_L(t,np.eye(2),init=(0.5,0,0,-0.5),final=(1.0,0.0,0.0,-1.0))
def brickwork_open_boundary_im(t):
    ret=dense_outer([np.eye(4).ravel()]*t).ravel()
    return ret

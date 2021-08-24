import numpy as np
def unitary_channel(F):
    L=int(np.log2(F.shape[0]))
    ret=np.kron(F,F.conj()).reshape((2,)*(4*L))
    tplist=sum(zip(range(L),range(L,2*L)),())+sum(zip(range(2*L,3*L),range(3*L,4*L)),())
    ret=ret.transpose(tplist)
    return ret.reshape((F.shape[0]**2,F.shape[1]**2))
def dephasing_channel(gamma,basis=np.eye(2)):
    D=np.diag([1.0,1.0-gamma,1.0-gamma,1.0])
    U=unitary_channel(basis)
    return U.T.conj()@D@U
def depolarizing_channel(p,dm=np.eye(2)/2):
    d=dm.shape[0]
    ret=np.eye(d**2)*(1-p)
    ret+=np.outer(dm.ravel(),np.eye(d).ravel())*p
    return ret

def operator_to_state(op):
    L=int(np.log2(op.shape[0]))
    op=op.reshape((2,)*(2*L))
    tplist=sum(zip(range(L),range(L,2*L)),())
    op=op.transpose(tplist)
    return op.reshape(4**L)

def state_to_operator(state):
    L=int(np.log2(state.shape[0])//2)
    state=state.reshape((2,)*(2*L))
    tplist=list(range(0,2*L,2))+list(range(1,2*L,2))
    state=state.transpose(tplist)
    return state.reshape(2**L,2**L)

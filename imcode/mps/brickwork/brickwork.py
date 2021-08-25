import numpy as np
from ... import dense
import numpy.linalg as la
from .. import MPO,MPS
def brickwork_Fe(L,gatese):
    Bs=[]
    for g in gatese:
        gatel=np.einsum("ac,bd->abcd",np.eye(2),np.eye(2)).reshape(1,4,2,2)
        gater=g.reshape(2,2,2,2).transpose([0,2,1,3]).reshape(4,1,2,2)
        Bs.append(gatel)
        Bs.append(gater)
    gatebound=np.array([[np.eye(2)]])
    if L%2==1:
        Bs.append(gatebound)
    return MPO.from_matrices(Bs)

def brickwork_Fo(L,gateso):
    gatebound=np.array([[np.eye(2)]])
    Bs=[gatebound]
    for g in gateso:
        gatel=np.einsum("ac,bd->abcd",np.eye(2),np.eye(2)).reshape(1,4,2,2)
        gater=g.reshape(2,2,2,2).transpose([0,2,1,3]).reshape(4,1,2,2)
        Bs.append(gatel)
        Bs.append(gater)
    if L%2==0:
        Bs.append(gatebound)
    return MPO.from_matrices(Bs)
def brickwork_F(L,gates,reversed=False):
    if not reversed:
        return (brickwork_Fo(L,gates[1::2])@brickwork_Fe(L,gates[::2])).contract()
    else:
        return (brickwork_Fe(L,gates[::2])@brickwork_Fo(L,gates[1::2])).contract()
def brickwork_H(L,gates):
    pass
def brickwork_La(t):
    Me=np.eye(4)[None,:,:]
    Mo=np.eye(4)[:,None,:]
    return MPS.from_matrices([Me,Mo]*t)
def brickwork_Lb(t,gate,init=np.eye(2)/2,final=np.eye(2)):
    init=gate@init.ravel()
    init=init.reshape((1,1,4))
    final=final.T.reshape((1,1,4))
    u,s,v=la.svd(gate)
    us=u*np.sqrt(s)
    vs=(v.T*np.sqrt(s)).T
    gatea=vs[None,:,:]
    gateb=us.T[:,None,:]
    return MPS.from_matrices([init]+[gatea,gateb]*(t-1)+[final])
def brickwork_Sa(t, gate):
    '''
        dual layer of the brickwork transfer matrix without boundary states
    '''
    u,s,v=la.svd(gate)
    us=u*np.sqrt(s)
    vs=(v.T*np.sqrt(s)).T
    gatea=vs.reshape((1,16,4,4))
    gateb=us.T.reshape((16,1,4,4))
    return MPO.from_matrices([gatea,gateb]*t)

def brickwork_Sb(t, gate,init=np.eye(4)/4,final=np.eye(4)):
    '''
        dual layer of the brickwork transfer matrix with boundary states
    '''
    inita=dense.operator_to_state(init)
    inita=gate@inita
    inita=inita.reshape((1,1,4,4))
    finala=dense.operator_to_state(final.T).reshape((1,1,4,4))
    u,s,v=la.svd(gate)
    us=u*np.sqrt(s)
    vs=(v.T*np.sqrt(s)).T
    gatea=vs.reshape((1,16,4,4))
    gateb=us.T.reshape((16,1,4,4))
    return MPO.from_matrices([inita]+[gatea,gateb]*(t-1)+[finala])

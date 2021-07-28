import numpy as np
import numpy.linalg as la
from .. import MPO
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
def brickwork_La():
    pass
def brickwork_Lb():
    pass
def brickwork_Sa(t, gate):
    '''
        dual layer of the brickwork transfer matrix without boundary states
    '''
    # gate=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape((16,16))
    gate=np.einsum("abcd,efgh->aebfcgdh",gate.reshape((2,2,2,2)),gate.reshape((2,2,2,2)).conj()).reshape(16,16)
    u,s,v=la.svd(gate)
    gateb=(v).reshape((16,1,4,4))
    gatea=(u*s).T.reshape((1,16,4,4))
    return MPO.from_matrices([gatea,gateb]*t)



def brickwork_Sb(t, gate,init=np.eye(4),final=np.eye(4)):
    '''
        dual layer of the brickwork transfer matrix with boundary states
    '''
    gate=gate.reshape((2,2,2,2))
    inita=np.einsum("cdab,abef,cdgh->egfh",init.reshape((2,2,2,2)),gate,gate.conj()) #No idea why init.T but works
    inita=inita.reshape((1,1,4,4))
    final=np.einsum("abcd->acbd",final.reshape((2,2,2,2))).reshape((4,4))
    finala=final.reshape((1,1,4,4))
    gate=np.einsum("abcd,efgh->aebfcgdh",gate.reshape((2,2,2,2)),gate.reshape((2,2,2,2)).conj()).reshape(16,16)
    u,s,v=la.svd(gate)
    gateb=(v).reshape((16,1,4,4))
    gatea=(u*s).T.reshape((1,16,4,4))
    return MPO.from_matrices([inita]+[gatea,gateb]*(t-1)+[finala])
def brickwork_T(t,even,odd,init=np.eye(4),final=np.eye(4)):
    return (brickwork_Sa(t,even)@brickwork_Sb(t,odd,init,final)).contract()

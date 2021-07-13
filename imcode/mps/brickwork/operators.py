import numpy as np
import numpy.linalg as la
from .. import MPO
def brickwork_Fe(gatese):
    Bs=[]
    for g in gatese:
        u,s,v=la.svd(gate)
        gateb=(v).reshape((4,1,2,2))
        gatea=(u*s).T.reshape((1,4,2,2))
        Bs.append(gateb)
        Bs.append(gatea)
    return MPO.from_matrices(Bs)

def brickwork_Fo(gateso):
    gatebound=np.array([[np.eye(2)]])
    Bs=[gatebound]
    for g in gateso:
        u,s,v=la.svd(gate)
        gateb=(v).reshape((16,1,4,4))
        gatea=(u*s).T.reshape((1,16,4,4))
        Bs.append(gateb)
        Bs.append(gatea)
    Bs.append(gatebound)
    return MPO.from_matrices(Bs)
def brickwork_F(gates):
    return (brickwork_Fe(gates[::2])@brickwork_Fo(gates[1::2])).contract()
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

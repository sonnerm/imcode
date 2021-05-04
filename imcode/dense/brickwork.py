import numpy as np
from .utils import dense_kron,dense_outer
def brickwork_F(gates):
    if len(gates)==1:
        return gates[0]
    evs=dense_kron([g for g in gates[::2]]+([np.eye(2)] if len(gates)%2==0 else []))
    ods=dense_kron([np.eye(2)]+[g for g in gates[1::2]]+([np.eye(2)] if len(gates)%2==1 else []))
    return ods@evs
# def brickwork_T(t,gate,lop,init=(0.5,0.0,0.0,0.5),final=(1.0,0.0,0.0,1.0)):
#     bs=brickwork_S(t,gate)
#     bl=brickwork_L(t,lop,init,final)
#     return bs@bl.T@bs@bl
def brickwork_T(t,even,odd,init=np.eye(4),final=np.eye(4)):
    sa=brickwork_Sa(t,odd)
    sb=brickwork_Sb(t,even,init,final)
    return sa@sb


def brickwork_Sa(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    dual=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape(16,16)
    return dense_kron([dual]*t)

def brickwork_Sb(t, gate,init=np.eye(4),final=np.eye(4)):
    gate=gate.reshape((2,2,2,2))
    dual=np.einsum("abcd,efgh->aecgbfdh",gate,gate.conj()).reshape((16,16))
    init=np.einsum("cdab,abef,cdgh->egfh",init.reshape((2,2,2,2)),gate,gate.conj()) #No idea why init.T but works
    init=init.reshape((4,4))
    # print(init)
    final=np.einsum("abcd->acbd",final.reshape((2,2,2,2))).reshape((4,4))
    return dense_kron([init]+[dual]*(t-1)+[final])

def brickwork_La(t):
    ret=dense_outer([np.eye(4).ravel()]*t).ravel()
    return ret

def brickwork_Lb(t,lop,init=np.eye(2),final=np.eye(2)):
    init=np.einsum("ab,cd,ca->bd",lop,lop.conj(),init)
    lop=np.kron(lop,lop.conj())
    ret=dense_outer([init.ravel()]+[lop.ravel()]*(t-1)+[final.ravel()]).ravel()
    return ret

import numpy as np
from .. import kron,outer
def brickwork_H(L,gates):
    assert L>1
    ret=np.zeros((2**L,2**L),dtype=np.common_type(*gates))
    for i in range(L-1):
        ret+=kron([np.eye(2**i),gates[i],np.eye(2**(L-i-2))])
    if len(gates)==L:
        ret+=kron([np.eye(2**(L-2)),gates[-1]]).reshape(2**(L-1),2,2**(L-1),2).transpose([1,0,3,2]).reshape((2**L,2**L))
    return ret



def brickwork_F(L,gates,reversed=False):
    if len(gates)==L-1:
        gates.append(np.eye(4))
    evs=kron([g for g in gates[::2]]+([np.eye(2)] if L%2==1 else []))
    ods=kron([g for g in gates[1::2]]+([np.eye(2)] if L%2==1 else []))
    ods=ods.reshape((2**(L-1),2,2**(L-1),2)).transpose([1,0,3,2]).reshape((2**L,2**L))
    if not reversed:
        return ods@evs
    else:
        return evs@ods
def brickwork_T(t,even,odd,init=np.eye(4)/4,final=np.eye(4)):
    sa=brickwork_Sa(t,odd)
    sb=brickwork_Sb(t,even,init,final)
    return sa@sb


def brickwork_Sa(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    dual=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape(16,16)
    return kron([dual]*t)

def brickwork_Sb(t, gate,init=np.eye(4)/4,final=np.eye(4)):
    gate=gate.reshape((2,2,2,2))
    dual=np.einsum("abcd,efgh->aecgbfdh",gate,gate.conj()).reshape((16,16))
    init=np.einsum("cdab,abef,cdgh->egfh",init.reshape((2,2,2,2)),gate,gate.conj()) #No idea why init.T but works
    init=init.reshape((4,4))
    # print(init)
    final=np.einsum("abcd->acbd",final.reshape((2,2,2,2))).reshape((4,4))
    return kron([init]+[dual]*(t-1)+[final])

def brickwork_La(t):
    ret=outer([np.eye(4).ravel()]*t).ravel()
    return ret

def brickwork_Lb(t,lop,init=np.eye(2)/2,final=np.eye(2)):
    init=np.einsum("ab,cd,ca->bd",lop,lop.conj(),init)
    lop=np.kron(lop,lop.conj())
    ret=outer([init.ravel()]+[lop.ravel()]*(t-1)+[final.ravel()]).ravel()
    return ret

import numpy as np
from .. import kron,outer,operator_to_state
def brickwork_H(L,gates):
    assert L>1
    ret=np.zeros((2**L,2**L),dtype=np.common_type(*gates))
    for i in range(L-1):
        ret+=kron([np.eye(2**i),gates[i],np.eye(2**(L-i-2))])
    if len(gates)==L:
        ret+=kron([np.eye(2**(L-2)),gates[-1]]).reshape(2**(L-1),2,2**(L-1),2).transpose([1,0,3,2]).reshape((2**L,2**L))
    return ret



def brickwork_F(L,gates,reversed=False):
    assert (len(gates)!=L) or (L%2==0) #periodic doesn't work for odd number of sites
    if (len(gates)==L-1) and (L%2==0):
        gates=gates+[np.eye(4)]#no modification inplace
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
    gate=gate.reshape((4,4,4,4)).transpose([0,2,1,3])
    return kron([gate.reshape(16,16)]*t)

def brickwork_Sb(t, gate,init=np.eye(4)/4,final=np.eye(4)):
    init=operator_to_state(init)
    init=(gate@init).reshape((4,4))
    final=operator_to_state(final).reshape((4,4))
    gate=gate.reshape((4,4,4,4)).transpose([0,2,1,3])
    return kron([init]+[gate]*(t-1)+[final])

def brickwork_La(t):
    ret=outer([np.eye(4).ravel()]*t).ravel()
    return ret

def brickwork_Lb(t,lop,init=np.eye(2)/2,final=np.eye(2)):
    init=lop@init.ravel()
    ret=outer([init.ravel()]+[lop.ravel()]*(t-1)+[final.ravel()]).ravel()
    return ret

import numpy as np
import ttarray as tt

def brickwork_Fe(L,gatese):
    gatese=np.asarray(gatese)
    if len(gatese.shape)==2:
        gatese=[gatese for _ in range(L//2)]
    if L%2==0:
        Bs=gatese
    else:
        gatebound=np.eye(2)[None,...,None]
        Bs=list(gatese)+[gatebound]
    return tt.frommatrices(Bs).recluster(((2,),)*L)

def brickwork_Fo(L,gateso):
    gatebound=np.eye(2)[None,...,None]
    if L%2==0:
        Bs=[gatebound]+gateso+[gatebound]
    else:
        Bs=[gatebound]+gateso
    return tt.frommatrices(Bs).recluster(((2,),)*L)

def brickwork_F(L,gates,reversed=False):
    if not reversed:
        return brickwork_Fo(L,gates[1::2])@brickwork_Fe(L,gates[::2]))
    else:
        return brickwork_Fe(L,gates[::2])@brickwork_Fo(L,gates[1::2]))
def brickwork_H(L,gates):
    pass
def brickwork_La(t):
    return tt.fromproduct([np.eye(4)]*t)
def brickwork_Lb(t,chs):
    init=init.reshape((1,1,4))
    final=final.T.reshape((1,1,4))
    us=u*np.sqrt(s)
    vs=(v.T*np.sqrt(s)).T
    gatea=vs[None,:,:]
    gateb=us.T[:,None,:]
    init=np.einsum("abc,bde->adce",init,gatea).reshape((1,4,16))
    gate=np.einsum("abc,bde->adce",gateb,gatea).reshape((4,4,16))
    final=np.einsum("abc,bde->adce",gateb,final).reshape((4,1,16))
    return MPS.from_matrices([init]+[gate]*(t-2)+[final])

def brickwork_Sa(t, chs):
    '''
        dual layer of the brickwork transfer matrix without boundary states
    '''
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=(chs for _ in range(t))
    dual=[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((16,16)) for ch in chs]
    return tt.fromproduct(dual)

def brickwork_Sb(t, chs):
    '''
        dual layer of the brickwork transfer matrix with boundary states
    '''
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=[chs for _ in range(t)]
    dual=[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((16,16)) for ch in chs]
    # dual[-1]=
    init=np.array(init)
    final=np.array(final)
    return tt.frommatrices(dual) # for now ...
def brickwork_open_boundary_im(t):
    return brickwork_La(t)

def interleave_brickwork(ima,imb):
    pass

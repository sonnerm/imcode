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
def brickwork_La(t,chs=np.eye(4)):
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=(chs for _ in range(t))
    return tt.fromproduct_slice([ch.T for ch in chs])
def brickwork_Lb(t,chs):
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=(chs for _ in range(t))
    chs=[ch.T.reshape((1,4,4,1)) for ch in chs]
    chs[-1]=np.tensordot(chs[-1],np.eye(2).ravel(),axis=((1,),(0,))).reshape((1,4,1))
    chs[0]=chs[0].transpose([1,0])
    return tt.frommatrices_slice([ch.T for ch in chs])

def brickwork_Sa(t, chs):
    '''
        dual layer of the brickwork transfer matrix without boundary states
    '''
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=(chs for _ in range(t))
    dual=[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((16,16)) for ch in chs]
    return tt.fromproduct_slice(dual)

def brickwork_Sb(t, chs):
    '''
        dual layer of the brickwork transfer matrix with boundary states
    '''
    chs=np.asarray(chs)
    if len(chs.shape==2):
        chs=[chs for _ in range(t)]
    dual=[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((1,16,16,1)) for ch in chs]
    dual[-1]=np.tensordot(dual[-1].reshape((4,4,4,4)),np.eye(2).ravel(),axis=((1,),(0,))).reshape((1,16,1))
    dual[-1]=np.tensordot(dual[-1].reshape((4,4,4,4)),np.eye(2).ravel(),axis=((2,),(0,))).reshape((1,16,1))
    dual[0]=dual[0].transpose([1,0])
    return tt.frommatrices_slice(dual)
def brickwork_open_boundary_im(t):
    return brickwork_La(t).asarray()

def interleave_brickwork(ima,imb):
    res=[]
    for l,r in zip(lhs,rhs):
        l=l.reshape((l.shape[0],4,4,l.shape[-1]))
        r=r.reshape((r.shape[0],4,4,r.shape[-1]))
        res.append(np.einsum("abcd,ecfg->aebfdg",l,r).reshape(l.shape[0]*r.shape[0],16,l.shape[-1]*r.shape[-1]))
    return res

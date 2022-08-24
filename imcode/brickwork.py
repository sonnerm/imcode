import numpy as np
import ttarray as tt

def brickwork_Fe(L,gatese):
    gatese=np.asarray(gatese)
    if len(gatese.shape)==2:
        gatese=[gatese for _ in range(L//2)]
    if L%2==0:
        Bs=gatese
    else:
        Bs=list(gatese)+[np.eye(2)]
    return tt.fromproduct(Bs).recluster(((2,2),)*L)

def brickwork_Fo(L,gateso):
    if L%2==0:
        Bs=[np.eye(2)]+list(gateso)+[np.eye(2)]
    else:
        Bs=[np.eye(2)]+list(gateso)
    return tt.fromproduct(Bs).recluster(((2,2),)*L)

def brickwork_F(L,gates,reversed=False):
    if not reversed:
        return brickwork_Fo(L,gates[1::2])@brickwork_Fe(L,gates[::2])
    else:
        return brickwork_Fe(L,gates[::2])@brickwork_Fo(L,gates[1::2])
def brickwork_H(L,gates):
    nmat=np.zeros((1,2,2,6))
    nmat[0,:,:,:-2]=np.eye(4).reshape((2,2,4))
    nmat[0,:,:,-2]=np.eye(2)
    ret=[nmat]
    for g in gates:
        nmat=np.zeros((6,2,2,6),dtype=g.dtype)
        nmat[-2,:,:,:-2]=np.eye(4).reshape((2,2,4))
        nmat[-2,:,:,-2]=np.eye(2)
        nmat[-1,:,:,-1]=np.eye(2)
        nmat[:-2,:,:,-1]=g.reshape((2,2,2,2)).transpose((0,2,1,3)).reshape((4,2,2))
        ret.append(nmat)
    ret[-1]=ret[-1][:,:,:,-1:]
    return tt.frommatrices(ret)


def brickwork_To(t, chs):
    chs=np.asarray(chs)
    if len(chs.shape)==2:
        chs=[chs for _ in range(t)]
    dual=[np.eye(4).reshape((4,1,4,1))]+[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((1,16,16,1)) for ch in chs]
    dual[-1]=np.tensordot(dual[-1].reshape((1,16,4,4,1)),np.eye(2).ravel(),axes=((3,),(0,)))
    To=tt.frommatrices_slice(dual)
    To=To.recluster(((4,1),(1,4))*(2*len(dual)-2)) #TODO: fix in ttarray
    To=To.recluster(((16,16),)*(len(dual)-1))
    return To

def brickwork_Te(t, chs):
    chs=np.asarray(chs)
    if len(chs.shape)==2:
        chs=[chs for _ in range(t)]
    dual=[ch.reshape((4,4,4,4)).transpose([2,0,3,1]).reshape((1,16,16,1)) for ch in chs]
    dual[0]=dual[0].reshape((4,4,4,4)).transpose([2,0,1,3]).reshape((4,16,4,1))
    dual.append(np.eye(2).reshape((1,1,4,1)))
    Te=tt.frommatrices_slice(dual)
    Te=Te.recluster(((4,1),(1,4))*(2*len(dual)-2)) #TODO: fix in ttarray
    Te=Te.recluster(((16,16),)*(len(dual)-1))
    return Te
def brickwork_open_boundary_im(t):
    return brickwork_La(t).asarray()
def interleave_brickwork(ima,imb):
    res=[]
    for l,r in zip(lhs,rhs):
        l=l.reshape((l.shape[0],4,4,l.shape[-1]))
        r=r.reshape((r.shape[0],4,4,r.shape[-1]))
        res.append(np.einsum("abcd,ecfg->aebfdg",l,r).reshape(l.shape[0]*r.shape[0],16,l.shape[-1]*r.shape[-1]))
    return res

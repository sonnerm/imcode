from .channel import im_channel_dense
import numpy as np
from ... import dense
import scipy.sparse as sparse
def boundary_dm_evolution(im,ch,init):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(ch,np.ndarray):
        ch=[ch for _ in range(im.L)]
    for i in range(im.L):
        bimc=im_channel_dense(im,i)
        dmsd=dms[-1].reshape((bimc.shape[1]//4,dim**2))
        dmsd=np.einsum("ab,cb->ca",ch[i],dmsd)
        dms.append(dmsd.ravel())
        dmsd=dmsd.reshape((bimc.shape[1],dim**2//4))
        dmsd=np.einsum("ab,bc->ac",bimc,dmsd)
        dms.append(dmsd.ravel())

    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//(dim**2),dim**2)),axis=0)) for d in dms]

def embedded_dm_evolution(left,ch,right,init):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(ch,np.ndarray):
        ch=[ch for _ in range(left.L)]
    for i in range(left.L):
        limc=im_channel_dense(left,i)
        limc=limc.reshape((limc.shape[0]//4,4,limc.shape[1]//4,4))
        rimc=im_channel_dense(right,i)
        rimc=rimc.reshape((rimc.shape[0]//4,4,rimc.shape[1]//4,4))
        dmsd=dms[-1].reshape((limc.shape[2]*rimc.shape[2],dim**2))
        dmsd=np.einsum("ab,cb->ca",ch[i],dmsd)
        dms.append(dmsd.ravel())
        dmsd=dmsd.reshape((limc.shape[2],rimc.shape[2],4,dim**2//4))
        dmsd=np.einsum("abcd,cfdh->afbh",limc,dmsd)
        dmsd=dmsd.reshape((limc.shape[0],rimc.shape[2],dim**2//4,4))
        dmsd=np.einsum("abcd,ecgd->eagb",rimc,dmsd)
        dms.append(dmsd.ravel())
    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//dim**2,dim**2)),axis=0)) for d in dms]
def boundary_z(im,lop,zs):
    pass
def embedded_z(left,lop,right,zs):
    pass
def boundary_norm(im,lop):
    return boundary_z(im,lop,[(1/2,1/2)]*im.L)
def embedded_norm(left,lop,right):
    return boundary_z(left,lop,right,[(1/2,1/2)]*left.L)

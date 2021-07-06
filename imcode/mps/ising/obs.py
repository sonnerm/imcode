from .channel import im_channel_dense
import numpy as np
from ... import dense
def boundary_dm_evolution(im,lop,init):
    dms=[init.ravel()]
    dim=init.shape[0]
    if isinstance(lop,np.ndarray):
        lop=[lop for _ in range(im.L)]
    for i in range(im.L):
        bimc=im_channel_dense(im,i)
        imc=dense.kron([bimc,np.eye((dim//2)**2)])
        lopc=dense.kron([np.eye(bimc.shape[1]//4),dense.unitary_channel(lop[i])])
        dms.append(lopc@dms[-1])
        dms.append(imc@dms[-1])
    return [np.sum(d.reshape((d.shape[0]//(dim**2),dim,dim)),axis=0) for d in dms]

def embedded_dm_evolution(left,lop,right,init):
    dms=[init]
    if isinstance(lop,np.ndarray):
        lop=[lop for _ in range(im.L)]
    for i in range(im.L):
        limc=im_channel_dense(left,i)
        rimc=im_channel_dense(right,i)
        rimc=rimc.reshape((rimc.shape[0]//4,4,rimc.shape[1]//4,4)).transpose([1,3,0,2]).reshape((rimc.shape[0],rimc.shape[1]))
        imc=dense.kron([limc,np.eye((lop.shape[0]//4)**2),rimc])
        lopc=dense.kron([np.eye(bimc.shape[0]//4),dense.unitary_channel(lop[i]),np.eye(rimc.shape[0]//4)])
        dms.append(lopc@dms[-1])
        dms.append(imc@dms[-1])
    return dms
def boundary_z(im,lop,zs):
    pass
def embedded_z(left,lop,right,zs):
    pass
def boundary_norm(im,lop):
    return boundary_z(im,lop,[(2,2)]*im.L)
def embedded_norm(left,lop,right):
    return boundary_z(left,lop,right,[(2,2)]*left.L)

from .channel import im_channel_dense
import numpy as np
from ... import dense
import scipy.sparse as sparse
def boundary_dm_evolution(im,lop,init):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(lop,np.ndarray):
        lop=[lop for _ in range(im.L)]
    for i in range(im.L):
        bimc=im_channel_dense(im,i)
        imc=dense.kron([bimc,np.eye((dim//2)**2)])
        lopc=dense.kron([np.eye(bimc.shape[1]//4),dense.unitary_channel(lop[i])])
        dms.append(lopc@dms[-1])
        dms.append(imc@dms[-1])
    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//(dim**2),dim**2)),axis=0)) for d in dms]

def embedded_dm_evolution(left,lop,right,init):
    dms=[dense.operator_to_state(init)]
    dim=init.shape[0]
    if isinstance(lop,np.ndarray):
        lop=[lop for _ in range(left.L)]
    for i in range(left.L):
        limc=im_channel_dense(left,i)
        limc=limc.reshape((limc.shape[0]//4,4,limc.shape[1]//4,4))
        rimc=im_channel_dense(right,i)
        rimc=rimc.reshape((rimc.shape[0]//4,4,rimc.shape[1]//4,4))
        if dim==2:
            imc=np.einsum("abcd,ebgd->aebcgd",limc,rimc)
        else:
            imc=np.einsum("abcd,efgh,ij->aebifcgdjh",limc,rimc,np.eye(dim**2//16))
        imc=imc.reshape((limc.shape[0]*rimc.shape[0]*dim**2,limc.shape[2]*rimc.shape[2]*dim**2))
        lopc=sparse.kron(np.eye(imc.shape[1]//dim**2),dense.unitary_channel(lop[i]))
        dms.append(lopc@dms[-1])
        dms.append(imc@dms[-1])
    return [dense.state_to_operator(np.sum(d.reshape((d.shape[0]//dim**2,dim**2)),axis=0)) for d in dms]
def boundary_z(im,lop,zs):
    pass
def embedded_z(left,lop,right,zs):
    pass
def boundary_norm(im,lop):
    return boundary_z(im,lop,[(1/2,1/2)]*im.L)
def embedded_norm(left,lop,right):
    return boundary_z(left,lop,right,[(1/2,1/2)]*left.L)

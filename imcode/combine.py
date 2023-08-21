import numpy as np
import ttarray as tt

_INTER_A=np.zeros((2,2,2))
_INTER_A[0,1,1]=_INTER_A[1,1,0]=_INTER_A[0,0,0]=_INTER_A[1,0,1]=1.0
_INTER_B=np.zeros((2,2,2))
_INTER_B[0,1,0]=_INTER_B[0,0,0]=_INTER_B[1,0,1]=1.0
_INTER_B[1,1,1]=-1.0

def interleave(lhs,rhs,coarse_grain=False,fermionic=False,truncate=True):
    '''
        Interleave two IMs. If `coarse_grain` is set it will performe a
        `coarse_grain` step after interleaving, but consumes less memory then
        constructing the interleaved IM.
        !Important! If you want to describe a fermionic system, this (and most
        other functions manipulating the IM) needs to be applied to the IM in
        fermionic JW order, not in spin IM order and the `fermionic` flag must
        be set!
    '''
    res=[]
    for l,r in zip(lhs.M,rhs.M):
        if coarse_grain:
            l=l.reshape((l.shape[0],4,4,l.shape[-1]))
            r=r.reshape((r.shape[0],4,4,r.shape[-1]))
            res.append(np.einsum("abcd,ecfg->aebfdg",l,r,optimize=True).reshape(l.shape[0]*r.shape[0],16,l.shape[-1]*r.shape[-1]))
        else:
            res.append(np.einsum("abd,ef->aebdf",l,np.eye(r.shape[0]),optimize=True).reshape(l.shape[0]*r.shape[0],16,l.shape[-1]*r.shape[0]))
            res.append(np.einsum("abd,ef->eabfd",r,np.eye(l.shape[-1]),optimize=True).reshape(l.shape[-1]*r.shape[0],16,l.shape[-1]*r.shape[-1]))
    res=tt.frommatrices(res)
    if fermionic:
        Omps=tt.frommatrices([_INTER_A[0,...][None,...]]+[_INTER_A]*3+([_INTER_B]*4+[_INTER_A]*4)*(min(lhs.L,rhs.L)-1)+[_INTER_B]*3+[_INTER_B[...,0][...,None]])
        chi=max(res.chi)
        res=Omps*res
        if truncate:
            res.truncate(chi_max=chi)
    return res

    
def coarse_grain(im,chs=None):
    '''
        Coarse grain the IM by connecting the input and output of two
        successive kernels. Optionally inserts a channel in the connected leg.
    '''
    res=[]
    if chs is not None:
        chs=np.array(chs)
        if len(chs.shape)==2:
            chs=[chs for _ in range(im.L//2)]
        elif len(chs.shape)==3:
            pass
        else:
            raise ValueError("chs must be either 2D (single channel, meant to be repeated) or 3D (sequence of channels) but has dim %i"%len(chs.shape))

        for l,r,c in zip(im.M[::2],im.M[1::2],chs):
            res.append(np.einsum("abcd,defg,ec->abfg",l.reshape((l.shape[0],4,4,l.shape[-1])),r.reshape((r.shape[0],4,4,r.shape[-1])),c,optimize=True))
    else:
        for l,r in zip(im.M[::2],im.M[1::2]):
            res.append(np.einsum("abcd,dcfg->abfg",l.reshape((l.shape[0],4,4,l.shape[-1])),r.reshape((r.shape[0],4,4,r.shape[-1])),optimize=True))
    if im.L%2==1:
        res.append(im.M[-1])
    return tt.frommatrices(res)


    
def fine_grain(im,factor=2):
    '''
    Fine grains IM by finding a possible interpolation with intermediate time-steps using matrix_power.
    The result of this function is gauge dependend and if the input is not in a `channel` form, the ouput
    might violate IM invariants such as CPTP
    !Experimental!
    '''
    raise NotImplementedError("Not here yet")
    pass

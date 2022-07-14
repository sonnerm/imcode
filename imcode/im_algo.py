import ttarray as tt
import numpy as np
import math
import itertools
def _generator_matrices(mps):
    if isinstance(mps,tt.TensorTrainArray):
        mps=mps.toslice()
    elif len(mps.shape)==2:
        mps=mps[None,...,None]
        mps=tt.slice(mps)
    else:
        mps=tt.slice(mps)
    assert int(math.log2(mps.shape[1]))==math.log2(mps.shape[1])
    assert mps.shape[1]==mps.shape[2]
    mps=mps.recluster(((2,2),)*int(math.log2(mps.shape[1])))
    return itertools.cycle(mps.tomatrices_unchecked())




def brickwork_lcga(Ts,init=np.eye(2)/2,boundary=None,chi_max=128,cutoff=1e-12,yieldcopy=True):
    '''
        Implements the light-cone growth algorithm for brickwork like circuits.
        works like a generator of intermediate influence matrices
        init defaults to infinite temperature
    '''
    bwobim=np.eye(4)
    gene=_generator_matrices(init)
    if boundary is None:
        cmps=tt.fromproduct([bwobim]) # empty ttarrays are not allowed
    else:
        cmps=boundary.copy()
    for T in Ts:
        # augment
        tdim=int(math.log2(T.shape[1]))//4 #math not numpy since the dimension can be quite large
        cdim=int(math.log2(cmps.shape[0]))//4 #math not numpy since the dimension can be quite large
        if tdim>cdim:
            cmps=tt.frommatrices_unchecked(cmps.tomatrices_unchecked()+[bwobim for _ in range(tdim-cdim)])
        # contract with initial
        T.tomatrices_unchecked([0])
        init=np.array(tt.frommatrices_slice(itertools.take(gene)))[None,...].transpose([0,1,3,2])
        T=tt.frommatrices([init]+T.tomatrices_unchecked())
        # apply
        cmps=T@cmps
        # truncate
        cmps.truncate(chi_max=chi_max,cutoff=cutoff)
        if yieldcopy:
            yield cmps.copy()
        else:
            yield cmps

def zoz_lcga(Ts,init=np.eye(2)/2,boundary=None,chi_max=128,cutoff=1e-12,yieldcopy=True):
    '''
        Implements the light-cone growth algorithm for zoz style circuits
    '''
    zozobim=np.array([1,0,0,1])[None,...,None]
    gene=_generator_matrices(init)
    if boundary is None:
        cmps=tt.frommatrices([zozobim]) # empty ttarrays are not allowed
    else:
        cmps=boundary.copy()
    for T in Ts:
        # augment
        # contract with initial
        init=next(gene)
        init=init.reshape((1,init.shape[0],4,init.shape[-1])).transpose([0,3,1,2])
        T=tt.frommatrices([init]+T.tomatrices_unchecked())

        tdim=int(math.log2(T.shape[1]//init.shape[2]))//2 #math not numpy since the dimension can be quite large
        cdim=int(math.log2(cmps.shape[0]//init.shape[2]))//2 #math not numpy since the dimension can be quite large
        if tdim>cdim:
            cmps=tt.frommatrices(cmps.tomatrices_unchecked()+[zozobim for _ in range(tdim-cdim)])
        # apply
        cmps=T@cmps
        # truncate
        cmps.truncate(chi_max=chi_max,cutoff=cutoff)
        if yieldcopy:
            yield cmps.copy()
        else:
            yield cmps

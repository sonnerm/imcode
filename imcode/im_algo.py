import ttarray as tt
import math
def brickwork_lcga(Ts,init=None,boundary=None,chi_max=128,cutoff=1e-12,yieldcopy=True):
    '''
        Implements the light-cone growth algorithm for brickwork like circuits.
        works like a generator of intermediate influence matrices
        init defaults to infinite temperature
    '''
    bwobim=np.eye(4)
    if boundary is None:
        cmps=tt.fromproduct([bwobim]) # empty ttarrays are not allowed
    else:
        cmps=boundary.copy()
    for T in Ts:
        # augment
        tdim=int(math.log2(T.shape[1]))//4 #math not numpy since the dimension can be quite large
        cdim=int(math.log2(cmps.shape[0]))//4 #math not numpy since the dimension can be quite large
        if tdim>cdim:
            cmps=tt.frommatrices_unchecked(cmps.asmatrices_unchecked()+[bwobim for _ in range(tdim-cdim)])
        # apply
        cmps=T@cmps
        # truncate
        cmps.truncate(chi_max=chi_max,cutoff=cutoff)
        if yieldcopy:
            yield cmps.copy()
        else:
            yield cmps

def zoz_lcga(Ts,init,boundary=None,chi_max=128,cutoff=1e-12,yieldcopy=True):
    '''
        Implements the light-cone growth algorithm for zoz style circuits
    '''
    zozobim=np.ones((4,))
    if boundary is None:
        cmps=tt.fromproduct([zozobim]) # empty ttarrays are not allowed
    else:
        cmps=boundary.copy()
    for T in Ts:
        # augment
        tdim=int(math.log2(T.shape[1]))//2 #math not numpy since the dimension can be quite large
        cdim=int(math.log2(cmps.shape[0]))//2 #math not numpy since the dimension can be quite large
        if tdim>cdim:
            cmps=tt.frommatrices_unchecked(cmps.asmatrices_unchecked()+[zozobim for _ in range(tdim-cdim)])
        # apply
        cmps=T@cmps
        # truncate
        cmps.truncate(chi_max=chi_max,cutoff=cutoff)
        if yieldcopy:
            yield cmps.copy()
        else:
            yield cmps

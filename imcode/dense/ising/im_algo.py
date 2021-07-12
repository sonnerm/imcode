from . import open_boundary_im
from .. import outer
import numpy as np
def im_rectangle(Ts,boundary=None):
    if isinstance(Ts,np.ndarray) and len(Ts.shape)==2:
        Ts=[Ts]*int(np.log2(Ts.shape[0]))
    if boundary is None:
        boundary=open_boundary_im(int(np.log2(Ts[0].shape[0]))//2)
    im=boundary
    yield im
    for T in Ts:
        im=T@im
        yield im
def im_diamond(Ts):
    imit=[1,0,0,1]
    im=None
    for T in Ts:
        if im is None:
            im=imit
        else:
            im=outer([imit,im,imit])
        im=T@im
        yield im
def im_triangle(Ts):
    imit=[1,0,0,1]
    im=None
    for T in Ts:
        if im is None:
            im=imit
        else:
            im=outer([im,imit])
        im=T@im
        yield im

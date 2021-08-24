from . import brickwork_La
from .. import outer
import numpy as np
def im_rectangle(Sas,Sbs,boundary=None,reversed=False):
    if isinstance(Sas,np.ndarray) and len(Sas.shape)==2:
        Sas=[Sas]*int(np.log2(Sas.shape[0]))
    if isinstance(Sbs,np.ndarray) and len(Sbs.shape)==2:
        Sbs=[Sbs]*int(np.log2(Sbs.shape[0]))
    if boundary is None:
        boundary=brickwork_La(int(np.log2(Ts[0].shape[0]))//2)
    im=boundary
    yield im
    for Sa,Sb in zip(Sas,Sbs):
        im=Sb@im
        im=Sa@im
        yield im
def im_diamond(Sas,inits=itertools.repeat(np.array([1,0,0,1])),finals=itertools.repeat(np.array([1,0,0,1]))):
    im=None
    for Sa,i,f in zip(Sas,inits,finals):
        if im is None:
            im=outer([i,f])
        else:
            im=outer([i,im,f])
        im=Sa@im
        yield im
def im_triangle(Sas,Sbs,boundary=None):
    if boundary==None:
        boundary=np.array(1.0)
    im=boundary
    for Sa,Sb in zip(Sas,Sbs):
        im=outer([i,brickwork_La(1)])
        im=Sb@im
        im=Sa@im
        yield im

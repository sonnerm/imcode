from . import brickwork_La
from .. import outer
import itertools
import numpy as np
import numpy.linalg as la
import imcode.dense

def im_rectangle(Sas,Sbs,boundary=None,reversed=False):
    if isinstance(Sas,np.ndarray) and len(Sas.shape)==2:
        Sas=[Sas]*(int(np.log2(Sas.shape[0]))//4)
    if isinstance(Sbs,np.ndarray) and len(Sbs.shape)==2:
        Sbs=[Sbs]*(int(np.log2(Sbs.shape[0]))//4)
    if boundary is None:
        boundary=brickwork_La(int(np.log2(Sas[0].shape[0]))//4)
    im=boundary
    yield im
    for Sa,Sb in zip(Sas,Sbs):
        im=Sb@im
        im=Sa@im
        yield im
def im_diamond(Sas):
    im=None
    for Sa in Sas:
        if im is None:
            im=outer([np.array([1,0,0,1])/2,np.array([1,0,0,1])])
        else:
            im=outer([np.array([1,0,0,1])/2,im,np.array([1,0,0,1])])
        im=Sa@im
        yield im
def im_triangle(Sas,Sbs,boundary=None):
    if boundary==None:
        boundary=np.array(1.0)
    im=boundary
    for Sa,Sb in zip(Sas,Sbs):
        im=outer([im,brickwork_La(1)])
        im=Sb@im
        im=Sa@im
        yield im
def im_diag(Sa,Sb):
    T=Sa@Sb
    ev,evv=la.eig(T)
    oev=evv[:,np.argmax(np.abs(ev))]
    normi=outer([np.array([1,0,0,1])]*(int(np.log2(oev.shape[0]))//2))
    oev/=oev@normi
    oev*=oev.shape[0]**(1/4)
    return oev

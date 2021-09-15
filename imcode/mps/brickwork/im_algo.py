from . import brickwork_La
from .. import MPO
from .. import outer
import itertools
import numpy as np
import numpy.linalg as la
import imcode.dense

def im_rectangle(Sas,Sbs,boundary=None,reversed=False,**kwargs):
    if isinstance(Sas,MPO):
        Sas=[Sas]*(Sas.L+1)
    if isinstance(Sbs,MPO):
        Sbs=[Sbs]*(Sbs.L+1)
    if boundary is None:
        boundary=brickwork_La(Sas[0].L)
    im=boundary
    yield im.copy()
    for Sa,Sb in zip(Sas,Sbs):
        if not reversed:
            T=(Sa@Sb).contract()
        else:
            T=(Sb@Sa).contract()
        im=(T@im).contract(**kwargs)
        yield im.copy()
# def im_diamond(Sas,**kwargs):
#     im=None
#     for Sa in Sas:
#         if im is None:
#             im=outer([np.array([1,0,0,1])/2,np.array([1,0,0,1])])
#         else:
#             im=outer([np.array([1,0,0,1])/2,im,np.array([1,0,0,1])])
#         im=Sa@im
#         yield im
def im_triangle(Sas,Sbs,boundary=None,**kwargs):
    if boundary==None:
        im=brickwork_La(1)
    else:
        im=outer(boundary,brickwork_La(1))
    for Sa,Sb in zip(Sas,Sbs):
        T=(Sa@Sb).contract()
        im=(T@im).contract(**kwargs)
        yield im.copy()
        im=outer([im,brickwork_La(1)])

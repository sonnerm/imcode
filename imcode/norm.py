import numpy as np
import math 
def ising_norm(im,initbc=None):
    if initbc is None:
        initbc=np.array([1.0])
    im=im.recluster(((initbc.shape[0],),)+((4,),)*(int(math.log2(im.shape[0]//initbc.shape[0]))//2))
    ms=im.tomatrices()
    cvec=np.tensordot(ms[0],initbc,axes=((1,),(0,)))[0]
    for m in ms[1:]:
        m=np.tensordot(m,cvec,axes=((0,),(0,)))
        cvec=np.tensordot(m,np.array([0.5,0,0,0.5]),axes=((0,),(0,)))
    return cvec[0]


def zoz_norm(im,initbc=None):
    pass

def brickwork_norm(im,initbc=None):
    if initbc is None:
        initbc=np.array([1.0])
    im=im.recluster(((initbc.shape[0],),)+((16,),)*(int(math.log2(im.shape[0]//initbc.shape[0]))//4))
    ms=im.tomatrices()
    cvec=np.tensordot(ms[0],initbc,axes=((1,),(0,)))[0]
    for m in ms[1:]:
        m=np.tensordot(m,cvec,axes=((0,),(0,)))
        m=np.tensordot(m.reshape((4,4,m.shape[-1])),np.array([0.5,0,0,0.5]),axes=((0,),(0,)))
        cvec=np.tensordot(m,np.array([1,0,0,1]),axes=((0,),(0,)))
    return cvec[0]

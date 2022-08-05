import numpy as np
def ising_norm(im,initbc=None):
    cvec=np.tensordot(im.M[0],initbc,axes=((1,),(0,)))[0]
    for m in im.M[1:]:
        m=np.tensordot(m,cvec,axes=((0,),(0,)))
        cvec=np.tensordot(m,np.array([0.5,0,0,0.5]),axes=((0,),(0,)))
    return cvec[0]


def zoz_norm(im,initbc=None):
    pass

def brickwork_norm(im,initbc=None):
    cvec=np.tensordot(im.M[0],initbc,axes=((1,),(0,)))[0]
    for m in im.M[1:]:
        m=np.tensordot(m,cvec,axes=((0,),(0,)))
        m=np.tensordot(m,np.array([0.5,0,0,0.5]),axes=((0,),(0,)))
        cvec=np.tensordot(m,np.array([1,0,0,1]),axes=((0,),(0,)))
    return cvec[0]

from .utils import apply
from .fold import open_boundary_im
def im_iterative(mpo,chi=None,options=None):
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options)

def im_finite(Ts,boundary=None,chi=None,options=None):
    t=Ts[0].L-1
    if boundary is None:
        vec=open_boundary_im(t)
    else:
        vec=boundary.copy()
    for T in Ts:
        apply(T,vec,chi,options)
    return vec

def im_dmrg(mpo,chi,initial=None,options=None):
    pass

from .utils import apply
from . import fold
from . import flat
def im_iterative(mpo,chi=None,options=None):
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options)

def im_finite(Ts,boundary=None,chi=None,options=None):
    t=Ts[0].L-1
    if boundary is None:
        if isinstance(Ts[0].sites[0],fold.FoldSite):
            vec=fold.open_boundary_im(t)
        elif isinstance(Ts[0].sites[0],flat.FlatSite):
            vec=flat.open_boundary_im(t)
        else:
            assert False
    else:
        vec=boundary.copy()
    for T in Ts:
        apply(T,vec,chi,options)
    return vec

def im_dmrg(mpo,chi,initial=None,options=None):
    pass

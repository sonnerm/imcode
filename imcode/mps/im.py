from .utils import apply
from . import fold
from . import flat
def im_iterative(mpo,chi=None,options=None,boundary=None):
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options,boundary=boundary)

def im_zipup(mpo,chi):
    options={"trunc_params":{"chi_max":chi},"m_temp":4,"verbose":False,"compression_method":"zip_up"}
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options)

def im_finite(Ts,boundary=None,chi=None,options=None):
    if boundary is None:
        if isinstance(Ts[0].sites[0],fold.FoldSite):
            t=Ts[0].L-1
            vec=fold.perfect_dephaser_im(t)
        # elif isinstance(Ts[0].sites[0],flat.FlatSite):
            # t=Ts[0].L//2
            # vec=flat.perfect_dephaser_im(t)
        else:
            assert False
    else:
        vec=boundary.copy()
    for T in Ts:
        apply(T,vec,chi,options)
    return vec

def im_dmrg(mpo,chi,initial=None,options=None):
    pass

from .. import MPS
def im_iterative(mpo,chi=None,options=None,boundary=None):
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options,boundary=boundary)

def im_zipup(mpo,chi):
    options={"trunc_params":{"chi_max":chi},"m_temp":4,"verbose":False,"compression_method":"zip_up"}
    return im_finite([mpo]*(2*(mpo.L-1)),chi=chi,options=options)

def im_finite(Ts,boundary=None,chi=None,options=None):
    if boundary is None:
        if isinstance(Ts[0].sites[0],fold.FoldSite):
            t=Ts[0].L-1
            vec=perfect_dephaser_im(t)
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
def im_triangle(Ts,chi=None,options=None):
    mps=MPS.from_product_state([[1,1,0,0]])
    for T in Ts:
        mps=expand_im(mps)
        apply(T,mps,chi,options)
    return mps


def im_dmrg(mpo,chi,initial=None,options=None):
    pass
def open_boundary_im(t):
    state = [[1,1,1,1]]+[[1,1,1,1]] * (len(sites)-2)+[[1,1,1,1]]
    psi = MPS.from_product_state(state)
    return psi
def perfect_dephaser_im(t):
    state = [[1,0,0,1]] * len(sites)
    psi = MPS.from_product_state(state)
    return psi
def dephaser_im(t,gamma=1):
    state = [[1,1-gamma,1-gamma,1]]+[[1,1-gamma,1-gamma,1]] * (len(sites)-2)+[[1,0,0,1]]
    psi = MPS.from_product_state(state)
    return psi

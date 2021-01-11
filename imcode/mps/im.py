from .utils import apply,BlipSite
from tenpy.networks.mps import MPS
def open_boundary_im(t):
    sites=[BlipSite() for _ in range(t+1)]
    state = [[1,1,0,0]]+[[1,1,1,1]] * (len(sites)-2)+[[1,1,0,0]]
    psi = MPS.from_product_state(sites, state)
    return psi

def perfect_dephaser_im(t):
    sites=[BlipSite() for _ in range(t+1)]
    state = [[1,1,0,0]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi
def im_iterative(mpo,chi=None,options=None):
    return im_finite([mpo]*(2*(mpo.L-1)),chi,options)

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

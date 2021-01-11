from .utils import apply,BlipSite
from tenpy.networks.mps import MPS
def open_boundary(t):
    sites=[BlipSite() for _ in range(t+1)]
    state = [[1,1,1,1]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi
def im_iterative(mpo,chi=None):
    pass

def im_finite(Ts,boundary=None,chi=None,options=None):
    t=Ts[0].L-1
    if boundary is None:
        vec=open_boundary(t)
    else:
        vec=boundary.copy()
    for T in Ts:
        apply(T,vec,chi,options)
    return vec

def im_dmrg(mpo,chi=None,initial=None):
    pass

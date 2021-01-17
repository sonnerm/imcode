from .utils import FoldSite
from tenpy.networks.mps import MPS
def open_boundary_im(t):
    sites=[FoldSite() for _ in range(t+1)]
    state = [[1,1,0,0]]+[[1,1,1,1]] * (len(sites)-2)+[[1,1,0,0]]
    psi = MPS.from_product_state(sites, state)
    return psi
def perfect_dephaser_im(t):
    sites=[FoldSite() for _ in range(t+1)]
    state = [[1,1,0,0]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi
def dephaser_im(t,gamma=1):
    sites=[FoldSite() for _ in range(t+1)]
    state = [[1,1,0,0]]+[[1,1,1-gamma,1-gamma]] * (len(sites)-2)+[[1,1,0,0]]
    psi = MPS.from_product_state(sites, state)
    return psi

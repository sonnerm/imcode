from .utils import FoldSite
from .. import MPS
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

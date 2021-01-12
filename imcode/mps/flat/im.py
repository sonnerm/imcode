from .utils import FlatSite
from tenpy.networks.mps import MPS
def open_boundary_im(t):
    sites=[FlatSite() for _ in range(2*t)]
    state = [[1,1]*(2*t)]
    psi = MPS.from_product_state(sites, state)
    return psi

def perfect_dephaser_im():
    #not at all trivial 'rainbow' state
    assert False
    pass

from .. import MPS
from . import open_boundary_im
def im_rectangle(Ts,boundary=None,chi=None,options=None):
    if boundary is None:
        boundary=open_boundary_im(t)
    mps=boundary
    pass
def im_diamond(Ts,chi=None,options=None):
    mps=MPS.from_product_state([[1,0,0,1]])
    pass
def im_triangle(Ts,chi=None,options=None):
    mps=MPS.from_product_state([[1,0,0,1]])
    pass

def open_boundary_im(t):
    state = [[1,1,1,1]]*t
    psi = MPS.from_product_state(state)
    return psi
def perfect_dephaser_im(t):
    state = [[1,0,0,1]] * t
    psi = MPS.from_product_state(state)
    return psi
def dephaser_im(t,gamma=1):
    state = [[1,1-gamma,1-gamma,1]]*t
    psi = MPS.from_product_state(state)
    return psi

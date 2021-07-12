from . import open_boundary_im
from .. import outer,MPS
from collections.abc import Iterable
def im_rectangle(Ts,boundary=None,chi=None,options=None):
    if not isinstance(Ts,Iterable):
        Ts=[Ts]*(2*Ts.L)
    if boundary is None:
        boundary=open_boundary_im(Ts[0].L)
    mps=boundary
    yield mps.copy()
    for T in Ts:
        mps=(T@mps).contract(chi=chi,options=options)
        yield mps.copy()
def im_diamond(Ts,chi=None,options=None):
    mpsit=MPS.from_product_state([[1,0,0,1]])
    mps=None
    for T in Ts:
        if mps is None:
            mps=mpsit
        else:
            mps=outer([mpsit,mps,mpsit])
        mps=(T@mps).contract(chi=chi,options=options)
        yield mps
def im_triangle(Ts,chi=None,options=None):
    mps=MPS.from_product_state([[1,0,0,1]])
    mps=None
    for T in Ts:
        if mps is None:
            mps=mpsit
        else:
            mps=outer([mps,mpsit])
        mps=(T@mps).contract(chi=chi,options=options)
        yield mps

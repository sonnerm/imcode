from . import open_boundary_im
from .. import outer,MPS
from collections.abc import Iterable
def im_rectangle(Ts,boundary=None,**kwargs):
    if not isinstance(Ts,Iterable):
        Ts=[Ts]*(2*Ts.L)
    if boundary is None:
        boundary=open_boundary_im(Ts[0].L)
    mps=boundary
    yield mps.copy()
    for T in Ts:
        mps=(T@mps).contract(**kwargs)
        yield mps.copy()
def im_diamond(Ts,boundary=None,**kwargs):
    mpsit=MPS.from_product_state([[1,0,0,1]])
    mps=boundary
    for T in Ts:
        if mps is None:
            mps=mpsit.copy()
        else:
            mps=outer([mpsit,mps,mpsit])
        mps=(T@mps).contract(**kwargs)
        yield mps.copy()
def im_triangle(Ts,boundary=None,**kwargs):
    mpsit=MPS.from_product_state([[1,0,0,1]])
    mps=boundary
    for T in Ts:
        if mps is None:
            mps=mpsit.copy()
        else:
            mps=outer([mps,mpsit])
        mps=(T@mps).contract(**kwargs)
        yield mps.copy()

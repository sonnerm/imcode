from ..dense import dephaser_im
from .utils import DiagonalLinearOperator
def dephaser_operator(t,gamma):
    return DiagonalLinearOperator(dephaser_im(t,gamma))

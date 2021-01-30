from .ising import ising_J,ising_W,ising_h
from .ising import hr_operator,ising_hr_T
# from .ising_Jr import Jr_operator,ising_Jr_T,ising_Jhr_T
from .dissipation import dephaser_operator
from .utils import DiagonalLinearOperator

def ising_dephase_T(t,J,g,h,gamma):
    U1=DiagonalLinearOperator(ising_W(t,g).diag*ising_h(t,h).diag*dephaser_operator(t,gamma).diag)
    U2=ising_J(t,J)
    return U2@U1
def ising_dephase_hr_T(t,J,g,gamma):
    U1=DiagonalLinearOperator(hr_operator(t).diag*ising_W(t,g).diag*dephaser_operator(t,gamma).diag)
    U2=ising_J(t,J)
    return U2@U1
def ising_dephase_hr_Tp(t,J,g,gamma):
    U1=DiagonalLinearOperator(ising_W(t,g).diag*dephaser_operator(t,gamma).diag)
    U2=ising_J(t,J)
    Up=hr_operator(t)
    return Up@U2@U1
# def ising_dephase_Jr_T(t,g,h,gamma):
#     U1=DiagonalLinearOperator(ising_h(t,h).diag*ising_W(T,g).diag*dephaser_operator(t,gamma).diag)
#     U2=Jr_operator(t)
#     return U2@U1
# def ising_dephase_Jhr_T(t,g,gamma):
#     U1=DiagonalLinearOperator(hr_operator(t).diag*ising_W(T,g).diag*dephaser_operator(t,gamma).diag)
#     U2=Jr_operator(t)
#     return U2@U1
# def ising_dephase_Jhr_Tp(t,g,gamma):
#     U1=DiagonalLinearOperator(ising_W(T,g).diag*dephaser_operator(t,gamma).diag)
#     U2=Jr_operator(t)
#     Up=hr_operator(t)
#     return Up@U2@U1

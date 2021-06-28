from .dissipation import dephaser_operator,depolarizer_operator
from .ising_hr import ising_hr_T,ising_hr_Tp
from .ising import ising_T
# from .ising_Jr import ising_Jhr_T,ising_Jhr_Tp
def ising_dephase_T(t,J,g,h,gamma):
    return (ising_T(t,J,g,h)@dephaser_operator(t,gamma)).contract()
def ising_dephase_hr_T(t,J,g,gamma):
    return (ising_hr_T(t,J,g)@dephaser_operator(t,gamma)).contract()
def ising_dephase_hr_Tp(t,J,g,gamma):
    return (ising_hr_Tp(t,J,g)@dephaser_operator(t,gamma)).contract()
def ising_dephase_Jr_T(t,g,h,gamma):
    return (ising_Jr_T(t,g,h)@dephaser_operator(t,gamma)).contract()
def ising_dephase_Jhr_T(t,g,gamma):
    return (ising_Jhr_T(t,g)@dephaser_operator(t,gamma)).contract()
def ising_dephase_Jhr_Tp(t,J,g,gamma):
    return (ising_Jhr_Tp(t,g)@dephaser_operator(t,gamma)).contract()


def ising_depolarize_T(t,J,g,h,gamma):
    return (ising_T(t,J,g,h)@depolarizer_operator(t,gamma)).contract()
def ising_depolarize_hr_T(t,J,g,gamma):
    return (ising_hr_T(t,J,g)@depolarizer_operator(t,gamma)).contract()
def ising_depolarize_hr_Tp(t,J,g,gamma):
    return (ising_hr_Tp(t,J,g)@depolarizer_operator(t,gamma)).contract()
def ising_depolarize_Jr_T(t,g,h,gamma):
    return (ising_Jr_T(t,g,h)@depolarizer_operator(t,gamma)).contract()
def ising_depolarize_Jhr_T(t,g,gamma):
    return (ising_Jhr_T(t,g)@depolarizer_operator(t,gamma)).contract()
def ising_depolarize_Jhr_Tp(t,J,g,gamma):
    return (ising_Jhr_Tp(t,g)@depolarizer_operator(t,gamma)).contract()

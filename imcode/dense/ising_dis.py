from .ising import ising_J,ising_W,ising_h
from .ising_hr import hr_operator,ising_hr_T
from .ising_Jr import Jr_operator,ising_Jr_T,ising_Jhr_T
from .dissipation import dephaser_operator,depolarizer_operator

def ising_dephase_T(t,J,g,h,gamma,init=(0.5,0.5)):
    U1=ising_W(t,g,init)*ising_h(t,h)*dephaser_operator(t,gamma)
    U2=ising_J(t,J)
    return U2@U1
def ising_dephase_hr_T(t,J,g,gamma,init=(0.5,0.5)):
    U1=hr_operator(t)*ising_W(t,g,init)*dephaser_operator(t,gamma)
    U2=ising_J(t,J)
    return U2@U1
def ising_dephase_hr_Tp(t,J,g,gamma,init=(0.5,0.5)):
    U1=ising_W(t,g,init)*dephaser_operator(t,gamma)
    U2=ising_J(t,J)
    Up=hr_operator(t)
    return Up@U2@U1
def ising_dephase_Jr_T(t,g,h,gamma,init=(0.5,0.5)):
    U1=ising_h(t,h)*ising_W(T,g,init)*dephaser_operator(t,gamma)
    U2=Jr_operator(t)
    return U2@U1
def ising_dephase_Jhr_T(t,g,gamma,init=(0.5,0.5)):
    U1=hr_operator(t)*ising_W(T,g,init)*dephaser_operator(t,gamma)
    U2=Jr_operator(t)
    return U2@U1
def ising_dephase_Jhr_Tp(t,g,gamma,init=(0.5,0.5)):
    U1=ising_W(T,g,init)*dephaser_operator(t,gamma)
    U2=Jr_operator(t)
    Up=hr_operator(t)
    return Up@U2@U1



def ising_depolarize_T(t,J,g,h,gamma,init=(0.5,0.5)):
    return ising_T(t,J,g,h,init)@depolarizer_operator(t,gamma)
def ising_depolarize_hr_T(t,J,g,gamma,init=(0.5,0.5)):
    return ising_hr_T(t,J,g,init)@depolarizer_operator(t,gamma)
def ising_depolarize_hr_Tp(t,J,g,gamma,init=(0.5,0.5)):
    return ising_hr_Tp(t,J,g,init)@depolarizer_operator(t,gamma)
def ising_depolarize_Jr_T(t,g,h,gamma,init=(0.5,0.5)):
    return ising_Jr_T(t,J,g,init)@depolarizer_operator(t,gamma)
def ising_depolarize_Jhr_T(t,g,gamma,init=(0.5,0.5)):
    return ising_Jhr_T(t,g,init)@depolarizer_operator(t,gamma)
def ising_depolarize_Jhr_Tp(t,g,gamma,init=(0.5,0.5)):
    return ising_Jhr_Tp(t,g,init)@depolarizer_operator(t,gamma)

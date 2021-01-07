from .utils import dense_kron,SX,SZ,ID
import numpy as np
import scipy.linalg as scla
def ising_H(J,g,h):
    '''
        Construct a dense Hamiltonian of a spin 1/2 Ising ring with parameters given by the arrays J,g,h.
        H=\sum_i J_i s^z_{i+1}s^z_{i} + \sum_i h_i s^z_i + \sum_i g_i s^x_i
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    L=len(h) # maybe change to explicit length?
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    ret=np.zeros((2**L,2**L),dtype=np.common_type(J,g,h,np.array(1.0)))
    for i,Jv in enumerate(J[:L-1]):
        ret+=Jv*dense_kron([ID]*i+[SZ]+[SZ]+[ID]*(L-i-2))
    if len(J)==L and L>1:
        ret+=J[-1]*dense_kron([SZ]+[ID]*(L-2)+[SZ])
    for i,hv in enumerate(h):
        ret+=hv*dense_kron([ID]*i+[SZ]+[ID]*(L-i-1))
    for i,gv in enumerate(g):
        ret+=gv*dense_kron([ID]*i+[SX]+[ID]*(L-i-1))
    return ret
def ising_F(J,g,h):
    '''
        Constructs a dense one period time evolution operator for the kicked ising chain model
        F = \exp(i\sum_i J_i s_i^zs_{i+1}^z + h_i s_i^z)\exp(i \sum_i g_i s^x_i)
        (s^x, s^z are Pauli matrices)
        length is taken from the size of h, J can be either the same length
        (open boundary condition) or one element shorter (periodic boundary conditions).
    '''
    L=len(h) #maybe change to explicit length
    return scla.expm(1.0j*ising_H(J,[0.0]*L,h))@scla.expm(1.0j*ising_H([0.0]*L,g,[0.0]*L))


def ising_T(T,J,g,h):
    '''
        Calculate a dense IM approach transfer matrix for the kicked ising
        chain. See arXiv:2009.10105 for details. The sites are ordered as
        follows: [s_0 forward s_T backward].
    '''
    h=np.array([0]+[h]*(T-1)+[0]+[-np.array(h).conj()]*(T-1))
    Jt=-np.pi/4-np.log(np.tan(g))*0.5j
    eta=np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
    Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
    U1=np.diag(np.exp(-1.0j*np.array(imbrie_H(Jt,np.zeros_like(h),h).diagonal())))
    U1*=np.exp(eta.real*(2*T))
    Pm=np.array([[1,1],[1,1]])
    Tm1=np.array([[np.exp(1.0j*J),np.exp(-1.0j*J)],[np.exp(-1.0j*J),np.exp(1.0j*J)]])
    Tm2=Tm1.conj()
    U2=ising_J(T,J)
    return U1@U2
def ising_J(T,J):
    return dense_kron([Pm]+[Tm1]*(T-1)+[Pm]+[Tm2]*(T-1))/2

def ising_h(T,h):
    pass
# def ising_W(T,g):
#     Jt=-np.pi/4-np.log(np.tan(g))*0.5j
#     eta=np.pi/4.0j+np.log(np.sin(g))/2+np.log(np.cos(g))/2
#     Jt=np.array([4*Jt]*(T)+[-4*Jt.conj()]*T)
#     U1=np.diag(np.exp(-1.0j*np.array(imbrie_H(Jt,np.zeros_like(h),h).diagonal())))
#     U1*=np.exp(eta.real*(2*T))
#     return U1

def hr_operator(T):
    pass

def ising_hr_T(T,J,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism described in arXiv:2012.00777. The averaging
        is performed over parameter h. Site ordering as in ising_T.
    '''
    ret=ising_T(T,J,0.0,g)
    return ret@hr(T,False)

def Jr_operator(T):
    pass
def ising_Jr_T(T,g,h):

    '''
        Calculate a dense spatial transfer matrix for the J disorder averaged
        influence matrix formalism similar to arXiv:2012.00777
        Site ordering as in ising_T.
    '''
    #TODO add new ref if available
    pass
def ising_Jhr_T(T,g):
    '''
        Calculate a dense spatial transfer matrix for the disorder averaged
        influence matrix formalism with averaging over both J and h.
        Site ordering as in ising_T.
    '''
    #TODO add new ref if available
    pass

def ising_SFF(T,J,g,h):
    pass
def ising_hr_SFF(T,J,g,h):
    pass

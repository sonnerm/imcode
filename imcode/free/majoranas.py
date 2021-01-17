import numpy as np
import scipy.linalg as la
import functools

def maj_to_quad(M):
    #converts a quadratic operator from majorana single body to many body form
    L=M.shape[0]//2
    assert M.shape[0]%2==0
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret+=me(L,i)@me(L,j)*M[2*i,2*j]
            ret+=me(L,i)@mo(L,j)*M[2*i,2*j+1]
            ret+=mo(L,i)@me(L,j)*M[2*i+1,2*j]
            ret+=mo(L,i)@mo(L,j)*M[2*i+1,2*j+1]
    return ret/4

def quad_to_maj(M):
    #converts a quadratic operator to majorana single body form
    L=int(round(np.log(M.shape[0])/np.log(2)))
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j))/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j))/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j))/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j))/2**L
    return ret*2#-np.diag(np.diag(ret))*(2*L-1)/(2*L)

def exp_to_maj(M):
    #converts an exponentiated quadratic operator to majorana single body form
    L=int(round(np.log(M.shape[0])/np.log(2)))
    ret=np.zeros((2*L,2*L),dtype=complex)
    Mi=la.inv(M)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=np.trace(me(L,i)@M@me(L,j)@Mi)/2**L
            ret[2*i+1,2*j]+=np.trace(mo(L,i)@M@me(L,j)@Mi)/2**L
            ret[2*i,2*j+1]+=np.trace(me(L,i)@M@mo(L,j)@Mi)/2**L
            ret[2*i+1,2*j+1]+=np.trace(mo(L,i)@M@mo(L,j)@Mi)/2**L
    return ret
def exp_to_maj_new(M):
    return quad_to_maj(M)/np.trace(M)
# def exp_to_maj_alt(M):
#     return la.expm(2*quad_to_maj(la.logm(M))) #for now

def maj_to_exp(M):
    #converts an majorana matrix to the corresponding exponentiated quadratic operator
    return la.expm(maj_to_quad(la.logm(M))) #for now
def vac(L):
    ret=np.zeros((2**L,))
    ret[-1]=1
    return ret

def gauss_to_full(M):
    #create a many body gaussian state from a majorana matrix
    return la.expm(maj_to_quad(M))@vac(M.shape[0]//2)
def full_to_gauss(s):
    pass

def projectors_eo(L):
    return ((np.eye(2**L)+dense_kron([sx]*L))/2,(np.eye(2**L)-dense_kron([sx]*L))/2)

def calc_maj_gs(M,par):
    pass
def calc_gauss_corr(M):
    pass
def calc_full_corr(state):
    L=int(round(np.log(state.shape[0])/np.log(2)))
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(L):
        for j in range(L):
            ret[2*i,2*j]+=state.T.conj()@(me(L,i)@(me(L,j)@state))
            ret[2*i+1,2*j]+=state.T.conj()@(mo(L,i)@(me(L,j)@state))
            ret[2*i,2*j+1]+=state.T.conj()@(me(L,i)@(mo(L,j)@state))
            ret[2*i+1,2*j+1]+=state.T.conj()@(mo(L,i)@(mo(L,j)@state))
    return ret
def calc_corr_ent(corr):
    pass

def calc_full_ent(corr):
    pass

def com(M1,M2):
    return M1@M2-M2@M1



import transfer_dual_keldysh as tdk
J=0.2
g=0.4
T=4
delta=0.001
op=tdk.get_dense_sample(T,J,g,0.0)
# U=exp_to_maj(op+delta*np.eye(op.shape[0]))
pe,po=projectors_eo(2*T)
U=quad_to_maj(op@pe)*4**(T-1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
ni=np.isclose(np.linalg.matrix_power(U,10),np.linalg.matrix_power(U,14))
ni

np.linalg.matrix_power(U,10)[~ni]
U[~ni]


from test_majoranas import *

def test_exp_maj():
    H=genop(1,True)
    U=la.expm(1.0j*H) #generate unitary
    Um=maj_to_exp(U)
    assert np.allclose(Um@Um.T.conj(),np.eye(Um.shape[0]))
    assert np.allclose(exp_to_maj(Um),U)
    exp_to_maj(Um)
    exp_to_maj_new(Um)

    O=la.expm(genop(4,False))
    Om=maj_to_exp(O)
    assert np.allclose(exp_to_maj(Om),O)

def test_diag_herm():
    H=genop(4,True)
    D,U=la.eigh(H)
    D=np.diag(D)
    Hm=maj_to_quad(H)
    Um=maj_to_exp(U)
    assert np.allclose(exp_to_maj(Um),U)
    assert np.allclose(U.T.conj()@H@U,D)
    Dm=Um.T.conj()@Hm@Um
    assert np.allclose(Dm,np.diag(np.diag(Dm)))

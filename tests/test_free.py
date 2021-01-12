import imcode.free as free
import numpy as np
import pytest
pytestmark=pytest.mark.skip()

def test_basis_op():
    L=3
    assert np.allclose(free.me(L,0)@free.mo(L,0)+free.mo(L,0)@free.me(L,0),np.zeros_like(free.me(L,0)))
    assert np.allclose(free.me(L,0)@free.me(L,1)+free.me(L,1)@free.me(L,0),np.zeros_like(free.me(L,0)))
    assert np.allclose(free.me(L,0)@free.mo(L,1)+free.mo(L,1)@free.me(L,0),np.zeros_like(free.me(L,0)))
    assert np.allclose(free.mo(L,0)@free.mo(L,1)+free.mo(L,1)@free.mo(L,0),np.zeros_like(free.me(L,0)))
    assert np.allclose(free.me(L,0)@free.me(L,0)+free.me(L,0)@free.me(L,0),2*np.eye(free.me(L,0).shape[0]))
    assert np.allclose(free.mo(L,0)@free.mo(L,0)+free.mo(L,0)@free.mo(L,0),2*np.eye(free.me(L,0).shape[0]))

# np.random.seed(2)
def genop(L,hermitian=False):
    H=1.0j*np.random.normal(size=(2*L,2*L))
    if not hermitian:
        H+=np.random.normal(size=(2*L,2*L))
    H=H-H.T
    return H

def test_quad_maj():
    H=genop(4,True)
    Hm=maj_to_quad(H)
    assert np.allclose(Hm,Hm.T.conj())
    assert np.allclose(quad_to_maj(Hm),H)
    O=genop(4,False)
    Om=maj_to_quad(O)
    assert np.allclose(quad_to_maj(Om),O)


def test_commute():
    O1=genop(4,False)
    O2=genop(4,False)
    Om1=maj_to_quad(O1)
    Om2=maj_to_quad(O2)
    assert np.allclose(com(O1,O2),quad_to_maj(com(Om1,Om2)))
    assert np.allclose(maj_to_quad(com(O1,O2)),com(Om1,Om2))

def test_exp_maj():
    H=genop(4,True)
    U=la.expm(1.0j*H) #generate unitary
    Um=maj_to_exp(U)
    assert np.allclose(Um@Um.T.conj(),np.eye(Um.shape[0]))
    assert np.allclose(exp_to_maj(Um),U)

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
def test_diag_gen():
    O=genop(4,True)
    D,U=la.eig(O)
    D=np.diag(D)
    Om=maj_to_quad(O)
    Um=maj_to_exp(U)
    assert np.allclose(exp_to_maj(Um),U)
    assert np.allclose(U.T.conj()@O@U,D)
    Dm=Um.T.conj()@Om@Um
    assert np.allclose(Dm,np.diag(np.diag(Dm)))


def test_gauss():
    iH=1.0j*genop(4,True)
    full=gauss_to_full(iH)
    assert np.allclose(full_to_gauss(full),iH)

def test_gs():
    pass

def test_corr():
    iH=1.0j*genop(4,True)
    full=gauss_to_full(iH)
    assert np.allclose(calc_full_corr(full),calc_gauss_corr(iH))

def test_ent():
    iH=1.0j*genop(4,True)
    full=gauss_to_full(iH)
    corr=calc_full_corr(full)
    assert np.allclose(calc_full_ent(full),calc_corr_ent(corr))

def test_mps():
    pass

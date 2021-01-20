import imcode.free as free
import scipy.linalg as la
import numpy as np
from .utils import seed_rng
import pytest
def test_basis_op():
    L=3
    assert free.me(L,0)@free.mo(L,0)+free.mo(L,0)@free.me(L,0)==pytest.approx(np.zeros_like(free.me(L,0)))
    assert free.me(L,0)@free.me(L,1)+free.me(L,1)@free.me(L,0)==pytest.approx(np.zeros_like(free.me(L,0)))
    assert free.me(L,0)@free.mo(L,1)+free.mo(L,1)@free.me(L,0)==pytest.approx(np.zeros_like(free.me(L,0)))
    assert free.mo(L,0)@free.mo(L,1)+free.mo(L,1)@free.mo(L,0)==pytest.approx(np.zeros_like(free.me(L,0)))
    assert free.me(L,0)@free.me(L,0)+free.me(L,0)@free.me(L,0)==pytest.approx(2*np.eye(free.me(L,0).shape[0]))
    assert free.mo(L,0)@free.mo(L,0)+free.mo(L,0)@free.mo(L,0)==pytest.approx(2*np.eye(free.me(L,0).shape[0]))
def test_projector_ops():
    L=3
    assert free.po(L)@free.po(L) == pytest.approx(free.po(L))
    assert free.pe(L)@free.pe(L) == pytest.approx(free.pe(L))
    assert free.pe(L)@free.po(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.po(L)@free.pe(L) == pytest.approx(np.zeros((2**L,2**L)))

    assert free.po(L)@free.me(L,1)@free.po(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.pe(L)@free.me(L,1)@free.pe(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.po(L)@free.me(L,1)@free.pe(L) + free.pe(L)@free.me(L,1)@free.po(L)== pytest.approx(free.me(L,1))

    assert free.po(L)@free.mo(L,1)@free.po(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.pe(L)@free.mo(L,1)@free.pe(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.po(L)@free.mo(L,1)@free.pe(L) + free.pe(L)@free.mo(L,1)@free.po(L)== pytest.approx(free.mo(L,1))

    assert free.po(L)@free.mo(L,0)@free.po(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.pe(L)@free.mo(L,0)@free.pe(L) == pytest.approx(np.zeros((2**L,2**L)))
    assert free.po(L)@free.mo(L,0)@free.pe(L) + free.pe(L)@free.mo(L,0)@free.po(L)== pytest.approx(free.mo(L,0))

    assert free.mo(L,0)@free.me(L,2)@free.po(L) == pytest.approx(free.po(L)@free.mo(L,0)@free.me(L,2))
    assert free.mo(L,0)@free.me(L,2)@free.pe(L) == pytest.approx(free.pe(L)@free.mo(L,0)@free.me(L,2))

    assert free.mo(L,1)@free.me(L,1)@free.po(L) == pytest.approx(free.po(L)@free.mo(L,1)@free.me(L,1))
    assert free.mo(L,1)@free.me(L,1)@free.pe(L) == pytest.approx(free.pe(L)@free.mo(L,1)@free.me(L,1))

    assert free.mo(L,0)@free.mo(L,1)@free.po(L) == pytest.approx(free.po(L)@free.mo(L,0)@free.mo(L,1))
    assert free.mo(L,0)@free.mo(L,1)@free.pe(L) == pytest.approx(free.pe(L)@free.mo(L,0)@free.mo(L,1))

    assert free.me(L,2)@free.me(L,1)@free.po(L) == pytest.approx(free.po(L)@free.me(L,2)@free.me(L,1))
    assert free.me(L,2)@free.me(L,1)@free.pe(L) == pytest.approx(free.pe(L)@free.me(L,2)@free.me(L,1))


def test_quad_maj():
    L=4
    seed_rng("free_quad_maj")
    He=1.0j*np.random.normal(size=(2*L,2*L))
    He-=He.T
    Ho=1.0j*np.random.normal(size=(2*L,2*L))
    Ho-=Ho.T
    Hm=free.maj_to_quad((He,Ho))
    assert Hm.T.conj()==pytest.approx(Hm)
    # print(He)
    qtm=free.quad_to_maj(Hm)
    assert qtm[0] == pytest.approx(He)
    assert qtm[1] == pytest.approx(Ho)
    Oe=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oe-=Oe.T
    Oo=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oo-=Oo.T
    Om=free.maj_to_quad((Oe,Oo))
    qtm=free.quad_to_maj(Om)
    assert qtm[0] == pytest.approx(Oe)
    assert qtm[1] == pytest.approx(Oo)
def test_trans_maj():
    L=3
    seed_rng("free_trans_maj")
    He=1.0j*np.random.normal(size=(2*L,2*L))
    He-=He.T
    Ue=la.expm(1.0j*He)
    Ho=1.0j*np.random.normal(size=(2*L,2*L))
    Ho-=Ho.T
    Uo=la.expm(1.0j*Ho)
    Um=free.maj_to_trans((Ue,Uo))
    assert Um.T.conj()@Um==pytest.approx(np.eye(2**L))
    ttm=free.trans_to_maj(Um)
    assert ttm[0] == pytest.approx(Ue)
    assert ttm[1] == pytest.approx(Uo)
    Oe=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oe=la.expm(Oe-Oe.T)
    Oo=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oo=la.expm(Oo-Oo.T)
    Om=free.maj_to_trans((Oe,Oo))
    ttm=free.trans_to_maj(Om)
    assert ttm[0] == pytest.approx(Oe)
    assert ttm[1] == pytest.approx(Oo)
def test_quad_maj_sum():
    L=4
    seed_rng("free_quad_sum")
    O1e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1e-=O1e.T
    O1o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1o-=O1o.T
    O2e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2e-=O2e.T
    O2o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2o-=O2o.T
    Om1=free.maj_to_quad((O1e,O1o))
    Om2=free.maj_to_quad((O2e,O2o))
    qtm=free.quad_to_maj(Om1+Om2)
    assert qtm[0] == pytest.approx(O1e+O2e)
    assert qtm[1] == pytest.approx(O1o+O2o)
    assert free.maj_to_quad((O1e+O2e,O1o+O2o)) == pytest.approx(Om1+Om2)

def test_quad_maj_scalar():
    L=4
    seed_rng("free_quad_scalar")
    Oe=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oe-=Oe.T
    Oo=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Oo-=Oo.T
    alpha=np.random.normal()+1.0j*np.random.normal()
    Om=free.maj_to_quad((Oe,Oo))
    qtm= free.quad_to_maj(alpha*Om)
    assert qtm[0] == pytest.approx(alpha*Oe)
    assert qtm[1] == pytest.approx(alpha*Oo)
    assert free.maj_to_quad((alpha*Oe,alpha*Oo)) == pytest.approx(alpha*Om)
def test_trans_maj_prod():
    L=4
    seed_rng("free_trans_prod")
    O1e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1e=la.expm(O1e-O1e.T)
    O1o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1o=la.expm(O1o-O1o.T)
    O2e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2e=la.expm(O2e-O2e.T)
    O2o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2o=la.expm(O2o-O2o.T)
    Om1=free.maj_to_trans((O1e,O1o))
    Om2=free.maj_to_trans((O2e,O2o))
    ttm=free.trans_to_maj(Om1@Om2)
    assert ttm[0] == pytest.approx(O1e@O2e)
    assert ttm[1] == pytest.approx(O1o@O2o)
    assert free.maj_to_trans((O1e@O2e,O1o@O2o)) == pytest.approx(Om1@Om2)

def test_quad_commute():
    L=5
    seed_rng("free_quad_commute")
    O1e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1e-=O1e.T
    O1o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O1o-=O1o.T
    O2e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2e-=O2e.T
    O2o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    O2o-=O2o.T
    Om1=free.maj_to_quad((O1e,O1o))
    Om2=free.maj_to_quad((O2e,O2o))
    qtm=free.quad_to_maj(Om1@Om2-Om2@Om1)
    assert qtm[0] == pytest.approx(O1e@O2e-O2e@O1e)
    assert qtm[1] == pytest.approx(O1o@O2o-O2o@O1o)
    assert free.maj_to_quad((O1e@O2e-O2e@O1e,O1o@O2o-O2o@O1o)) == pytest.approx(Om1@Om2-Om2@Om1)
# def test_trans_commute():
#     L=4
#     seed_rng("free_trans_commute")
#     O1e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
#     O1e=la.expm(O1e-O1e.T)
#     O1o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
#     O1o=la.expm(O1o-O1o.T)
#     O2e=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
#     O2e=la.expm(O2e-O2e.T)
#     O2o=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
#     O2o=la.expm(O2o-O2o.T)
#     Om1=free.maj_to_trans((O1e,O1o))
#     Om2=free.maj_to_trans((O2e,O2o))
#     ttm=free.trans_to_maj(Om1@Om2-Om2@Om1)
#     assert ttm[0] == pytest.approx(O1e@O2e-O2e@O1e)
#     assert ttm[1] == pytest.approx(O1o@O2o-O2o@O1o)
#     assert free.maj_to_trans((O1e@O2e-O2e@O1e,O1o@O2o-O2o@O1o)) == pytest.approx(Om1@Om2-Om2@Om1)
@pytest.mark.skip
def test_diag_herm():
    seed_rng("free_diag_herm")
    L=5
    He=1.0j*np.random.normal(size=(2*L,2*L))
    He-=He.T
    Ho=1.0j*np.random.normal(size=(2*L,2*L))
    Ho-=Ho.T
    De,Ue=la.eigh(He)
    De=np.diag(De)
    Do,Uo=la.eigh(Ho)
    Do=np.diag(Do)
    Hm=free.maj_to_quad((He,Ho))
    Um=free.maj_to_trans((Ue,Uo))
    ttm=free.trans_to_maj(Um)
    print(ttm[0])
    print(Ue)
    assert ttm[0]==pytest.approx(Ue)
    assert ttm[1]==pytest.approx(Uo)
    Dm=Um.T.conj()@Hm@Um
    assert np.diag(np.diag(Dm))==pytest.approx(Dm)

@pytest.mark.skip
def test_diag_gen():
    seed_rng("free_diag_gen")
    L=5
    He=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    He-=He.T
    Ho=1.0j*np.random.normal(size=(2*L,2*L))+np.random.normal(size=(2*L,2*L))
    Ho-=Ho.T
    De,Ue=la.eig(He)
    De=np.diag(De)
    Do,Uo=la.eig(Ho)
    Do=np.diag(Do)
    Hm=free.maj_to_quad((He,Ho))
    Um=free.maj_to_trans((Ue,Uo))
    assert free.trans_to_maj(Um)==pytest.approx((Ue,Uo))
    Dm=la.inv(Um)@Hm@Um
    assert np.diag(np.diag(Dm))==pytest.approx(Dm)

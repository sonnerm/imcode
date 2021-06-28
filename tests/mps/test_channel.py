from tenpy.networks.mpo import MPO
from tenpy.linalg.np_conserved import Array
from tenpy.linalg.charges import LegCharge
import pytest
import numpy as np
from imcode.mps import unitary_channel,mpo_to_state,state_to_mpo,mpo_to_dense,apply,mps_to_dense
from .utils import seed_rng

def test_mpo_state_L1():
    L=1
    seed_rng("mpo_state_L1")
    Ws=np.random.normal(size=(L,1,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    Wsa=[Array.from_ndarray(W,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws]
    mp=MPO([FlatSite() for _ in range(L)],Wsa)
    dop=mpo_to_dense(mp)
    sop=mpo_to_state(mp)
    assert Ws[0,0,0] == pytest.approx(dop)
    assert mpo_to_dense(state_to_mpo(sop))==pytest.approx(dop)
def test_mpo_state_product():
    seed_rng("mpo_state_product")
    L=4
    Ws=np.random.normal(size=(L,1,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    Ws=[Array.from_ndarray(W,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws]
    mp=MPO([FlatSite() for _ in range(L)],Ws)
    dop=mpo_to_dense(mp)
    sop=mpo_to_state(mp)
    assert mps_to_dense(sop)==pytest.approx(dop)
    assert mpo_to_dense(state_to_mpo(sop))==pytest.approx(dop)
def random_mpo():
    pass
def test_mpo_state_bd10():
    seed_rng("mpo_state_bd10")
    L=4
    Ws=list(np.random.normal(size=(L,10,10,2,2)))
    Ws[0]=np.random.normal(size=(1,10,2,2))
    Ws[-1]=np.random.normal(size=(10,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    leg_m=LegCharge.from_trivial(10)
    Wsa=[Array.from_ndarray(Ws[0],[leg_b,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
    Wsa.extend([Array.from_ndarray(W,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws[1:-1]])
    Wsa.append(Array.from_ndarray(Ws[-1],[leg_m,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    mp=MPO([FlatSite() for _ in range(L)],Wsa)
    dop=mpo_to_dense(mp)
    sop=mpo_to_state(mp)
    assert mps_to_dense(sop)==pytest.approx(dop)
    assert mpo_to_dense(state_to_mpo(sop))==pytest.approx(dop)

def test_unitary_channel_bd10():
    seed_rng("unitary_channel_bd10")
    L=5
    Ws=list(np.random.normal(size=(L,10,10,2,2)))
    Ws[0]=np.random.normal(size=(1,10,2,2))
    Ws[-1]=np.random.normal(size=(10,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    leg_m=LegCharge.from_trivial(10)
    Wsa=[Array.from_ndarray(Ws[0],[leg_b,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
    Wsa.extend([Array.from_ndarray(W,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws[1:-1]])
    Wsa.append(Array.from_ndarray(Ws[-1],[leg_m,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    mpc=MPO([FlatSite() for _ in range(L)],Wsa)
    dmpc=mpo_to_dense(mpc)
    smpc=mpo_to_state(mpc)
    Ws=list(np.random.normal(size=(L,10,10,2,2)))
    Ws[0]=np.random.normal(size=(1,10,2,2))
    Ws[-1]=np.random.normal(size=(10,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    leg_m=LegCharge.from_trivial(10)
    Wsa=[Array.from_ndarray(Ws[0],[leg_b,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
    Wsa.extend([Array.from_ndarray(W,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws[1:-1]])
    Wsa.append(Array.from_ndarray(Ws[-1],[leg_m,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    mpo=MPO([FlatSite() for _ in range(L)],Wsa)
    dmpo=mpo_to_dense(mpo)
    umpo=unitary_channel(mpo)
    apply(umpo,smpc)
    assert mpo_to_dense(state_to_mpo(smpc))==pytest.approx(dmpo@dmpc@dmpo.T.conj())

def test_unitary_channel_product():
    seed_rng("unitary_channel_product")
    L=3
    Ws=np.random.normal(size=(L,1,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    Ws=[Array.from_ndarray(W,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws]
    mpc=MPO([FlatSite() for _ in range(L)],Ws)
    dmpc=mpo_to_dense(mpc)
    smpc=mpo_to_state(mpc)
    Ws=np.random.normal(size=(L,1,1,2,2))
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    Ws=[Array.from_ndarray(W,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for W in Ws]
    mpo=MPO([FlatSite() for _ in range(L)],Ws)
    dmpo=mpo_to_dense(mpo)
    umpo=unitary_channel(mpo)
    apply(umpo,smpc)
    assert mpo_to_dense(state_to_mpo(smpc))==pytest.approx(dmpo@dmpc@dmpo.T.conj())

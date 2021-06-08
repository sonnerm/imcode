import imcode.mps as mps
import pytest
from ..utils import seed_rng
import numpy as np
def test_mps_direct_dm_evo():
    seed_rng("mps_direct_em_evo")
    L=8
    chi=64
    t=6
    J=np.random.normal()
    g=np.random.normal()
    h=np.random.normal()
    T=dense.ising_T(t,J,g,h)
    im=dense.im_iterative(T)
    lop=dense.ising_W(t,g)@dense.ising_h(t,h)
    F=mps.ising_F([J]*(L-1),[g]*L,[h]*L)
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    init=[1.0,0.0,0.0,0.0]
    assert isinstance(F.sites[0],flat.FlatSite)
    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[0]=np.array([[1,0],[0,0]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    fdm=mps.mpo_to_state(MPO(F.sites,szop))

    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[0]=np.array([[1,0],[0,0]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    state0=mps.mpo_to_state(MPO(F.sites,szop))

    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[0]=np.array([[0,1],[0,0]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    state1=mps.mpo_to_state(MPO(F.sites,szop))

    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[0]=np.array([[0,0],[1,0]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    state2=mps.mpo_to_state(MPO(F.sites,szop))

    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[0]=np.array([[0,0],[0,1]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    state3=mps.mpo_to_state(MPO(F.sites,szop))

    channel=mps.unitary_channel(F)
    dms=mps.get_dm_evolution(T,lop,init)
    for ts in range(t):
        fdm=mps.apply(channel,fdm,chi)
        dm=np.zeros((2,2))
        dm[0,0]=mps.boundary_obs(state0,fdm)
        dm[0,1]=mps.boundary_obs(state1,fdm)
        dm[1,0]=mps.boundary_obs(state2,fdm)
        dm[1,1]=mps.boundary_obs(state3,fdm)
        assert dms[ts]==pytest.approx(dm)

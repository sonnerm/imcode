import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest

def test_dephase(seed_rng):
    L=4
    t=5
    chi=64
    J,g,h=np.random.normal(size=3)
    gamma=0.3
    cha=dense.unitary_channel(dense.ising.ising_F(L+1,[J]*L,[g]*(L+1),[h]*(L+1)))@dense.kron([dense.dephasing_channel(gamma)]*(L+1))
    ch1=dense.unitary_channel(dense.ising.ising_F(1,[],[g],[h]))@dense.dephasing_channel(gamma)
    Ts=((mps.ising.ising_J(t,J)@mps.ising.ising_W(t,[ch1]*t)).contract() for t in range(1,5,2))
    im=list(mps.ising.im_diamond(Ts,chi_max=chi))[-1]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    dms=mps.ising.boundary_dm_evolution(im,ch1,init)
    state=dense.outer([np.eye(2).ravel()/2]*(L)+[init.ravel()])
    summi=dense.outer([np.eye(2).ravel()]*(L))
    ddms=[np.einsum("a,abc->bc",summi,state.reshape((4**L),2,2))]
    for i in range(t):
        state=cha@state
        ddms.append(np.einsum("a,abc->bc",summi,state.reshape((4**L),2,2)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)


def test_depolarize(seed_rng):
    L=4
    t=5
    chi=64
    J,g,h=np.random.normal(size=3)
    gamma=0.3
    cha=dense.unitary_channel(dense.ising.ising_F(L+1,[J]*L,[g]*(L+1),[h]*(L+1)))@dense.kron([dense.depolarizing_channel(gamma)]*(L+1))
    ch1=dense.unitary_channel(dense.ising.ising_F(1,[],[g],[h]))@dense.depolarizing_channel(gamma)
    Ts=((mps.ising.ising_J(t,J)@mps.ising.ising_W(t,[ch1]*t)).contract() for t in range(1,5,2))
    im=list(mps.ising.im_diamond(Ts,chi_max=chi))[-1]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    dms=mps.ising.boundary_dm_evolution(im,ch1,init)
    state=dense.outer([np.eye(2).ravel()/2]*(L)+[init.ravel()])
    summi=dense.outer([np.eye(2).ravel()]*(L))
    ddms=[np.einsum("a,abc->bc",summi,state.reshape((4**L),2,2))]
    for i in range(t):
        state=cha@state
        ddms.append(np.einsum("a,abc->bc",summi,state.reshape((4**L),2,2)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)
def test_depolarize_z(seed_rng):
    pass

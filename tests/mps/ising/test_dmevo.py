import imcode.mps as mps
import imcode.dense as dense
import scipy.linalg as la
import numpy as np
import pytest

def test_boundary_single_dmevo_lops(seed_rng):
    L=4
    t=4
    chi=64
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L,))
    hs=np.random.normal(size=(L,))
    Ts=[mps.ising.ising_T(t,J,g,h,np.eye(2),np.eye(2)) for J,g,h in zip(Js,gs,hs)]
    lops=[np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j for _ in range(t)]
    lops=[la.expm(l-l.T.conj()) for l in lops]
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    im=mps.ising.open_boundary_im(t)
    for T in Ts:
        im=(T@im).contract(chi_max=chi)
    dms=mps.ising.boundary_dm_evolution(im,lops,init)

def test_boundary_single_dmevo_ising(seed_rng):
    L=4
    t=4
    chi=64
    Js=np.random.normal(size=(L,))
    gs=np.random.normal(size=(L+1,))
    hs=np.random.normal(size=(L+1,))
    Ts=[mps.ising.ising_T(t,J,g,h,np.eye(2),np.eye(2)) for J,g,h in zip(Js,gs[:-1],hs[:-1])]
    lop=la.expm(1.0j*hs[-1]*dense.SZ)@la.expm(1.0j*gs[-1]*dense.SX)
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    init=init+init.T.conj()
    init=init@init
    init/=np.trace(init)
    im=mps.ising.open_boundary_im(t)
    for T in Ts:
        im=(T@im).contract(chi_max=chi)
    dms=mps.ising.boundary_dm_evolution(im,lop,init)
    F=dense.ising.ising_F(L+1,Js,gs,hs)
    Fc=dense.unitary_channel(F)
    state=dense.outer([np.eye(2).ravel()/2]*(L)+[init.ravel()])
    summi=dense.outer([np.eye(2).ravel()]*(L))
    ddms=[np.einsum("a,ab->b",summi,state.reshape((4**L),4)).reshape((2,2))]
    for i in range(t):
        state=Fc@state
        ddms.append(np.einsum("a,ab->b",summi,state.reshape((4**L),4)).reshape((2,2)))
    for d,dd in zip(dms[::2],ddms):
        assert d==pytest.approx(dd)

def test_embedded_double_dmevo(seed_rng):
    pass

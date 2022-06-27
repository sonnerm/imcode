import imcode.mps as mps
import numpy.linalg as la
import numpy as np
import imcode.dense as dense
import pytest
def test_mps_ising_h(seed_rng):
    t=3
    h=np.random.normal()#+np.random.normal()*1.0j
    dih=dense.ising.ising_h(t,h)
    mih=mps.ising.ising_h(t,h)
    assert dih==pytest.approx(mih.to_dense())

def test_mps_ising_g(seed_rng):
    t=3
    g=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diW=dense.ising.ising_g(t,g,init,final)
    miW=mps.ising.ising_g(t,g,init,final)
    assert diW==pytest.approx(miW.to_dense())

def test_mps_ising_J(seed_rng):
    t=3
    J=np.random.normal()+np.random.normal()*1.0j
    diJ=dense.ising.ising_J(t,J)
    miJ=mps.ising.ising_J(t,J)
    assert diJ==pytest.approx(miJ.to_dense())

def test_mps_ising_T(seed_rng):
    t=3
    J=np.random.normal()+np.random.normal()*1.0j
    g=np.random.normal()+np.random.normal()*1.0j
    h=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diT=dense.ising.ising_T(t,J,g,h,init,final)
    miT=mps.ising.ising_T(t,J,g,h,init,final)
    assert diT==pytest.approx(miT.to_dense())

def test_mps_ising_T_onesite(seed_rng):
    t=1
    J=np.random.normal()+np.random.normal()*1.0j
    g=np.random.normal()+np.random.normal()*1.0j
    h=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diT=dense.ising.ising_T(t,J,g,h,init,final)
    miT=mps.ising.ising_T(t,J,g,h,init,final)
    assert diT==pytest.approx(miT.to_dense())

def test_mps_ising_g_onesite(seed_rng):
    t=1
    g=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diW=dense.ising.ising_g(t,g,init,final)
    miW=mps.ising.ising_g(t,g,init,final)
    assert diW==pytest.approx(miW.to_dense())
def test_mps_ising_W_onesite(seed_rng):
    t=1
    H=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    H=H.T.conj()+H
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    _,U=la.eigh(H)
    ch=dense.unitary_channel(U)
    diW=dense.ising.ising_W(t,[ch],init,final)
    miW=mps.ising.ising_W(t,[ch],init,final)
    assert diW==pytest.approx(miW.to_dense())


def test_mps_ising_W(seed_rng):
    t=4
    H=[np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2)) for _ in range(t)]
    H=[h.T.conj()+h for h in H]
    init=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    final=np.random.normal(size=(2,2))+1.0j*np.random.normal(size=(2,2))
    U=[la.eigh(h)[1] for h in H]
    ch=[dense.unitary_channel(u) for u in U]
    diW=dense.ising.ising_W(t,ch,init,final)
    miW=mps.ising.ising_W(t,ch,init,final)
    assert diW==pytest.approx(miW.to_dense(),abs=1e-6,rel=1e-6)

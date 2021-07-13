import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest
def test_mps_ising_h(seed_rng):
    t=3
    h=np.random.normal()#+np.random.normal()*1.0j
    dih=dense.ising.ising_h(t,h)
    mih=mps.ising.ising_h(t,h)
    assert dih==pytest.approx(mih.to_dense())

def test_mps_ising_W(seed_rng):
    t=3
    g=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diW=dense.ising.ising_W(t,g,init,final)
    miW=mps.ising.ising_W(t,g,init,final)
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

def test_mps_ising_W_onesite(seed_rng):
    t=1
    g=np.random.normal()+np.random.normal()*1.0j
    init=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    final=np.random.normal(size=(2,2))+np.random.normal(size=(2,2))*1.0j
    diW=dense.ising.ising_W(t,g,init,final)
    miW=mps.ising.ising_W(t,g,init,final)
    assert diW==pytest.approx(miW.to_dense())

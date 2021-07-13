import imcode.mps as mps
import imcode.dense as dense
import numpy as np
import pytest
def test_mps_ising_F(seed_rng):
    L=5
    J=np.random.normal(size=(L-1,))+np.random.normal(size=(L-1,))*1.0j
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    diF=dense.ising.ising_F(L,J,g,h)
    miF=mps.ising.ising_F(L,J,g,h)
    assert diF==pytest.approx(miF.to_dense())

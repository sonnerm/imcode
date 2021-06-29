import imcode.mps as mps
import numpy as np
import imcode.dense as dense
import pytest
def test_mps_ising_H(seed_rng):
    L=5
    J=np.random.normal(size=(L-1,))+np.random.normal(size=(L-1,))*1.0j #only obc are supported
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    diH=dense.ising.ising_H(J,g,h)
    miH=mps.ising.ising_H(J,g,h)
    assert diH==pytest.approx(miH.to_dense())

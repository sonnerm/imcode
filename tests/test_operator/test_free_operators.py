import imcode.free as free
import imcode.dense as dense
import imcode.mps.fold as fold
import pytest
import numpy as np
from ..utils import seed_rng

def test_free_dense_ising_H():
    seed_rng("free_dense_ising_H")
    L=5
    J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
    g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
    print(free.ising_H(J,g))
    print(dense.ising_H(J,g,np.zeros_like(J)))
    assert free.maj_to_quad(free.ising_H(J,g)) == pytest.approx(dense.ising_H(J,g,np.zeros_like(J)))

def test_free_dense_ising_F():
    seed_rng("free_dense_ising_F")
    L=5
    J=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
    g=np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,))
    assert free.maj_to_trans(free.ising_F(J,g)) == pytest.approx(dense.ising_F(J,g,np.zeros_like(J)))

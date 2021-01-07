import pytest
from pytest import mark
import numpy as np
import imcode.mps as mps

def test_ising_H():
    L=5
    np.random.seed(hash("mps_ising_H")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    assert mps.mpo_to_dense(mps.ising_H(J,g,h))==pytest.approx(dense.ising_H(J,g,h))

def test_ising_F():
    L=5
    np.random.seed(hash("mps_ising_F")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    assert mps.mpo_to_dense(mps.ising_F(J,g,h))==pytest.approx(dense.ising_F(J,g,h))

def test_ising_T():
    t=3
    np.random.seed(hash("mps_ising_T")%2**32)
    J=np.random.normal()+1.0j*np.random.normal()
    g=np.random.normal()+1.0j*np.random.normal()
    h=np.random.normal()+1.0j*np.random.normal()
    assert mps.mpo_to_dense(mps.ising_T(t,J,g,h))==pytest.approx(dense.ising_T(t,J,g,h))

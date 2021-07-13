import numpy as np
from imcode import dense
import pytest
import scipy.linalg as scla
def test_dense_ising_F_complex(seed_rng):
    L=5
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    # simple implementation of ising_F, to be compared to future implementations
    diF=dense.ising.ising_F(L,J,g,h)
    def simple_ising_F(J, g, h):
        return scla.expm(1.0j * dense.ising.ising_H(L,J, [0.0] * L, h)) @ scla.expm(1.0j * dense.ising.ising_H(L,[0.0] * L, g, [0.0] * L))
    assert diF.dtype==np.complex_
    assert diF.conj().T@diF!=pytest.approx(np.eye(diF.shape[0])) #not unitary
    assert diF@diF.T.conj()!=pytest.approx(np.eye(diF.shape[0])) #not unitary
    assert diF.T.conj()!=pytest.approx(diF)
    assert diF.T!=pytest.approx(diF)
    assert diF.conj()!=pytest.approx(diF)
    assert simple_ising_F(J, g, h) == pytest.approx(diF)
def test_dense_ising_F_real(seed_rng):
    L=5
    J = np.random.normal(size=L)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    # simple implementation of ising_F, to be compared to future implementations
    diF=dense.ising.ising_F(L,J,g,h)
    def simple_ising_F(J, g, h):
        return scla.expm(1.0j * dense.ising.ising_H(L,J, [0.0] * L, h)) @ scla.expm(1.0j * dense.ising.ising_H(L,[0.0] * L, g, [0.0] * L))
    assert diF.dtype==np.complex_
    assert diF.conj().T@diF==pytest.approx(np.eye(diF.shape[0])) #unitary
    assert diF@diF.T.conj()==pytest.approx(np.eye(diF.shape[0]))
    assert diF.T.conj()!=pytest.approx(diF)
    assert diF.T!=pytest.approx(diF)
    assert diF.conj()!=pytest.approx(diF)
    assert simple_ising_F(J, g, h) == pytest.approx(diF)

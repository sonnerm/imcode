import imcode
import scipy.linalg as scla
import numpy as np
import pytest
def test_ising_F_complex(seed_rng):
    L=5
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    # simple implementation of ising_F, to be compared to future implementations
    miF=imcode.ising_F(L,J,g,h)
    def simple_ising_F(J, g, h):
        return scla.expm(1.0j * np.array(imcode.ising_H(L,J, [0.0] * L, h)) @ scla.expm(1.0j * np.array(imcode.ising_H(L,[0.0] * L, g, [0.0] * L))))
    diF=np.array(miF)
    assert diF.dtype==np.complex_
    assert diF.conj().T@diF!=pytest.approx(np.eye(diF.shape[0])) #not unitary
    assert diF@diF.T.conj()!=pytest.approx(np.eye(diF.shape[0])) #not unitary
    assert diF.T.conj()!=pytest.approx(diF)
    assert diF.T!=pytest.approx(diF)
    assert diF.conj()!=pytest.approx(diF)
    assert simple_ising_F(J, g, h) == pytest.approx(diF)

def test_ising_F_real(seed_rng):
    L=5
    J = np.random.normal(size=L)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    miF=imcode.ising_F(L,J,g,h)
    def simple_ising_F(J, g, h):
        return scla.expm(1.0j * np.array(imcode.ising_H(L,J, [0.0] * L, h)) @ scla.expm(1.0j * np.array(imcode.ising_H(L,[0.0] * L, g, [0.0] * L))))
    diF=np.array(miF)
    assert diF.dtype==np.complex_
    assert diF.conj().T@diF==pytest.approx(np.eye(diF.shape[0])) #unitary
    assert diF@diF.T.conj()==pytest.approx(np.eye(diF.shape[0]))
    assert diF.T.conj()!=pytest.approx(diF)
    assert diF.T!=pytest.approx(diF)
    assert diF.conj()!=pytest.approx(diF)
    assert simple_ising_F(J, g, h) == pytest.approx(diF)

def test_ising_F_trotter(seed_rng):
    L=5
    J=np.random.normal(size=(L-1,))+np.random.normal(size=(L-1,))*1.0j
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    diF=dense.ising.ising_F(L,J,g,h)
    miF=mps.ising.ising_F(L,J,g,h)
    assert diF==pytest.approx(miF.to_dense())

def test_ising_H(seed_rng):
    # Build operator manually from basic operators and compare also for complex coefficients <=> non-hermitian operators
    L = 5
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    ret = np.zeros_like(dense.one(L),dtype=complex)
    for i in range(L):
        ret += dense_sz(L, i) @ dense_sz(L, (i + 1) % L) * J[i]
        ret += dense_sz(L, i) * h[i]
        ret += dense_sx(L, i) * g[i]
    miH=imcode.ising_H(L,J, g, h)
    assert diH.dtype==np.complex_
    assert ret == pytest.approx(diH)
    assert (ret - dense.sz(L, 0) @ dense.sz(L, L - 1) * J[-1]) == pytest.approx(dense.ising.ising_H(L, J[:-1], g, h))

    # Real coefficients <=> Real Hermitian operator
    J = np.random.normal(size=L)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    diH=dense.ising.ising_H(L, J, g, h)
    assert diH.dtype==np.float_
    assert (diH == diH.T).all()
    
def test_single_site_ising_H():
    g = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    h = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    diH=dense.ising.ising_H(1,[],g,h)
    assert diH==pytest.approx(dense.SZ*h+dense.SX*g)
    diH=dense.ising.ising_H(1,[0.0],g,h)
    assert diH==pytest.approx(dense.SZ*h+dense.SX*g)
def test_dense_ising_H_real(seed_rng):
    L=5
    J=np.random.normal(size=(L,))
    g=np.random.normal(size=(L,))
    h=np.random.normal(size=(L,))
    diH=dense.ising.ising_H(L,J,g,h)
    assert diH.dtype==np.float_
    assert diH.conj()==pytest.approx(diH)
    assert diH.T.conj()==pytest.approx(diH)
    assert diH.T==pytest.approx(diH)
def test_dense_ising_H_complex(seed_rng):
    L=5
    J=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    diH=dense.ising.ising_H(L,J,g,h)
    assert diH.dtype==np.complex_
    #This model is always symmetric
    assert diH.T==pytest.approx(diH)
    #ensure that tests cover generic case
    assert diH.conj()!=pytest.approx(diH)
    assert diH.T.conj()!=pytest.approx(diH)

def test_dense_ising_H_complex_obc(seed_rng):
    L=5
    J=np.random.normal(size=(L-1,))+np.random.normal(size=(L-1,))*1.0j
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    diH=dense.ising.ising_H(L,J,g,h)
    assert diH.dtype==np.complex_
    #This model is always symmetric
    assert diH.T==pytest.approx(diH)
    #ensure that tests cover generic case
    assert diH.conj()!=pytest.approx(diH)
    assert diH.T.conj()!=pytest.approx(diH)

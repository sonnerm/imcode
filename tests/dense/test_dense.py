import pytest

import imcode.dense as dense
import numpy as np
import scipy.linalg as scla


def test_basic_ops():
    # Are one site many body operators sane?
    assert (dense.sx(1, 0) == dense.SX).all()
    assert (dense.sy(1, 0) == dense.SY).all()
    assert (dense.sz(1, 0) == dense.SZ).all()
    assert (dense.sp(1, 0) == dense.SP).all()
    assert (dense.sm(1, 0) == dense.SM).all()

    # Testing algebra on both sites in a 2 site system
    assert (dense.sx(2, 0) @ dense.sx(2, 0) == dense.one(2)).all()
    assert (dense.sy(2, 0) @ dense.sy(2, 0) == dense.one(2)).all()
    assert (dense.sz(2, 0) @ dense.sz(2, 0) == dense.one(2)).all()

    assert (dense.sx(2, 1) @ dense.sx(2, 1) == dense.one(2)).all()
    assert (dense.sy(2, 1) @ dense.sy(2, 1) == dense.one(2)).all()
    assert (dense.sz(2, 1) @ dense.sz(2, 1) == dense.one(2)).all()

    assert (dense.sx(2, 1) @ dense.sy(2, 1) == 1.0j * dense.sz(2, 1)).all()
    assert (dense.sx(2, 1) @ dense.sz(2, 1) == -1.0j * dense.sy(2, 1)).all()

    # Testing addition
    lhs = (dense.sx(2, 0) + dense.sz(2, 1)) @ (dense.sy(2, 0) + dense.sy(2, 1))
    rhs = 1.0j * dense.sz(2, 0) + dense.sx(2, 0) @ dense.sy(2, 1) + \
        dense.sz(2, 1) @ dense.sy(2, 0) - 1.0j * dense.sx(2, 1)
    assert (lhs == rhs).all()
    # Operators on different sites commute
    assert (dense.sx(2, 1) @ dense.sz(2, 0) ==
            dense.sz(2, 0) @ dense.sx(2, 1)).all()
    assert ((dense.sx(3, 1) + dense.sy(3, 2)) @ dense.sz(3, 0) ==
            dense.sz(3, 0) @ (dense.sx(3, 1) + dense.sy(3, 2))).all()



def test_ising_H():
    # Build operator manually from basic operators and compare also for complex coefficients <=> non-hermitian operators
    L = 5
    np.random.seed(hash("dense_test_ising_H") % 2**32)
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    ret = np.zeros_like(dense.one(L),dtype=complex)
    for i in range(L):
        ret += dense.sz(L, i) @ dense.sz(L, (i + 1) % L) * J[i]
        ret += dense.sz(L, i) * h[i]
        ret += dense.sx(L, i) * g[i]
    diH=dense.ising_H(J, g, h)
    assert diH.dtype==np.complex_
    assert ret == pytest.approx(diH)
    assert (ret - dense.sz(L, 0) @ dense.sz(L, L - 1) * J[-1]) == pytest.approx(dense.ising_H(J[:-1], g, h))

    # Real coefficients <=> Real Hermitian operator
    J = np.random.normal(size=L)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    diH=dense.ising_H(J, g, h)
    assert diH.dtype==np.float_
    assert (dense.ising_H(J, g, h) == dense.ising_H(J, g, h).T).all()
def test_single_site_ising_H():
    g = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    h = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    diH=dense.ising_H([],g,h)
    assert diH==pytest.approx(dense.SZ*h+dense.SX*g)
    diH=dense.ising_H([0.0],g,h)
    assert diH==pytest.approx(dense.SZ*h+dense.SX*g)
def test_ising_F_simple():
    np.random.seed(hash("dense_test_ising_F") % 2**32)
    L=5
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)

    # simple implementation of ising_F, to be compared to future implementations
    def simple_ising_F(J, g, h):
        return scla.expm(1.0j * dense.ising_H(J, [0.0] * L, h)) @ scla.expm(1.0j * dense.ising_H([0.0] * L, g, [0.0] * L))
    assert simple_ising_F(J, g, h) == pytest.approx(dense.ising_F(J, g, h))

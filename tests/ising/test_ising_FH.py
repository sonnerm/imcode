import imcode
import scipy.linalg as scla
from imcode import SX,SZ,ID
import numpy as np
import pytest
import functools
import ttarray as tt

def simple_ising_F(L, J, g, h):
    return scla.expm(1j * np.array(imcode.ising_H(L,J,0,h))) @ scla.expm(1j * np.array(imcode.ising_H(L,0,g,0)))

def simple_ising_H(L,J,g,h):
    ret = np.zeros((2**L,2**L),dtype=complex)
    for i in range(L):
        if i<L-1:
            ret += J[i]*mkron([ID]*i+[SZ,SZ]+[ID]*(L-i-2))
        ret += h[i]*mkron([ID]*i+[SZ]+[ID]*(L-i-1))
        ret += g[i]*mkron([ID]*i+[SX]+[ID]*(L-i-1))
    return ret
def mkron(args):
    return functools.reduce(np.kron,args)

def test_ising_F(seed_rng):
    L=5
    J = np.random.normal(size=L-1)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    miF=imcode.ising_F(L,J,g,h)
    diF=np.array(miF)
    assert diF.dtype==np.complex_
    assert simple_ising_F(L, J, g, h) == pytest.approx(diF)

def test_ising_F_trotter(seed_rng):
    L=5
    J=np.random.normal(size=(L-1,))
    g=np.random.normal(size=(L,))
    h=np.random.normal(size=(L,))
    dt=0.001
    miF=np.array(imcode.ising_F(L,J*dt,g*dt,h*dt))
    miH=np.array(imcode.ising_H(L,J*dt,g*dt,h*dt))
    assert scla.expm(1.0j*miH)==pytest.approx(miF,rel=10*dt)

def test_ising_H(seed_rng):
    # Build operator manually from basic operators and compare also for complex coefficients <=> non-hermitian operators
    L = 5
    J = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    g = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    h = np.random.normal(size=L) + 1.0j * np.random.normal(size=L)
    miH=imcode.ising_H(L,J, g, h)
    ret=simple_ising_H(L,J,g,h)
    assert miH.dtype==np.complex_
    assert isinstance(miH,tt.TensorTrainArray)
    assert np.array(miH)==pytest.approx(ret)
    # Real coefficients <=> Real Hermitian operator
    J = np.random.normal(size=L)
    g = np.random.normal(size=L)
    h = np.random.normal(size=L)
    miH=imcode.ising_H(L,J, g, h)
    ret=simple_ising_H(L,J,g,h)
    assert miH.dtype==np.float_
    assert isinstance(miH,tt.TensorTrainArray)
    assert np.array(miH)==pytest.approx(ret)

def test_single_site_ising_H():
    g = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    h = np.random.normal(size=1) + 1.0j * np.random.normal(size=1)
    diH=np.array(imcode.ising_H(1,[],g,h))
    assert diH==pytest.approx(SZ*h+SX*g)
    diH=np.array(imcode.ising_H(1,4.0,g,h))
    assert diH==pytest.approx(SZ*h+SX*g)

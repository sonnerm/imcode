import imcode
import numpy.linalg as la
import scipy.linalg as scla
from imcode import SX,SZ,SY,ID
import numpy as np
import pytest
import functools
import ttarray as tt
def simple_brickwork_F(L, gates,reversed=False):
    gatese=np.array(gates).copy()
    gatese[1::2,:,:]=0
    gateso=np.array(gates).copy()
    gateso[::2,:,:]=0
    Fe=scla.expm(1j * np.array(imcode.brickwork_H(L,gatese)))
    Fo=scla.expm(1j * np.array(imcode.brickwork_H(L,gateso)))
    if reversed:
        return Fe@Fo
    else:
        return Fo@Fe

def simple_brickwork_H(L,gates):
    ret = np.zeros((2**L,2**L),dtype=complex)
    for i,g in enumerate(gates):
        ret += mkron([ID]*i+[g]+[ID]*(L-i-2))
    return ret
def mkron(args):
    return functools.reduce(np.kron,args)
def test_brickwork_F_trotter(seed_rng):
    L=5
    gates=np.random.normal(size=(L-1,4,4))+np.random.normal(size=(L-1,4,4))*1.0j
    gates=gates+gates.transpose([0,2,1]).conj()
    gates/=max(max(np.abs(la.eigvalsh(g))) for g in gates)
    dt=0.001
    gates*=dt
    egates=[scla.expm(1.0j*g) for g in gates]
    miF=np.array(imcode.brickwork_F(L,egates))
    miH=np.array(imcode.brickwork_H(L,gates))
    assert scla.expm(1.0j*miH)==pytest.approx(miF,rel=10*dt,abs=10*dt)
def test_four_site_brickwork(seed_rng):
    L=4
    gates=np.random.normal(size=(L-1,4,4))+np.random.normal(size=(L-1,4,4))*1.0j
    egates=[scla.expm(1.0j*g) for g in gates]
    dhF=simple_brickwork_H(L,gates)
    mhF=imcode.brickwork_H(L,gates)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_brickwork_F(L,gates)
    mhF=imcode.brickwork_F(L,egates)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_brickwork_F(L,gates,reversed=True)
    mhF=imcode.brickwork_F(L,egates,reversed=True)
    assert dhF==pytest.approx(mhF.todense())

def test_five_site_brickwork(seed_rng):
    L=5
    gates=np.random.normal(size=(L-1,4,4))+np.random.normal(size=(L-1,4,4))*1.0j
    egates=[scla.expm(1.0j*g) for g in gates]
    dhF=simple_brickwork_H(L,gates)
    mhF=imcode.brickwork_H(L,gates)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_brickwork_F(L,gates)
    mhF=imcode.brickwork_F(L,egates)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_brickwork_F(L,gates,reversed=True)
    mhF=imcode.brickwork_F(L,egates,reversed=True)
    assert dhF==pytest.approx(mhF.todense())

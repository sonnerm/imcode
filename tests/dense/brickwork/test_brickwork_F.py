import imcode.dense as dense
import numpy as np
import scipy.linalg as la
import pytest
def test_two_site_brickwork_F(seed_rng):
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    diH=dense.brickwork.brickwork_F(2,[gate])
    assert diH==pytest.approx(gate)

def test_four_site_open_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(3)]
    diH=dense.brickwork.brickwork_H(4,gates)
    dt=1e-5
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(4,egates)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(4,[g.T for g in egates],True).T)
    assert diF!=pytest.approx(diF.T.conj())
    assert diF@diF.T.conj()!=pytest.approx(np.eye(16))
    diFT=dense.brickwork.brickwork_F(4,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diFT!=pytest.approx(diF.T.conj())
    assert diFT@diFT.T.conj()!=pytest.approx(np.eye(16))


def test_four_site_per_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(4)]
    diH=dense.brickwork.brickwork_H(4,gates)
    dt=1e-5
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(4,egates)
    miF=la.expm(1.0j*dt*diH)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(4,[g.T for g in egates],True).T)
    assert diF!=pytest.approx(diF.T.conj())
    assert diF@diF.T.conj()!=pytest.approx(np.eye(16))
    diFT=dense.brickwork.brickwork_F(4,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diFT!=pytest.approx(diF.T.conj())
    assert diFT@diFT.T.conj()!=pytest.approx(np.eye(16))

def test_four_site_per_brickwork_F_real(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(4)]
    gates=[g/2+g.T.conj()/2 for g in gates]
    diH=dense.brickwork.brickwork_H(4,gates)
    dt=1e-5
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(4,egates)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(4,[g.T for g in egates],True).T)
    assert diF!=pytest.approx(diF.T.conj())
    assert diF@diF.T.conj()==pytest.approx(np.eye(16))
    diFT=dense.brickwork.brickwork_F(4,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diFT!=pytest.approx(diF.T.conj())
    assert diFT@diFT.T.conj()==pytest.approx(np.eye(16))
def test_five_site_open_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(4)]
    diH=dense.brickwork.brickwork_H(5,gates)
    dt=1e-5
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(5,egates)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(5,[g.T for g in egates],True).T)
    assert diF!=pytest.approx(diF.T.conj())
    assert diF@diF.T.conj()!=pytest.approx(np.eye(16))
    diFT=dense.brickwork.brickwork_F(5,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt**2,abs=10*dt**2)
    assert diFT!=pytest.approx(diF.T.conj())
    assert diFT@diFT.T.conj()!=pytest.approx(np.eye(16))


def test_five_site_per_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(5)]
    diH=dense.brickwork.brickwork_H(5,gates)
    with pytest.raises(AssertionError):
        diF=dense.brickwork.brickwork_F(5,gates)

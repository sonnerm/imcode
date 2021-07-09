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
    dt=1e-4
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(4,egates)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt,abs=10*dt)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(4,[g.T for g in egates],True).T)
    diFT=dense.brickwork.brickwork_F(4,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt,abs=10*dt)


def test_four_site_per_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(4)]
    diH=dense.brickwork.brickwork_H(4,gates)
    dt=1e-4
    egates=[la.expm(1.0j*dt*g) for g in gates]
    diF=dense.brickwork.brickwork_F(4,egates)
    assert diF==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt,abs=10*dt)
    assert diF==pytest.approx(dense.brickwork.brickwork_F(4,[g.T for g in egates],True).T)
    diFT=dense.brickwork.brickwork_F(4,egates,True)
    assert diFT==pytest.approx(la.expm(1.0j*dt*diH),rel=10*dt,abs=10*dt)

import imcode.dense as dense
import numpy as np
import pytest
def test_two_site_brickwork_H(seed_rng):
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    diH=dense.brickwork.brickwork_H(2,[gate])
    assert diH==pytest.approx(gate.reshape((4,4)))

def test_four_site_brickwork_H(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(3)]
    diH=dense.brickwork.brickwork_H(4,gates)
    comp=dense.kron([gates[0],np.eye(4)])+dense.kron([np.eye(2),gates[1],np.eye(2)])+dense.kron([np.eye(4),gates[2]])
    assert diH==pytest.approx(comp)

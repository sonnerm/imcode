import imcode.dense as dense
import numpy as np
import pytest
def test_two_site_brickwork_H(seed_rng):
    gate = np.random.normal(size=(2,2,2,2)) + 1.0j * np.random.normal(size=(2,2,2,2))
    diH=dense.brickwork.brickwork_H(1,[gate])
    assert diH==pytest.approx(gate.reshape((4,4)))

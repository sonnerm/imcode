import imcode.dense as dense
import imcode.mps as mps
import numpy as np
import pytest
def test_two_site_mps_brickwork_F(seed_rng):
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    miF=mps.brickwork.brickwork_F(2,[gate])
    diF=dense.brickwork.brickwork_F(2,[gate])
    assert miF.to_dense()==pytest.approx(diF)
    # assert miF.to_dense()==pytest.approx(gate)

def test_four_site_mps_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(3)]
    miF=mps.brickwork.brickwork_F(4,gates)
    diF=dense.brickwork.brickwork_F(4,gates)
    assert miF.to_dense()==pytest.approx(diF)

    miF=mps.brickwork.brickwork_F(4,gates,True)
    diF=dense.brickwork.brickwork_F(4,gates,True)
    assert miF.to_dense()==pytest.approx(diF)


def test_five_site_mps_brickwork_F(seed_rng):
    gates = [np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4)) for _ in range(4)]
    miF=mps.brickwork.brickwork_F(5,gates)
    diF=dense.brickwork.brickwork_F(5,gates)
    assert miF.to_dense()==pytest.approx(diF)

    miF=mps.brickwork.brickwork_F(5,gates,True)
    diF=dense.brickwork.brickwork_F(5,gates,True)
    assert miF.to_dense()==pytest.approx(diF)

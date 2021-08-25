import numpy as np
import imcode.mps as mps
import imcode.dense as dense
import pytest
def test_mps_brickwork_Lb(seed_rng):
    t=2
    gate = np.random.normal(size=(2,2)) + 1.0j * np.random.normal(size=(2,2))
    gate=dense.unitary_channel(gate)
    miL=mps.brickwork.brickwork_Lb(t,gate)
    diL=dense.brickwork.brickwork_Lb(t,gate)
    assert miL.to_dense()==pytest.approx(diL)
    t=3
    init = np.random.normal(size=(2,2)) + 1.0j * np.random.normal(size=(2,2))
    final = np.random.normal(size=(2,2)) + 1.0j * np.random.normal(size=(2,2))
    miL=mps.brickwork.brickwork_Lb(t,gate,init,final)
    diL=dense.brickwork.brickwork_Lb(t,gate,init,final)
    assert miL.to_dense()==pytest.approx(diL)

def test_mps_brickwork_La():
    t=2
    miL=mps.brickwork.brickwork_La(t)
    diL=dense.brickwork.brickwork_La(t)
    assert miL.to_dense()==pytest.approx(diL)
    t=3
    miL=mps.brickwork.brickwork_La(t)
    diL=dense.brickwork.brickwork_La(t)
    assert miL.to_dense()==pytest.approx(diL)

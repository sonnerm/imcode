import numpy as np
import imcode.mps as mps
import imcode.dense as dense
import pytest
def test_mps_brickwork_Sb(seed_rng):
    t=2
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    init = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    final = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    gate=dense.unitary_channel(gate)
    miS=mps.brickwork.brickwork_Sb(t,gate,init,final)
    diS=dense.brickwork.brickwork_Sb(t,gate,init,final)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_brickwork_Sa(seed_rng):
    t=2
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    gate=dense.unitary_channel(gate)
    miS=mps.brickwork.brickwork_Sa(t,gate)
    diS=dense.brickwork.brickwork_Sa(t,gate)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_brickwork_Sb_L1(seed_rng):
    t=1
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    init = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    final = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    gate=dense.unitary_channel(gate)
    miS=mps.brickwork.brickwork_Sb(t,gate,init,final)
    diS=dense.brickwork.brickwork_Sb(t,gate,init,final)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_brickwork_Sa_L1(seed_rng):
    t=1
    gate = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    gate=dense.unitary_channel(gate)
    miS=mps.brickwork.brickwork_Sa(t,gate)
    diS=dense.brickwork.brickwork_Sa(t,gate)
    assert miS.to_dense()==pytest.approx(diS)

import numpy as np
import imcode.mps as mps
import imcode.dense as dense
import pytest
def test_mps_heisenberg_Lb(seed_rng):
    t=2
    hx,hy,hz=np.random.normal(size=(3,))+np.random.normal(size=(3,))*1.0j
    miL=mps.brickwork.heisenberg_Lb(t,hx,hy,hz)
    diL=dense.brickwork.heisenberg_Lb(t,hx,hy,hz)
    assert miL.to_dense()==pytest.approx(diL)
    t=3
    hx,hy,hz=np.random.normal(size=(3,))+np.random.normal(size=(3,))*1.0j
    init = np.random.normal(size=(2,2)) + 1.0j * np.random.normal(size=(2,2))
    final = np.random.normal(size=(2,2)) + 1.0j * np.random.normal(size=(2,2))
    miL=mps.brickwork.heisenberg_Lb(t,hx,hy,hz,init,final)
    diL=dense.brickwork.heisenberg_Lb(t,hx,hy,hz,init,final)
    assert miL.to_dense()==pytest.approx(diL)

def test_mps_heisenberg_La():
    t=2
    miL=mps.brickwork.heisenberg_La(t)
    diL=dense.brickwork.brickwork_La(t)
    assert miL.to_dense()==pytest.approx(diL)
    t=3
    miL=mps.brickwork.heisenberg_La(t)
    diL=dense.brickwork.brickwork_La(t)
    assert miL.to_dense()==pytest.approx(diL)

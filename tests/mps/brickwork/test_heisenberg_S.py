import numpy as np
import imcode.mps as mps
import imcode.dense as dense
import pytest
def test_mps_heisenberg_Sb(seed_rng):
    t=2
    Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze=np.random.normal(size=(9,))+np.random.normal(size=(9,))*1.0j
    init = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    final = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    miS=mps.brickwork.heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze,init,final)
    diS=dense.brickwork.heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze,init,final)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_heisenberg_Sa(seed_rng):
    t=2
    Jx,Jy,Jz=np.random.normal(size=(3,))+np.random.normal(size=(3,))*1.0j
    miS=mps.brickwork.heisenberg_Sa(t,Jx,Jy,Jz)
    diS=dense.brickwork.heisenberg_Sa(t,Jx,Jy,Jz)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_heisenberg_Sb_L1(seed_rng):
    t=1
    Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze=np.random.normal(size=(9,))+np.random.normal(size=(9,))*1.0j
    init = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    final = np.random.normal(size=(4,4)) + 1.0j * np.random.normal(size=(4,4))
    miS=mps.brickwork.heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze,init,final)
    diS=dense.brickwork.heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze,init,final)
    assert miS.to_dense()==pytest.approx(diS)

def test_mps_heisenberg_Sa_L1(seed_rng):
    t=1
    Jx,Jy,Jz=np.random.normal(size=(3,))+np.random.normal(size=(3,))*1.0j
    miS=mps.brickwork.heisenberg_Sa(t,Jx,Jy,Jz)
    diS=dense.brickwork.heisenberg_Sa(t,Jx,Jy,Jz)
    assert miS.to_dense()==pytest.approx(diS)

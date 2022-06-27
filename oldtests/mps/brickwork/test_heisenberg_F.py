import imcode.mps as mps
import imcode.dense as dense
import numpy as np
import pytest
def test_four_site_mps_heisenberg_F(seed_rng):
    L=4
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dhF=dense.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=mps.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.to_dense())

    dhF=dense.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    mhF=mps.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    assert dhF==pytest.approx(mhF.to_dense())

def test_five_site_mps_heisenberg_F(seed_rng):
    L=5
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dhF=dense.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=mps.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.to_dense())

    dhF=dense.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    mhF=mps.brickwork.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    assert dhF==pytest.approx(mhF.to_dense())

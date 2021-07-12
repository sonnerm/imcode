import imcode.dense as dense
import numpy as np
import pytest
def test_two_site_heisenberg_H(seed_rng):
    Jx,Jy,Jz,hx1,hy1,hz1,hx2,hy2,hz2=np.random.normal(size=9)+1.0j*np.random.normal(size=9)
    diH=dense.brickwork.heisenberg_H(2,[Jx],[Jy],[Jz],[hx1,hx2],[hy1,hy2],[hz1,hz2])
    assert diH==pytest.approx(dense.brickwork.heisenberg_gate(Jx,Jy,Jz,hx1,hy1,hz1,hx2,hy2,hz2))

def test_one_site_heisenberg_H(seed_rng):
    hx,hy,hz=np.random.normal(size=3)+1.0j*np.random.normal(size=3)
    diH=dense.brickwork.heisenberg_H(1,[],[],[],[hx],[hy],[hz])
    assert diH==pytest.approx(dense.brickwork.heisenberg_lop(hx,hy,hz))

def test_four_site_open_heisenberg_H(seed_rng):
    Jx,Jy,Jz=np.random.normal(size=(3,3))+1.0j*np.random.normal(size=(3,3))
    hx,hy,hz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    diH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)

def test_four_site_per_heisenberg_H(seed_rng):
    Jx,Jy,Jz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    hx,hy,hz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    diH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)

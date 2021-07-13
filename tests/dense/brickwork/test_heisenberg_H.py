import imcode.dense as dense
import numpy as np
from imcode.dense import SX,SY,SZ,ID
import pytest
def test_two_site_heisenberg_H(seed_rng):
    Jx,Jy,Jz,hx1,hy1,hz1,hx2,hy2,hz2=np.random.normal(size=9)+1.0j*np.random.normal(size=9)
    diH=dense.brickwork.heisenberg_H(2,[Jx],[Jy],[Jz],[hx1,hx2],[hy1,hy2],[hz1,hz2])
    miH=Jx*np.kron(SX,SX)+Jy*np.kron(SY,SY)+Jz*np.kron(SZ,SZ)
    miH+=hx1*np.kron(SX,ID)+hy1*np.kron(SY,ID)+hz1*np.kron(SZ,ID)
    miH+=hx2*np.kron(ID,SX)+hy2*np.kron(ID,SY)+hz2*np.kron(ID,SZ)
    assert diH==pytest.approx(miH)

def test_one_site_heisenberg_H(seed_rng):
    hx,hy,hz=np.random.normal(size=3)+1.0j*np.random.normal(size=3)
    diH=dense.brickwork.heisenberg_H(1,[],[],[],[hx],[hy],[hz])
    miH=hx*SX+hy*SY+hz*SZ
    assert diH==pytest.approx(miH)

def test_four_site_open_heisenberg_H(seed_rng):
    Jx,Jy,Jz=np.random.normal(size=(3,3))+1.0j*np.random.normal(size=(3,3))
    hx,hy,hz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    diH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)

def test_four_site_per_heisenberg_H(seed_rng):
    Jx,Jy,Jz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    hx,hy,hz=np.random.normal(size=(3,4))+1.0j*np.random.normal(size=(3,4))
    diH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)

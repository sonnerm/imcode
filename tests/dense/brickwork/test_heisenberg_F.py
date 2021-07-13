import imcode.dense as dense
from imcode.dense import SX,SY,SZ,ID
import numpy as np
import scipy.linalg as la
import pytest
def test_two_site_heisenberg_F(seed_rng):
    hx,hy,hz = np.random.normal(size=(3,2)) + 1.0j * np.random.normal(size=(3,2))
    Jx,Jy,Jz = np.random.normal(size=(3,1)) + 1.0j * np.random.normal(size=(3,1))
    dhF=dense.brickwork.heisenberg_F(2,Jx,Jy,Jz,hx,hy,hz)
    mhF=la.expm(1.0j*(hx[0]*np.kron(SX,ID)+hy[0]*np.kron(SY,ID)+hz[0]*np.kron(SZ,ID)))
    mhF=mhF@la.expm(1.0j*(hx[1]*np.kron(ID,SX)+hy[1]*np.kron(ID,SY)+hz[1]*np.kron(ID,SZ)))
    mhF=mhF@la.expm(1.0j*(Jx[0]*np.kron(SX,SX)+Jy[0]*np.kron(SY,SY)+Jz[0]*np.kron(SZ,SZ)))
    ghF=dense.brickwork.heisenberg_gate(Jx,Jy,Jz,hx[0],hy[0],hz[0],hx[1],hy[1],hz[1])
    dhFT=dense.brickwork.heisenberg_F(2,Jx,Jy,Jz,hx,hy,hz,True)
    assert dhFT==pytest.approx(dhF)
    assert ghF==pytest.approx(dhF)
    assert mhF==pytest.approx(dhF)

def test_four_site_hb_ising_F(seed_rng):
    dt=1e-5
    Jz,hx,hz=np.random.normal(size=(3,4))*dt
    dhF=dense.brickwork.heisenberg_F(4,[0.0]*4,[0.0]*4,Jz,hx,[0.0]*4,hz)
    dhFT=dense.brickwork.heisenberg_F(4,[0.0]*4,[0.0]*4,Jz,hx,[0.0]*4,hz,reversed=True)
    dhH=dense.brickwork.heisenberg_H(4,[0.0]*4,[0.0]*4,Jz,hx,[0.0]*4,hz)
    diF=dense.ising.ising_F(4,Jz,hx,hz)
    assert dhF==pytest.approx(dhFT,rel=10*dt,abs=10*dt**2)
    assert dhF==pytest.approx(la.expm(1.0j*dhH),rel=10*dt,abs=10*dt**2)
    assert dhF==pytest.approx(diF,rel=10*dt,abs=10*dt**2)#not quite the same prescription

def test_four_site_open_heisenberg_F(seed_rng):
    dt=1e-4
    Jx,Jy,Jz=np.random.normal(size=(3,3))*dt+1.0j*np.random.normal(size=(3,3))*dt
    hx,hy,hz=np.random.normal(size=(3,4))*dt+1.0j*np.random.normal(size=(3,4))*dt
    dhH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)
    dhF=dense.brickwork.heisenberg_F(4,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(la.expm(1.0j*dhH),rel=10*dt**2,abs=10*dt**2)
    assert dhF!=pytest.approx(dhF.T.conj())
    assert dhF@dhF.T.conj()!=pytest.approx(np.eye(16))
    dhFT=dense.brickwork.heisenberg_F(4,Jx,Jy,Jz,hx,hy,hz,True)
    assert dhFT==pytest.approx(la.expm(1.0j*dhH),rel=10*dt**2,abs=10*dt**2)
    assert dhFT!=pytest.approx(dhF.T.conj())
    assert dhFT@dhFT.T.conj()!=pytest.approx(np.eye(16))

def test_four_site_per_heisenberg_F(seed_rng):
    dt=1e-5
    Jx,Jy,Jz=np.random.normal(size=(3,4))*dt+1.0j*np.random.normal(size=(3,4))*dt
    hx,hy,hz=np.random.normal(size=(3,4))*dt+1.0j*np.random.normal(size=(3,4))*dt
    dhH=dense.brickwork.heisenberg_H(4,Jx,Jy,Jz,hx,hy,hz)
    dhF=dense.brickwork.heisenberg_F(4,Jx,Jy,Jz,hx,hy,hz)
    mhF=la.expm(1.0j*dhH)
    print(np.max(np.abs(dhF-np.diag(np.diag(dhF)))))
    print(np.max(np.abs(mhF-np.diag(np.diag(mhF)))))
    assert dhF==pytest.approx(la.expm(1.0j*dhH),rel=20*dt**2,abs=20*dt**2)
    assert dhF!=pytest.approx(dhF.T.conj())
    assert dhF@dhF.T.conj()!=pytest.approx(np.eye(16))
    dhFT=dense.brickwork.heisenberg_F(4,Jx,Jy,Jz,hx,hy,hz,True)
    assert dhFT==pytest.approx(la.expm(1.0j*dhH),rel=20*dt**2,abs=20*dt**2)
    assert dhFT!=pytest.approx(dhF.T.conj())
    assert dhFT@dhFT.T.conj()!=pytest.approx(np.eye(16))

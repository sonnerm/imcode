import imcode
import scipy.linalg as scla
from imcode import SX,SZ,SY,ID
import numpy as np
import pytest
import functools
import ttarray as tt
def simple_heisenberg_F(L, Jx,Jy,Jz, hx, hy, hz,reversed=False):
    Jxe,Jye,Jze=Jx.copy(),Jy.copy(),Jz.copy()
    Jxe[1::2],Jye[1::2],Jze[1::2]=0,0,0
    Jxo,Jyo,Jzo=Jx.copy(),Jy.copy(),Jz.copy()
    Jxo[::2],Jyo[::2],Jzo[::2]=0,0,0
    Fe=scla.expm(1j * np.array(imcode.heisenberg_H(L,Jxe,Jye,Jze)))
    Fm=scla.expm(1j* np.array(imcode.heisenberg_H(L,0,0,0,hx,hy,hz)))
    Fo=scla.expm(1j * np.array(imcode.heisenberg_H(L,Jxo,Jyo,Jzo)))
    if reversed:
        return Fe@Fm@Fo
    else:
        return Fo@Fe@Fm

def simple_heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz):
    ret = np.zeros((2**L,2**L),dtype=complex)
    for i in range(L):
        if i<L-1:
            ret += Jx[i]*mkron([ID]*i+[SX,SX]+[ID]*(L-i-2))
            ret += Jy[i]*mkron([ID]*i+[SY,SY]+[ID]*(L-i-2))
            ret += Jz[i]*mkron([ID]*i+[SZ,SZ]+[ID]*(L-i-2))
        ret += hx[i]*mkron([ID]*i+[SX]+[ID]*(L-i-1))
        ret += hy[i]*mkron([ID]*i+[SY]+[ID]*(L-i-1))
        ret += hz[i]*mkron([ID]*i+[SZ]+[ID]*(L-i-1))
    return ret
def mkron(args):
    return functools.reduce(np.kron,args)

def test_one_site_heisenberg(seed_rng):
    L=1
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dhF=simple_heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    assert dhF==pytest.approx(mhF.todense())
@pytest.mark.skip
def test_heisenberg_F_trotter(seed_rng):
    L=5
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dt=0.001
    miF=np.array(imcode.heisenberg_F(L,Jx*dt,Jy*dt,Jz*dt,hx*dt,hy*dt,hz*dt))
    miH=np.array(imcode.heisenberg_H(L,Jx*dt,Jy*dt,Jz*dt,hx*dt,hy*dt,hz*dt))
    assert scla.expm(1.0j*miH)==pytest.approx(miF,rel=10*dt)
def test_four_site_heisenberg(seed_rng):
    L=4
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dhF=simple_heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())

def test_five_site_heisenberg(seed_rng):
    L=5
    Jx,Jy,Jz,hx,hy,hz=np.random.normal(size=(6,L))+np.random.normal(size=(6,L))*1.0j
    Jx,Jy,Jz=Jx[:-1],Jy[:-1],Jz[:-1]
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz,reversed=True)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_F(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())
    dhF=simple_heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    mhF=imcode.heisenberg_H(L,Jx,Jy,Jz,hx,hy,hz)
    assert dhF==pytest.approx(mhF.todense())

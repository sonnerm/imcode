import numpy as np
import scipy.linalg as la
from .utils import dense_kron,SX,SY,SZ,ID
from .brickwork import brickwork_L,brickwork_S,brickwork_T

def heisenberg_H(Jx,Jy,Jz,hx,hy,hz):
    L=len(hx) # maybe change to explicit length?
    assert len(hy)==L
    assert len(hz)==L
    assert len(Jx)==len(Jy)
    assert len(Jx)==len(Jz)
    Jx,Jy,Jz,hx,hy,hz=np.array(Jx),np.array(Jy),np.array(Jz),np.array(hx),np.array(hy),np.array(hz),
    ret=np.zeros((2**L,2**L),dtype=np.common_type(Jx,Jy,Jz,hx,hy,hz,np.array(1.0j)))
    for i,Jxi,Jyi,Jzi in zip(range(L-1),Jx[:L-1],Jy[:L-1],Jz[:L-1]):
        ret+=Jxi*dense_kron([ID]*i+[SX]+[SX]+[ID]*(L-i-2))
        ret+=Jyi*dense_kron([ID]*i+[SY]+[SY]+[ID]*(L-i-2))
        ret+=Jzi*dense_kron([ID]*i+[SZ]+[SZ]+[ID]*(L-i-2))
    if len(Jx)==L and L>1:
        ret+=Jx[-1]*dense_kron([SX]+[ID]*(L-2)+[SX])
        ret+=Jy[-1]*dense_kron([SY]+[ID]*(L-2)+[SY])
        ret+=Jz[-1]*dense_kron([SZ]+[ID]*(L-2)+[SZ])
    for i,hxi,hyi,hzi in zip(range(L),hx,hy,hz):
        ret+=hxi*dense_kron([ID]*i+[SX]+[ID]*(L-i-1))
        ret+=hyi*dense_kron([ID]*i+[SY]+[ID]*(L-i-1))
        ret+=hzi*dense_kron([ID]*i+[SZ]+[ID]*(L-i-1))
    return ret

def heisenberg_F(Jx,Jy,Jz,hx,hy,hz):
    evs=[np.array(x).copy() for x in (Jx,Jy,Jz)]
    for x in evs:
        x[1::2]=0
    evs+=[np.array(x)/2 for x in (hx,hy,hz)]
    ods=[np.array(x).copy() for x in (Jx,Jy,Jz)]
    for x in ods:
        x[::2]=0
    ods+=[np.array(x)/2 for x in (hx,hy,hz)]
    return la.expm(1.0j*heisenberg_H(*evs))@la.expm(1.0j*heisenberg_H(*ods))
def heisenberg_gate(Jx,Jy,Jz):
    H=np.kron(SX,SX)*Jx+np.kron(SX,SX)*Jx+np.kron(SX,SX)*Jy
    print(H)
    return la.expm(1.0j*np.array(H)).reshape((2,2,2,2)).transpose([0,2,1,3])

def heisenberg_lop(hx,hy,hz):
    return SX*hx+SY*hy+SZ*hz
def heisenberg_S(t,Jx,Jy,Jz):
    return brickwork_S(t,heisenberg_gate(Jx,Jy,Jz))
def heisenberg_L(t,hx,hy,hz,init=(0.5,0.5)):
    return brickwork_L(t,heisenberg_lop(hx/2,hy/2,hz/2))
def heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz,init=(0.5,0.5)):
    return brickwork_T(t,heisenberg_gate(Jx,Jy,Jz),heisenberg_lop(hx/2,hy/2,hz/2))

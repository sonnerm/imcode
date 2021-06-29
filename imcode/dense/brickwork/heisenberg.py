import numpy as np
import scipy.linalg as la
from .utils import kron,SX,SY,SZ,ID
from .brickwork import brickwork_Sa,brickwork_Sb,brickwork_T,brickwork_La,brickwork_Lb,brickwork_F

def heisenberg_H(Jx,Jy,Jz,hx,hy,hz):
    L=len(hx) # maybe change to explicit length?
    assert len(hy)==L
    assert len(hz)==L
    assert len(Jx)==len(Jy)
    assert len(Jx)==len(Jz)
    Jx,Jy,Jz,hx,hy,hz=np.array(Jx),np.array(Jy),np.array(Jz),np.array(hx),np.array(hy),np.array(hz),
    ret=np.zeros((2**L,2**L),dtype=np.common_type(Jx,Jy,Jz,hx,hy,hz,np.array(1.0j)))
    for i,Jxi,Jyi,Jzi in zip(range(L-1),Jx[:L-1],Jy[:L-1],Jz[:L-1]):
        ret+=Jxi*kron([ID]*i+[SX]+[SX]+[ID]*(L-i-2))
        ret+=Jyi*kron([ID]*i+[SY]+[SY]+[ID]*(L-i-2))
        ret+=Jzi*kron([ID]*i+[SZ]+[SZ]+[ID]*(L-i-2))
    if len(Jx)==L and L>1:
        ret+=Jx[-1]*kron([SX]+[ID]*(L-2)+[SX])
        ret+=Jy[-1]*kron([SY]+[ID]*(L-2)+[SY])
        ret+=Jz[-1]*kron([SZ]+[ID]*(L-2)+[SZ])
    for i,hxi,hyi,hzi in zip(range(L),hx,hy,hz):
        ret+=hxi*kron([ID]*i+[SX]+[ID]*(L-i-1))
        ret+=hyi*kron([ID]*i+[SY]+[ID]*(L-i-1))
        ret+=hzi*kron([ID]*i+[SZ]+[ID]*(L-i-1))
    return ret

def heisenberg_F(Jx,Jy,Jz,hx,hy,hz):
    gates=[heisenberg_gate(jx,jy,jz) for (jx,jy,jz) in zip(Jx,Jy,Jz)]
    lop=[heisenberg_lop(chx,chy,chz) for (chx,chy,chz) in zip(hx,hy,hz)]
    return brickwork_F(gates,lop)
def heisenberg_lop(hx,hy,hz):
    return la.expm(1j*(SX*hx+SY*hy+SZ*hz))
def heisenberg_gate(Jx,Jy,Jz,hx=0,hy=0,hz=0,hxe=None,hye=None,hze=None):
    H=np.kron(SX,SX)*Jx+np.kron(SY,SY)*Jy+np.kron(SZ,SZ)*Jz
    if hxe is None:
        hxe=hx
    if hye is None:
        hye=hy
    if hze is None:
        hze=hz
    lop1=heisenberg_lop(hx,hy,hz)
    lop2=heisenberg_lop(hxe,hye,hze)
    return np.kron(lop1,lop2)@la.expm(1.0j*np.array(H))
def heisenberg_Sa(t,Jx,Jy,Jz):
    return brickwork_Sa(t,heisenberg_gate(Jx,Jy,Jz))
def heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,init=np.eye(4),final=np.eye(4)):
    return brickwork_Sb(t,heisenberg_gate(Jx,Jy,Jz,hx,hy,hz),init,final)
def heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz,init=np.eye(4),final=np.eye(4)):
    return brickwork_T(t,heisenberg_gate(Jx,Jy,Jz),heisenberg_gate(Jx,Jy,Jz,hx,hy,hz),init,final)
def heisenberg_La(t):
    return brickwork_La(t)
def heisenberg_Lb(t,hx,hy,hz,init=np.eye(2),final=np.eye(2)):
    return brickwork_Lb(t,heisenberg_lop(hx,hy,hz),init,final)

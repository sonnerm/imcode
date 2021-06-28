import numpy as np
import scipy.linalg as la
from ...dense import dense_kron,SX,SY,SZ,ID,heisenberg_lop,heisenberg_gate
from . import brickwork_Sa,brickwork_Sb,brickwork_T,brickwork_La,brickwork_Lb
def heisenberg_Sa(t,Jx,Jy,Jz):
    return brickwork_Sa(t,heisenberg_gate(Jx,Jy,Jz))
def heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,Jxe=None,Jye=None,Jze=None,hxe=None,hye=None,hze=None,init=np.eye(4),final=np.eye(4)):
    if Jxe is None:
        Jxe=Jx
    if Jye is None:
        Jye=Jy
    if Jze is None:
        Jze=Jz

    if hxe is None:
        hxe=hx
    if hye is None:
        hye=hy
    if hze is None:
        hze=hz
    return brickwork_Sb(t,heisenberg_gate(Jx,Jy,Jz,hx,hy,hz),init,final)
def heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz,Jxe=None,Jye=None,Jze=None,hxe=None,hye=None,hze=None,init=np.eye(4),final=np.eye(4)):
    if Jxe is None:
        Jxe=Jx
    if Jye is None:
        Jye=Jy
    if Jze is None:
        Jze=Jz

    if hxe is None:
        hxe=hx
    if hye is None:
        hye=hy
    if hze is None:
        hze=hz
    return brickwork_T(t,heisenberg_gate(Jxe,Jye,Jze),heisenberg_gate(Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze),init,final)
def heisenberg_La(t):
    return brickwork_La(t)
def heisenberg_Lb(t,hx,hy,hz,init=np.eye(2),final=np.eye(2)):
    return brickwork_Lb(t,heisenberg_lop(hx,hy,hz),init,final)
def heisenberg_F(Jx,Jy,Jz,hx=None,hy=None,hz=None):
    gates=[]
    assert len(hx)%2==0
    assert len(Jx)==len(Jy)
    assert len(Jx)==len(Jz)
    assert len(hx)==len(Jx)+1
    assert len(hx)==len(hy)
    assert len(hx)==len(hz)
    for i in range(len(Jx)//2):
        gates.append(heisenberg_gate(Jx[2*i],Jy[2*i],Jz[2*i],hx[2*i],hy[2*i],hz[2*i],hx[2*i+1],hy[2*i+1],hz[2*i+1]))
        gates.append(heisenberg_gate(Jx[2*i+1],Jy[2*i+1],Jz[2*i+1]))
    gates.append(heisenberg_gate(Jx[-1],Jy[-1],Jz[-1],hx[-2],hy[-2],hz[-2],hx[-1],hy[-1],hz[-1]))
    return brickwork_F(gates)

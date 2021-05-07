import numpy as np
import scipy.linalg as la
from ...dense import dense_kron,SX,SY,SZ,ID,heisenberg_lop,heisenberg_gate
from . import brickwork_Sa,brickwork_Sb,brickwork_T,brickwork_La,brickwork_Lb
def heisenberg_Sa(t,Jx,Jy,Jz):
    return brickwork_Sa(t,heisenberg_gate(Jx,Jy,Jz))
def heisenberg_Sb(t,Jx,Jy,Jz,hx,hy,hz,init=np.eye(4),final=np.eye(4)):
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

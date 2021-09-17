import numpy as np
import scipy.linalg as la
from ...dense.brickwork import heisenberg_lop,heisenberg_gate
from ...dense import SX,SY,SZ,ID,unitary_channel
from .brickwork import brickwork_Sa,brickwork_Sb,brickwork_La,brickwork_Lb,brickwork_F,brickwork_H
def heisenberg_F(L,Jx,Jy,Jz,hx=None,hy=None,hz=None,reversed=False):
    ogates=[heisenberg_gate(jx,jy,jz) for (jx,jy,jz) in zip(Jx[1::2],Jy[1::2],Jz[1::2])]
    egates=[heisenberg_gate(jx,jy,jz,hxe,hye,hze,hxo,hyo,hzo) for (jx,jy,jz,hxe,hye,hze,hxo,hyo,hzo) in zip(Jx[::2],Jy[::2],Jz[::2],hx[::2],hy[::2],hz[::2],hx[1::2],hy[1::2],hz[1::2])]
    gates=[None]*(len(ogates)+len(egates))
    gates[::2]=egates
    gates[1::2]=ogates
    return brickwork_F(L,gates,reversed)
def heisenberg_H(L,Jx,Jy,Jz,hx=None,hy=None,hz=None):
    gates=[]
    SX2=np.kron(SX,SX)
    SY2=np.kron(SY,SY)
    SZ2=np.kron(SZ,SZ)
    for i in range(L-1):
        gates.append(Jx*SX2+Jy*SY2+Jz*SZ2)
    return brickwork_H(L,gates)

def heisenberg_Sa(t,Jx,Jy,Jz):
    return brickwork_Sa(t,unitary_channel(heisenberg_gate(Jx,Jy,Jz)))
def heisenberg_Sb(t,Jx,Jy,Jz,hx=None,hy=None,hz=None,hxe=None,hye=None,hze=None,init=np.eye(4)/4,final=np.eye(4)):
    return brickwork_Sb(t,unitary_channel(heisenberg_gate(Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze)),init,final)
# def heisenberg_T(t,Jx,Jy,Jz,hx,hy,hz,Jxe=None,Jye=None,Jze=None,hxe=None,hye=None,hze=None,init=np.eye(4),final=np.eye(4)):
#     if Jxe is None:
#         Jxe=Jx
#     if Jye is None:
#         Jye=Jy
#     if Jze is None:
#         Jze=Jz
#
#     if hxe is None:
#         hxe=hx
#     if hye is None:
#         hye=hy
#     if hze is None:
#         hze=hz
#     return brickwork_T(t,heisenberg_gate(Jxe,Jye,Jze),heisenberg_gate(Jx,Jy,Jz,hx,hy,hz,hxe,hye,hze),init,final)
def heisenberg_La(t):
    return brickwork_La(t)
def heisenberg_Lb(t,hx,hy,hz,init=np.eye(2)/2,final=np.eye(2)):
    return brickwork_Lb(t,unitary_channel(heisenberg_lop(hx,hy,hz)),init,final)

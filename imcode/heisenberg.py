import numpy as np
from . import SX,SY,SZ,ID
from .channel import unitary_channel
from .brickwork import brickwork_Sa,brickwork_Sb,brickwork_La,brickwork_Lb,brickwork_F,brickwork_H
import scipy.linalg as sla
def heisenberg_gate(Jx=0,Jy=0,Jz=0,hx1=0,hy1=0,hz1=0,hx2=0,hy2=0,hz2=0):
    H=np.kron(SX,SX)*Jx+np.kron(SY,SY)*Jy+np.kron(SZ,SZ)*Jz
    lop1=sla.expm(1.0j*hx1*SX+hy1*SY+hz1*SZ)
    lop2=sla.expm(1.0j*hx2*SX+hy2*SY+hz2*SZ)
    return np.kron(lop1,lop2)@sla.expm(1.0j*np.array(H))
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
def heisenberg_La(t):
    return brickwork_La(t)
def heisenberg_Lb(t,hx,hy,hz,init=np.eye(2)/2,final=np.eye(2)):
    return brickwork_Lb(t,unitary_channel(heisenberg_lop(hx,hy,hz)),init,final)

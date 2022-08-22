import numpy as np
import ttarray as tt
from . import SX,SY,SZ,ID
from .channel import unitary_channel
from .brickwork import brickwork_Te,brickwork_To,brickwork_F,brickwork_H
import scipy.linalg as sla
'''
    Order: first local gates, then even gates, then odd gates
'''

def heisenberg_gate(Jx=0,Jy=0,Jz=0,hx1=0,hy1=0,hz1=0,hx2=0,hy2=0,hz2=0):
    H=np.kron(SX,SX)*Jx+np.kron(SY,SY)*Jy+np.kron(SZ,SZ)*Jz
    lop1=sla.expm(1.0j*hx1*SX+1.0j*hy1*SY+1.0j*hz1*SZ)
    lop2=sla.expm(1.0j*hx2*SX+1.0j*hy2*SY+1.0j*hz2*SZ)
    return sla.expm(1.0j*np.array(H))@np.kron(lop1,lop2)
def heisenberg_F(L,Jx,Jy,Jz,hx=0.0,hy=0.0,hz=0.0,reversed=False):
    Jx,Jy,Jz,hx,hy,hz=(np.asarray(k) for k in (Jx,Jy,Jz,hx,hy,hz))
    hx,hy,hz=(np.tile(h,L) if len(h.shape)==0 else h for h in (hx,hy,hz))
    Jx,Jy,Jz=(np.tile(J,L-1) if len(J.shape)==0 else J for J in (Jx,Jy,Jz))
    if L==1:
        return tt.array(sla.expm(1.0j*hx[0]*SX+1.0j*hy[0]*SY+1.0j*hz[0]*SZ))
    ogates=[heisenberg_gate(jx,jy,jz) for (jx,jy,jz) in zip(Jx[1::2],Jy[1::2],Jz[1::2])]
    egates=[heisenberg_gate(jx,jy,jz,hxe,hye,hze,hxo,hyo,hzo) for (jx,jy,jz,hxe,hye,hze,hxo,hyo,hzo) in zip(Jx[::2],Jy[::2],Jz[::2],hx[::2],hy[::2],hz[::2],hx[1::2],hy[1::2],hz[1::2])]
    gates=[None]*(len(ogates)+len(egates))
    gates[::2]=egates
    gates[1::2]=ogates
    if L%2==1:
        if reversed:
            gates[-1]=np.kron(np.eye(2),sla.expm(1j*hx[-1]*SX+1j*hy[-1]*SY+1j*hz[-1]*SZ))@gates[-1]
        else:
            gates[-1]=gates[-1]@np.kron(np.eye(2),sla.expm(1j*hx[-1]*SX+1j*hy[-1]*SY+1j*hz[-1]*SZ))
    return brickwork_F(L,gates,reversed)
def heisenberg_H(L,Jx,Jy,Jz,hx=0.0,hy=0.0,hz=0.0):

    Jx,Jy,Jz,hx,hy,hz=(np.asarray(k) for k in (Jx,Jy,Jz,hx,hy,hz))
    hx,hy,hz=(np.tile(h,L) if len(h.shape)==0 else h for h in (hx,hy,hz))
    Jx,Jy,Jz=(np.tile(J,L-1) if len(J.shape)==0 else J for J in (Jx,Jy,Jz))
    if L==1:
        return tt.array(hx[0]*SX+hy[0]*SY+hz[0]*SZ)
    gates=[]
    SX2=np.kron(SX,SX)
    SY2=np.kron(SY,SY)
    SZ2=np.kron(SZ,SZ)

    SX1=np.kron(SX,ID)
    SY1=np.kron(SY,ID)
    SZ1=np.kron(SZ,ID)
    for Jxc,Jyc,Jzc,hxc,hyc,hzc in zip(Jx,Jy,Jz,hx,hy,hz):
        gates.append(Jxc*SX2+Jyc*SY2+Jzc*SZ2+hxc*SX1+hyc*SY1+hzc*SZ1)
    gates[-1]+=hx[-1]*np.kron(ID,SX)+hy[-1]*np.kron(ID,SY)+hz[-1]*np.kron(ID,SZ)
    return brickwork_H(L,gates)

def heisenberg_Te(t,Jx,Jy,Jz,hx=0,hy=0,hz=0):
    return brickwork_Te(t,unitary_channel(heisenberg_gate(Jx,Jy,Jz,0.0,0.0,0.0,hx,hy,hz)))
def heisenberg_To(t,Jx,Jy,Jz,hx=0,hy=0,hz=0):
    return brickwork_To(t,unitary_channel(heisenberg_gate(Jx,Jy,Jz,0.0,0.0,0.0,hx,hy,hz)))
# def heisenberg_La(t):
#     return brickwork_La(t)
# def heisenberg_Lb(t,hx,hy,hz,init=np.eye(2)/2,final=np.eye(2)):
#     return brickwork_Lb(t,unitary_channel(heisenberg_lop(hx,hy,hz)),init,final)

import numpy as np
import ttarray as tt
from . import SX,SZ,ID,ZE
from . import brickwork
def ising_H(L,J,g,h):
    J=np.asarray(J)
    g=np.asarray(g)
    h=np.asarray(h)
    if len(g.shape)==0:
        g=np.tile(g,L)
    if len(h.shape)==0:
        h=np.tile(h,L)
    if L==1:
        return tt.fromproduct([np.array(h[0]*SZ+g[0]*SX)])
    if len(J.shape)==0:
        J=np.tile(J,L-1)
    Wsa=np.array([[ID,J[0]*SZ,h[0]*SZ+g[0]*SX]]).transpose([0,2,3,1])
    Ws=[Wsa]
    for i in range(1,L-1):
        Wsc=np.array([[ID,J[i]*SZ,h[i]*SZ+g[i]*SX],[ZE,ZE,SZ],[ZE,ZE,ID]]).transpose([0,2,3,1])
        Ws.append(Wsc)
    Wsb=np.array([[h[-1]*SZ+g[-1]*SX],[SZ],[ID]]).transpose([0,2,3,1])
    Ws.append(Wsb)
    return tt.frommatrices(Ws)

def ising_F(L,J,g,h):
    J=np.asarray(J)
    g=np.asarray(g)
    h=np.asarray(h)
    if len(g.shape)==0:
        g=np.tile(g,L)
    if len(h.shape)==0:
        h=np.tile(h,L)
    Wloc=[np.diag([np.exp(1.0j*hc),np.exp(-1.0j*hc)])@np.array([[np.cos(gc),1.0j*np.sin(gc)],[1.0j*np.sin(gc),np.cos(gc)]]) for hc,gc in zip(h,g)]
    mploc=tt.fromproduct(Wloc)
    if L==1:
        return mploc
    if len(J.shape)==0:
        J=np.tile(J,L-1)
    mpJ=brickwork.brickwork_F(L,[np.diag([np.exp(1.0j*Jc),np.exp(-1.0j*Jc),np.exp(-1.0j*Jc),np.exp(1.0j*Jc)]) for Jc in J])
    return mpJ@mploc
def ising_zoz(J,g,h):
    pass

def ising_lop(t,ch):
    pass
def ising_J(t,J):
    pass

def ising_T(t,J,g,h):
    return ising_J(t,J)@ising_lop(t,g,h)

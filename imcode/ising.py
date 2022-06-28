import numpy as np
import ttarray as tt
from . import SX,SZ,ID
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
    Wg=[np.array([[[[np.cos(gc),1.0j*np.sin(gc)],[1.0j*np.sin(gc),np.cos(gc)]]]]) for gc in g]
    Wh=[np.array([[[[np.exp(1.0j*hc),0],[0,np.exp(-1.0j*hc)]]]]) for hc in h]
    if L==1:
        return multiply_mpos([mph,mpg])
    if len(J.shape)==0:
        J=np.tile(J,L-1)
    UP=np.array([[1,0],[0,0]])
    DO=np.array([[0,0],[0,1]])
    WJ=[np.array([[UP,DO]])]
    for Jc in J[:-1]:
        wjc=np.array([[UP*np.exp(1.0j*Jc),DO*np.exp(-1.0j*Jc)],[UP*np.exp(-1.0j*Jc),DO*np.exp(1.0j*Jc)]])
        WJ.append(wjc)
    UI=np.array([[np.exp(1.0j*J[-1]),0],[0,np.exp(-1.0j*J[-1])]])
    DI=np.array([[np.exp(-1.0j*J[-1]),0],[0,np.exp(1.0j*J[-1])]])
    wjc=np.array([[UI],[DI]])
    WJ.append(wjc)
    mpJ=tt.frommatrices(WJ)
    return mpJ@mph@mpg
def ising_gate(J,g,h):
    pass
def ising_lop(t,ch,init=np.eye(2)/2,final=np.eye(2)):
    pass
def ising_J(t,J):
    pass

def ising_T(t,J,g,h,init=np.eye(2)/2,final=np.eye(2)):
    init=np.array(init)
    if len(init.shape)==2:
        init=init[None,...,None]
    return (ising_J(t,J)@ising_g(t,g,init,final)@ising_h(t,h)).contract()

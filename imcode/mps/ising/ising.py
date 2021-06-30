from .. import MPO
import numpy as np
from ...dense import ID,SZ,SX
ZE=np.zeros_like(ID)
def ising_H(J,g,h):
    L=len(h) # maybe change to explicit length?
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    if L==1:
        Wsn=np.array([[h[0]*SZ+g[0]*SX]])
        return MPO([Wsn])
    if len(J)==L and L>1:
        raise NotImplementedError("PBE's don't work yet")
    Wsa=np.array([[ID,J[0]*SZ,h[0]*SZ+g[0]*SX]])
    Ws=[Wsa]
    for i in range(1,L-1):
        Wsc=np.array([[ID,J[i]*SZ,h[i]*SZ+g[i]*SX],[ZE,ZE,SZ],[ZE,ZE,ID]])
        Ws.append(Wsc)
    Wsb=np.array([[h[-1]*SZ+g[-1]*SX],[SZ],[ID]])
    Ws.append(Wsb)
    return MPO.from_matrices(Ws)
def ising_F(J,g,h):
    L=len(h) # maybe change to explicit length?
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    Wg=[np.array([[[[np.cos(gc),1.0j*np.sin(gc)],[1.0j*np.sin(gc),np.cos(gc)]]]]) for gc in g]
    mpg=MPO.from_matrices(Wg)
    Wh=[np.array([[[[np.exp(1.0j*hc),0],[0,np.exp(-1.0j*hc)]]]]) for hc in h]
    mph=MPO.from_matrices(Wh)
    if len(J)==L and L>1:
        raise NotImplementedError("PBE's don't work yet")
    if L==1:
        return multiply_mpos([mph,mpg])
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
    mpJ=MPO.from_matrices(WJ)
    return (mpJ@mph@mpg).contract()
def ising_W(t,g,init=np.eye(2)/2,final=np.eye(2)):
    pass
#TODO refactor

def ising_h(t,h):
    k=-np.conj(h)
    Ha=np.array([[np.diag(np.exp(1.0j*np.array([h+k,h-k,-h+k,-h-k])))]])
    return MPO.from_matrices([Ha]*(t))

def ising_J(t,J):
    K=-np.conj(J)
    Jprim=np.array([[+J+K,+J-K,-J+K,-J-K],
                    [+J-K,+J+K,-J-K,-J+K],
                    [-J+K,-J-K,+J+K,+J-K],
                    [-J-K,-J+K,+J-K,+J+K]
    ])
    Ja=np.array([[np.exp(1.0j*Jprim)]])
    return MPO.from_matrices([Ja]*t)
def ising_T(t,J,g,h,init=np.eye(2)/2,final=np.eye(2)):
    return (ising_J(t,J)@ising_W(t,g,init,final)@ising_h(t,h)).contract()
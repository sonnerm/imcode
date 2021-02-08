import numpy as np
from .flat import FlatSite
from tenpy.linalg.charges import LegCharge
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from ..dense import SZ,ID,SX
ZE=np.zeros_like(ID)
def ising_H(J,g,h):
    L=len(h) # maybe change to explicit length?
    sites=[FlatSite() for _ in range(L)]
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    leg_p=LegCharge.from_trivial(2)
    leg_m=LegCharge.from_trivial(3)
    leg_b=LegCharge.from_trivial(1)
    if L==1:
        Wsn=np.array([[h[0]*SZ+g[0]*SX]])
        Ws=[npc.Array.from_ndarray(Wsn,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
        return MPO(sites,Ws)
    if len(J)==L and L>1:
        raise NotImplementedError("PBE's don't work yet")
    Wsa=np.array([[ID,J[0]*SZ,h[0]*SZ+g[0]*SX]])
    Ws=[npc.Array.from_ndarray(Wsa,[leg_b,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
    for i in range(1,L-1):
        Wsc=np.array([[ID,J[i]*SZ,h[i]*SZ+g[i]*SX],[ZE,ZE,SZ],[ZE,ZE,ID]])
        Ws.append(npc.Array.from_ndarray(Wsc,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    Wsb=np.array([[h[-1]*SZ+g[-1]*SX],[SZ],[ID]])
    Ws.append(npc.Array.from_ndarray(Wsb,[leg_m,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    return MPO(sites,Ws,bc="finite")
def ising_F(J,g,h):
    L=len(h) # maybe change to explicit length?
    sites=[FlatSite() for _ in range(L)]
    J=np.array(J)
    g=np.array(g)
    h=np.array(h)
    leg_p=LegCharge.from_trivial(2)
    leg_m=LegCharge.from_trivial(3)
    leg_b=LegCharge.from_trivial(1)
    Wg=[np.array([[[[np.cos(gc),1.0j*np.sin(gc)],[1.0j*np.sin(gc),np.cos(gc)]]]]) for gc in g]
    Wg=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in Wg]
    mpg=MPO(Sites,Wg,bc="finite")
    Wh=[np.array([[[[np.exp(1.0j*hc),0],[0,np.exp(-1.0j*hc)]]]]) for hc in h]
    Wh=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in Wh]
    mph=MPO(Sites,Wh,bc="finite")
    if len(J)==L and L>1:
        raise NotImplementedError("PBE's don't work yet")
    if L==1:
        return multiply_mpos([mph,mpg])
    WJ=[]
    return multiply_mpos([mpJ,mph,mpg])

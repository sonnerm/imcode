from .. import MPO
import numpy as np
ID=np.eye(2)
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
    leg_b=LegCharge.from_trivial(1)
    Wg=[np.array([[[[np.cos(gc),1.0j*np.sin(gc)],[1.0j*np.sin(gc),np.cos(gc)]]]]) for gc in g]
    Wg=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in Wg]
    mpg=MPO(sites,Wg,bc="finite")
    Wh=[np.array([[[[np.exp(1.0j*hc),0],[0,np.exp(-1.0j*hc)]]]]) for hc in h]
    Wh=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in Wh]
    mph=MPO(sites,Wh,bc="finite")
    if len(J)==L and L>1:
        raise NotImplementedError("PBE's don't work yet")
    if L==1:
        return multiply_mpos([mph,mpg])
    UP=np.array([[1,0],[0,0]])
    DO=np.array([[0,0],[0,1]])
    WJ=[npc.Array.from_ndarray(np.array([[UP,DO]]),[leg_b,leg_p.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])]
    for Jc in J[:-1]:
        wjc=np.array([[UP*np.exp(1.0j*Jc),DO*np.exp(-1.0j*Jc)],[UP*np.exp(-1.0j*Jc),DO*np.exp(1.0j*Jc)]])
        WJ.append(npc.Array.from_ndarray(wjc,[leg_p,leg_p.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    UI=np.array([[np.exp(1.0j*J[-1]),0],[0,np.exp(-1.0j*J[-1])]])
    DI=np.array([[np.exp(-1.0j*J[-1]),0],[0,np.exp(1.0j*J[-1])]])
    wjc=np.array([[UI],[DI]])
    WJ.append(npc.Array.from_ndarray(wjc,[leg_p,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]))
    mpJ=MPO(sites,WJ,bc="finite")
    return multiply_mpos([mpJ,mph,mpg])
def ising_W(t,g,init=(0.5,0.5)):
    s2=np.abs(np.sin(g))**2
    c2=np.abs(np.cos(g))**2
    mx=-1.0j*np.conj(np.sin(g))*np.cos(g)
    px=1.0j*np.sin(g)*np.conj(np.cos(g))
    Wprim=np.array([[c2,px,mx,s2],
                    [px,c2,s2,mx],
                    [mx,s2,c2,px],
                    [s2,mx,px,c2]
    ])
    init=np.diag(list(init)+[0,0])
    W_0a=np.einsum("cd,cb,ac->abcd",init,np.eye(4),np.eye(1))
    W_ia=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),Wprim)
    W_Ta=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),Wprim)
    return MPO.from_matrices([W_0a]+[W_ia]*(t-1)+[W_Ta])
#TODO refactor

def ising_h(t,h):
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([np.exp(1.0j*(h-np.conj(h))),np.exp(-1.0j*(h-np.conj(h))),np.exp(1.0j*(h+np.conj(h))),np.exp(-1.0j*(h+np.conj(h)))]))
    return MPO.from_matrices([Ida]+[Ha]*(t-1)+[Ida])

def ising_J(t,J):
    Iprim=np.array([[1.0,1.0,0.0,0.0],[1.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])#/np.sqrt(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    pj=np.exp(2.0j*J)
    mj=np.exp(-2.0j*np.conj(J))
    ip=np.exp(1.0j*(J-np.conj(J)))
    Jprim=np.array([[ip,mj,pj,ip],
                    [mj,ip,ip,pj],
                    [pj,ip,ip,mj],
                    [ip,pj,mj,ip]
    ])
    Ja=np.einsum("ab,cd->abcd",np.eye(1),Jprim)
    return MPO.from_matrices([Ida]+[Ja]*(t-1)+[Ida])
def ising_T(t,J,g,h,init=np.eye(2)/2):
    return (ising_J(t,J)@ising_W(t,g,init)@ising_h(t,h)).contract()

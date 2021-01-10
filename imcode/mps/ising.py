from .utils import BlipSite,multiply_mpos
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc
import numpy as np
def ising_H(J,g,h):
    raise NotImplementedError("Not yet implemented")

def ising_F(J,g,h):
    L=len(h)
    sites=[SpinHalfSite() for _ in range(L)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    leg_m=LegCharge.from_trivial(2)
    Wprim=np.array([])
    raise NotImplementedError("Not yet implemented")
def ising_W(t,g):
    sites=[BlipSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    leg_m=LegCharge.from_trivial(4)
    s2=np.abs(np.sin(g))**2
    c2=np.abs(np.cos(g))**2
    mx=-1.0j*np.conj(np.sin(g))*np.cos(g)
    px=1.0j*np.sin(g)*np.conj(np.cos(g))
    Wprim=np.array([[c2,s2,mx,px],
                    [s2,c2,px,mx],
                    [mx,px,c2,s2],
                    [px,mx,s2,c2]
    ])
    W_0a=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),np.eye(1))
    W_ia=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),Wprim)
    W_Ta=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),Wprim)
    W_0=npc.Array.from_ndarray(W_0a,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    W_i=npc.Array.from_ndarray(W_ia,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    W_T=npc.Array.from_ndarray(W_Ta,[leg_m,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[W_0]+[W_i]*(t-1)+[W_T])

def ising_h(t,h):
    sites=[BlipSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([np.exp(1.0j*(h-np.conj(h))),np.exp(-1.0j*(h-np.conj(h))),np.exp(1.0j*(h+np.conj(h))),np.exp(-1.0j*(h+np.conj(h)))]))
    H=npc.Array.from_ndarray(Ha,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[H]*(t-1)+[Id])

def ising_J(t,J):
    sites=[BlipSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Iprim=np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]])/np.sqrt(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    pj=np.exp(2.0j*J)
    mj=np.exp(-2.0j*np.conj(J))
    ip=np.exp(1.0j*(J-np.conj(J)))
    Jprim=np.array([[ip,ip,pj,mj],
                    [ip,ip,mj,pj],
                    [pj,mj,ip,ip],
                    [mj,pj,ip,ip]
    ])
    Ja=np.einsum("ab,cd->abcd",np.eye(1),Jprim)
    J=npc.Array.from_ndarray(Ja,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[J]*(t-1)+[Id])
def ising_T(t,J,g,h):
    return multiply_mpos([ising_J(t,J),ising_W(t,g),ising_h(t,h)])

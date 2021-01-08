from .utils import multiply_mpos
from .ising import ising_W
from functools import lru_cache
import numpy as np
from tenpy.linalg.charges import LegCharge
@lru_cache(None)
def hr_operator(t):
    tarr=[0]+list(range(t//2+t%2))+list(range(t//2))[::-1]+[0]
    Iprim=np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    legs=[LegCharge.from_qflat(BlipSite(True).chinfo,list(range(-i,i+1))) for i in tarr]
    leg_i1=LegCharge.from_trivial(4,BlipSite(True).chinfo)
    leg_i2=BlipSite(True).leg
    leg_p=sites[0].leg
    leg_t=LegCharge.from_trivial(1,sites[0].chinfo)
    Js=[get_proj(Iprim,lc,ln.conj(),leg_i2,leg_i1.conj()).drop_charge() for lc,ln in zip(legs[1:-2],legs[2:-1])]
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+Js+[Id])
def _get_proj(op,left,right,p,ps):
    preop=np.einsum("ab,cd->abcd",np.ones((left.ind_len,right.ind_len)),op)
    return npc.Array.from_ndarray(preop,[left,right,p,ps],labels=["wL","wR","p","p*"],dtype=complex,qtotal=[0],raise_wrong_sector=False)
def ising_hr_J(t,J):
    tarr=[0]+list(range(T//2+T%2))+list(range(T//2))[::-1]+[0]
    Iprim=np.array([[1.0,1.0,0.0,0.0],[1.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])/np.sqrt(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    pj=np.exp(-2.0j*J)
    mj=np.exp(2.0j*J)
    id=1.0
    Jprim=np.array([[id,id,pj,mj],
                    [id,id,mj,pj],
                    [pj,mj,id,id],
                    [mj,pj,id,id]
    ])
    legs=[tenpy.linalg.charges.LegCharge.from_qflat(BlipSite(True).chinfo,list(range(-i,i+1))) for i in tarr]
    leg_i1=tenpy.linalg.charges.LegCharge.from_trivial(4,BlipSite(True).chinfo)
    leg_i2=BlipSite(True).leg
    leg_p=sites[0].leg
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    Js=[get_proj(Jprim,lc,ln.conj(),leg_i2,leg_i1.conj()).drop_charge() for lc,ln in zip(legs[1:-2],legs[2:-1])]
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+Js+[Id])

def ising_hr_Tp(t,J,g):
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    W_mpo=ising_W(t,g)
    J_mpo_p=ising_hr_J(t,J)
    return multiply_mpos([J_mpo_p,W_mpo])

import tenpy
import numpy as np
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO

from tenpy.networks.site import SpinHalfSite

def get_W_mpo(sites,g):
    T=len(sites)//2
    leg_m=tenpy.linalg.charges.LegCharge.from_trivial(2)
    leg_p=sites[0].leg
    s=1.0j*np.sin(g)
    c=np.cos(g)
    Wprim2=np.array([[c,s],[s,c]])
    Wprim1=np.array([[c,-s],[-s,c]])
    W_1=npc.Array.from_ndarray(np.einsum("cd,cb,ac->abcd",np.eye(2),np.eye(2),Wprim1),[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    W_2=npc.Array.from_ndarray(np.einsum("cd,cb,ac->abcd",np.eye(2),np.eye(2),Wprim2),[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO(sites,[W_2]+[W_1]*T+[W_2]*(T-1),bc="infinite")

def get_h_mpo(sites,h):
    T=len(sites)//2
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    hprim1=np.array([[np.exp(1.0j*h),0],[0,np.exp(-1.0j*h)]])
    hprim2=np.array([[np.exp(-1.0j*h),0],[0,np.exp(1.0j*h)]])
    h_1=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),hprim1),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    h_2=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),hprim2),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    Id=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),np.eye(2)),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO(sites,[Id]+[h_1]*(T-1)+[Id]+[h_2]*(T-1),bc="infinite")

def get_J_mpo(sites,J):
    T=len(sites)//2
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Jprim2=np.array([[np.exp(1.0j*J),np.exp(-1.0j*J)],[np.exp(-1.0j*J),np.exp(1.0j*J)]])
    Jprim1=np.array([[np.exp(-1.0j*J),np.exp(1.0j*J)],[np.exp(1.0j*J),np.exp(-1.0j*J)]])
    Idprim=np.array([[1,1],[1,1]])/np.sqrt(2)
    J_1=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),Jprim1),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    J_2=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),Jprim2),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    Id=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),Idprim),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO(sites,[Id]+[J_1]*(T-1)+[Id]+[J_2]*(T-1),bc="infinite")

def get_proj(op,left,right,p,ps):
    preop=np.einsum("ab,cd->abcd",np.ones((left.ind_len,right.ind_len)),op)
    return npc.Array.from_ndarray(preop,[left,right,p,ps],labels=["wL","wR","p","p*"],dtype=complex,qtotal=[0])
def get_J_mpo_proj(sites,J):
    T=len(sites)//2
    leg_t=LegCharge.from_trivial(1)
    tar=list(range(0,T))+list(range(0,T))[::-1]
    legs=[LegCharge.from_qflat(chinfo,list(range(-i,i+1))) for i in tarr]
    Jprim2=np.array([[np.exp(1.0j*J),np.exp(-1.0j*J)],[np.exp(-1.0j),np.exp(1.0j*J)]])
    Jprim1=np.array([[np.exp(-1.0j*J),np.exp(1.0j*J)],[np.exp(1.0j),np.exp(-1.0j*J)]])
    Idprim=np.array([[1,1],[1,1]])/np.sqrt(2)
    oprim=[Idprim]+[Jprim1]*(T-1)+[Idprim]+[Jprim2]*(T-1)
    lo=[leg_p]+
    ops=[get_proj(op,lc,ln.conj(),leg_i2,lo.conj()).drop_charge() for lc,ln,op,lo in zip(legs[1:-2],legs[2:-1],oprim,legso)]
    return MPO(sites,ops,bc="infinite")

def mps_to_full(mps):
    psi = mps.get_theta(0, mps.L)
    psi = npc.trace(psi,'vL', 'vR')
    psi = psi.to_ndarray()
    return psi.ravel()*mps.norm

def get_free_mps(sites):
    state = [[1,1]] * len(sites) #free state
    psi = MPS.from_product_state(sites, state,bc="infinite")
    return psi
def test_sample():
    import transfer_dual_keldysh as tk
    T=6
    sites=[SpinHalfSite(None) for _ in range(2*T)]
    J,g,h=0.9,0.5,0.3
    h_mpo=get_h_mpo(sites,h)
    W_mpo=get_W_mpo(sites,g)
    J_mpo=get_J_mpo(sites,J)
    mps=get_free_mps(sites)
    ful=mps_to_full(mps)
    for _ in range(4):
        h_mpo.apply_naively(mps)
        W_mpo.apply_naively(mps)
        J_mpo.apply_naively(mps)
    op=tk.get_F_dual_op(T,J,g,h,tk.trivial_sector(2*T))
    op=op.H
    rful=op@op@op@ful
    rmps=mps_to_full(mps)
    assert np.isclose(rful,rmps).all()

def test_proj():
    import transfer_dual_keldysh as tk
    T=6
    sites=[SpinHalfSite(None) for _ in range(2*T)]
    J,g,h=0.9,0.5,0.3
    W_mpo=get_W_mpo(sites,g)
    J_mpo=get_J_mpo_proj(sites,J)
    mps=get_free_mps(sites)
    ful=mps_to_full(mps)
    for _ in range(4):
        h_mpo.apply_naively(mps)
        W_mpo.apply_naively(mps)
        J_mpo.apply_naively(mps)
    op=tk.get_F_dual_op(T,J,g,h,tk.trivial_sector(2*T))
    op=op.H
    rful=op@op@op@ful
    rmps=mps_to_full(mps)
    assert np.isclose(rful,rmps).all()

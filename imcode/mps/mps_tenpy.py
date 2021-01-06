import numpy as np
import tenpy
import os
import h5py
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO



def calc_mps(doc):
    J=doc["J"]
    h=doc["h"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    sites=[BlipSite() for _ in range(T+1)]
    mps=get_it_mps(sites)
    W_mpo=get_W_mpo(sites,g)
    J_mpo=get_J_mpo(sites,J)
    h_mpo=get_h_mpo(sites,h)
    for _ in range(T+1):
        apply_all(mps,h_mpo,W_mpo,J_mpo,chi_max=chi)
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f,mps,"/mps")
        f["bond_entropy"]=mps.entanglement_entropy()
        f["blip_distance_1"]=get_blip_dist1(mps)
        f["blip_distance_2"]=get_blip_dist2(mps)
        f["lohschmidt"]=get_lohschmidt(mps)
        f["czz"]=get_czz(mps,W_mpo,h_mpo)
def get_czz_oneside(mps):
    lstate= [[1,-1,0,0]]+[[1,1,1,1]]*(mps.L-2)+[[1,-1,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)/2
def get_blip_average_uu(mps):
    lstate= [[1,0,0,0]]+[[0,0,1,1]]*(mps.L-2)+[[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)

def get_blip_average_int(mps,g,h):
    lstate= [[1,0,0,0]]+[[0,0,1,1]]*(mps.L-2)+[[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    if mps.L==3:
        hmpo=get_h_mpo(mps.sites,h)
        mpc=mps.copy()
        hmpo.apply_naively(mpc)
        return mpc.overlap(lmps)
    def get_W_mpo_mod(sites,g):
        T=len(sites)-1
        leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
        leg_p=sites[0].leg
        leg_m=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0]*4)
        s2=-np.sin(g)**2
        c2=np.cos(g)**2
        mx=0
        px=0
        Wprim=np.array([[mx,mx,mx,mx],
                        [mx,mx,mx,mx],
                        [mx,mx,c2,s2],
                        [mx,mx,s2,c2]
        ])
        W_bda=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),np.eye(1))
        W_0a=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),np.eye(1))
        W_ia=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(4),Wprim)
        W_Ta=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),Wprim)
        W_bd=npc.Array.from_ndarray(W_bda,[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
        W_0=npc.Array.from_ndarray(W_0a,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
        W_i=npc.Array.from_ndarray(W_ia,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
        W_T=npc.Array.from_ndarray(W_Ta,[leg_m,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
        return MPO(sites,[W_bd,W_0]+[W_i]*(T-3)+[W_T,W_bd])
    W_mod=get_W_mpo_mod(mps.sites,g)
    hmpo=get_h_mpo(mps.sites,h)
    mpc=mps.copy()
    hmpo.apply_naively(mpc)
    W_mod.apply_naively(mpc)
    mpc.canonical_form(False)
    return mpc.overlap(lmps)

def get_blip_average_ud(mps):
    lstate= [[1,0,0,0]]+[[0,0,1,1]]*(mps.L-2)+[[0,1,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)
def calc_slowspin_v2(doc):
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    h=doc["h"]
    T=doc["T"]
    sites=[BlipSite() for _ in range(T+1)]
    gslow=doc["g_s"]
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"a") as f:
        bs=[]
        for gs in gslow:
            bs.append(get_blip_average_int(mps,gs,h))
        f["blip_average_intv2"]=np.array(bs)
def calc_slowspin(doc):
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    g=doc["g"]
    h=doc["h"]
    T=doc["T"]
    sites=[BlipSite() for _ in range(T+1)]
    W_mpo=get_W_mpo(sites,g)
    h_mpo=get_h_mpo(sites,h)
    gslow=doc["g_s"]
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"a") as f:
        f["czz"]=get_czz(mps,W_mpo,h_mpo)
        f["blip_distance_2"]=get_blip_dist2(mps)
        h_mpo.apply_naively(mps)
        czzs,bsuu,bsud=[],[],[]
        for gs in gslow:
            mpc=mps.copy()
            Wslow_mpo=get_W_mpo(sites,gs)
            Wslow_mpo.apply_naively(mpc)
            mpc.canonical_form(False)
            czzs.append(get_czz_oneside(mpc))
            bsuu.append(get_blip_average_uu(mpc))
            bsud.append(get_blip_average_ud(mpc))
        f["czz_slow"]=np.array(czzs)
        f["blip_slow_uu"]=np.array(bsuu)
        f["blip_slow_ud"]=np.array(bsud)
def calc_finite(doc):
    J=doc["J"]
    h=doc["h"]
    g=doc["g"]
    T=doc["T"]
    L=doc["L"]
    chi=doc["chi"]
    sites=[BlipSite() for _ in range(T+1)]
    mps=get_open_mps(sites)
    W_mpo=get_W_mpo(sites,g)
    J_mpo=get_J_mpo(sites,J)
    h_mpo=get_h_mpo(sites,h)
    for _ in range(L):
        apply_all(mps,h_mpo,W_mpo,J_mpo,chi_max=chi)
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f,mps,"/mps")
        f["bond_entropy"]=mps.entanglement_entropy()
        f["blip_distance_1"]=get_blip_dist1(mps)
        f["blip_distance_2"]=get_blip_dist2(mps)
        f["lohschmidt"]=get_lohschmidt(mps)
        f["czz"]=get_czz(mps,W_mpo,h_mpo)

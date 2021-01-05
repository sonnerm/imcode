import numpy as np
import tenpy
import os
import h5py
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
chinfo=tenpy.linalg.charges.ChargeInfo([1],["blip"])
class BlipSite(tenpy.networks.site.Site):
    def __init__(self,conserve=False):
        if conserve:
            leg=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0,0,1,-1])
        else:
            leg=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0,0,0,0])
        self.conserve=conserve
        super().__init__(leg,["+","-","b","a"],)
def get_W_mpo(sites,g):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
    leg_p=sites[0].leg
    leg_m=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0]*4)
    s2=np.sin(g)**2
    c2=np.cos(g)**2
    mx=1.0j*np.sin(g)*np.cos(g)
    px=-1.0j*np.sin(g)*np.cos(g)
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
    return MPO(sites,[W_0]+[W_i]*(T-1)+[W_T])

def get_h_mpo(sites,h):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,1.0,np.exp(2.0j*h),np.exp(-2.0j*h)]))
    H=npc.Array.from_ndarray(Ha,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[H]*(T-1)+[Id])
def get_zz_mpo(sites):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Za=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,-1.0,0.0,0.0]))
    Z=npc.Array.from_ndarray(Za,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Z]+[Id]*(T-1)+[Z])

def get_J_mpo(sites,J):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
    leg_p=sites[0].leg
    if sites[0].conserve:
        raise NotImplementedError()
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
    Ja=np.einsum("ab,cd->abcd",np.eye(1),Jprim)
    J=npc.Array.from_ndarray(Ja,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[J]*(T-1)+[Id])
def get_it_mps(sites):
    state = [[1,1,0,0]] * len(sites) #infinite temperature state
    psi = MPS.from_product_state(sites, state)
    return psi

def get_loc_mps(sites):
    state = [[1/2,1/2,1/2,1/2]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi
def get_open_mps(sites):
    state = [[1,1,1,1]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi


def get_lohschmidt(mps):
    lstate= [[1,0,0,0]]+[[0,0,1,0]]*(mps.L-2)+[[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)
def get_blip_dist1(mps):
    if mps.L<4:
        return 0.0
    lstate= [[1,0,0,0],[0,0,1,0]]+[[1,1,0,0]]*(mps.L-4)+[[0,0,0,1],[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)
def get_blip_dist2(mps):
    if mps.L<6:
        return 0.0
    lstate= [[1,0,0,0],[0,0,1,0],[0,0,0,1]]+[[1,1,0,0]]*(mps.L-6)+[[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return mps.overlap(lmps)
def get_czz(mps,W_mpo,h_mpo):
    mpc=mps.copy()
    W_mpo.apply_naively(mpc)
    h_mpo.apply_naively(mpc)
    leg_p=mps.sites[0].leg
    zop=npc.Array.from_ndarray(np.diag([1,-1,1,1]),[leg_p,leg_p.conj()],labels=["p","p*"])
    mpc.apply_local_op(0,zop,True)
    mpc.apply_local_op(mps.L-1,zop,True)
    return mps.overlap(mpc)/2

def apply_all(mps,h_mpo,W_mpo,J_mpo,chi_max=128):
    options={"trunc_params":{"chi_max":chi_max},"verbose":False,"compression_method":"SVD"}
    h_mpo.apply_naively(mps)
    W_mpo.apply_naively(mps)
    J_mpo.apply_naively(mps)
    mps.compress(options)

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

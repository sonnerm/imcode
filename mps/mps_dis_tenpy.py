import tenpy
from functools import lru_cache
import h5py
import os
import numpy as np
import numpy.linalg as la
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.mpo import MPOEnvironment
class BlipSite(tenpy.networks.site.Site):
    def __init__(self,conserve=False):
        if conserve:
            chinfo=tenpy.linalg.charges.ChargeInfo([1],["blip"])
            self.chinfo=chinfo
            leg=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0,0,1,-1])
        else:
            chinfo=tenpy.linalg.charges.ChargeInfo()
            self.chinfo=chinfo
            leg=tenpy.linalg.charges.LegCharge.from_trivial(4)
        self.conserve=conserve
        super().__init__(leg,["+","-","b","a"],)
def embed_mps(mps):
    nsites=[BlipSite(False) for _ in range(mps.L)]
    return MPS(nsites,[mps.get_B(i).drop_charge() for i in range(mps.L)],[mps.get_SL(0)]+[mps.get_SR(i) for i in range(mps.L)],mps.bc,mps.form,mps.norm)

def get_W_mpo(sites,g):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    leg_m=tenpy.linalg.charges.LegCharge.from_trivial(4,sites[0].chinfo)
    leg_p=sites[0].leg
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
# datastore.unpickle_numpy(datastore.cl.mps_hr_sum.find_one({"T":34,"g":{"$gt":0.16,"$lt":0.25},"chi":128}))["norm_sojourn"]["ex1"]/2**35
def get_h_mpo(sites,h):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,1.0,np.exp(2.0j*h),np.exp(-2.0j*h)]))
    H=npc.Array.from_ndarray(Ha,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[H]*(T-1)+[Id])

def get_zz_mpo(sites):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Za=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,-1.0,0.0,0.0]))
    Z=npc.Array.from_ndarray(Za,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Z]+[Id]*(T-1)+[Z])

def get_J_mpo(sites,J):
    T=len(sites)-1
    if sites[0].conserve: #violates conservation
        raise ValueError()
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    leg_p=sites[0].leg
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



def get_proj(op,left,right,p,ps):
    preop=np.einsum("ab,cd->abcd",np.ones((left.ind_len,right.ind_len)),op)
    return npc.Array.from_ndarray(preop,[left,right,p,ps],labels=["wL","wR","p","p*"],dtype=complex,qtotal=[0],raise_wrong_sector=False)
def get_J_mpo_proj(sites,J):
    if sites[0].conserve: #violates conservation
        raise ValueError()
    T=len(sites)-1
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
def get_J_mpo_cons(sites,J):
    T=len(sites)-1
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
    legs=[tenpy.linalg.charges.LegCharge.from_qflat(BlipSite(True).chinfo,list(range(-2*i,2*i+1))) for i in tarr]
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1,sites[0].chinfo)
    leg_p=sites[0].leg
    leg_i=BlipSite(True).leg
    if sites[0].conserve:
        Js=[get_proj(Jprim,lc,ln.conj(),leg_i,leg_i.conj()) for lc,ln in zip(legs[1:-2],legs[2:-1])]
    else:
        Js=[get_proj(Jprim,lc,ln.conj(),leg_i,leg_i.conj()).drop_charge() for lc,ln in zip(legs[1:-2],legs[2:-1])]
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+Js+[Id])

@lru_cache(None)
def get_Jr_mpo(L):
    sites=[BlipSite(False) for _ in range(L)]
    if sites[0].conserve: #violates conservation
        raise ValueError()
    T=len(sites)-1
    tarr=[0]+list(range(T//2+T%2))+list(range(T//2))[::-1]+[0]
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    legs=[LegCharge.from_qflat(BlipSite(True).chinfo,list(range(-i,i+1))) for i in tarr]
    leg_i1=LegCharge.from_trivial(4,BlipSite(True).chinfo)
    leg_i2=BlipSite(True).leg
    leg_p=sites[0].leg
    leg_t=LegCharge.from_trivial(1,sites[0].chinfo)
    Js=[get_proj(Iprim,lc,ln.conj(),leg_i2,leg_i1.conj()).drop_charge() for lc,ln in zip(legs[1:-2],legs[2:-1])]
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+Js+[Id])

def get_it_mps(sites):
    state = [[1,1,0,0]] * len(sites) #infinite temperature state
    psi = MPS.from_product_state(sites, state)
    return psi

def get_loc_mps(sites):
    state = [[1/2,1/2,1/2,1/2]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi

def get_czz(mps,W_mpo,h_mpo):
    mpc=mps.copy()
    W_mpo.apply_naively(mpc)
    h_mpo.apply_naively(mpc)
    leg_p=mps.sites[0].leg
    zop=npc.Array.from_ndarray(np.diag([1,-1,1,1]),[leg_p,leg_p.conj()],labels=["p","p*"])
    mpc.apply_local_op(0,zop,True)
    mpc.apply_local_op(mps.L-1,zop,True)
    return mps.overlap(mpc)/2

def get_norm(mps,W_mpo,h_mpo):
    mpc=mps.copy()
    W_mpo.apply_naively(mpc)
    h_mpo.apply_naively(mpc)
    leg_p=mps.sites[0].leg
    return mps.overlap(mpc)/2
def get_czz_norm(mps,W_mpo,h_mpo):
    mpc=mps.copy()
    W_mpo.apply_naively(mpc)
    h_mpo.IdL[0]=0
    h_mpo.IdR[mps.L]=0
    norm=MPOEnvironment(mps,h_mpo,mpc).full_contraction(0)*mps.norm*mpc.norm
    leg_p=mps.sites[0].leg
    zop=npc.Array.from_ndarray(np.diag([1,-1,1,1]),[leg_p,leg_p.conj()],labels=["p","p*"])
    mpc.apply_local_op(0,zop,True)
    mpc.apply_local_op(mps.L-1,zop,True)
    czz=MPOEnvironment(mps,h_mpo,mpc).full_contraction(0)*mps.norm*mpc.norm
    return czz,norm
# import datastore
# # !!
# doc=datastore.cl.mps_unit.find_one({"T":30,"g":{"$gt":0.3},"J":{"$gt":0.3},"h":{"$exists":0},"chi":32})
# # !scp baobab:scratch/mps_hr_lowchi/out/{str(doc["_id"])}.h5 in
# import h5py
# f=h5py.File("in/%s.h5"%(str(doc["_id"])))
# mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
# # calc_cz(doc)
#
# J=doc["J"]
# g=doc["g"]
# T=doc["T"]
# chi=doc["chi"]
# sites=[BlipSite(False) for _ in range(T+1)]
# h_mpo=get_hr_mpo(mps.L)
# W_mpo=get_W_mpo(sites,g)
# mps.L
# h_mpo.L
# W_mpo.L
# czz1=get_czz(mps,W_mpo,h_mpo)
# czz1
# norm1
# norm1=get_norm(mps,W_mpo,h_mpo)
# czz2,norm2=get_czz_norm(mps,W_mpo,h_mpo)
#
# czz2/norm2
# norm2
# norm1

# alc_cz(doc)



def apply_all(mps,W_mpo,J_mpo,chi_max=128):
    options={"trunc_params":{"chi_max":chi_max},"m_temp":4,"verbose":False,"compression_method":"zip_up"}
    W_mpo.apply(mps,options)
    J_mpo.apply(mps,options)
def calc_mps(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    sites=[BlipSite(False) for _ in range(T+1)]
    mps=get_it_mps(sites)
    W_mpo=get_W_mpo(sites,g)
    J_mpo_p=get_J_mpo_proj(sites,J)
    J_mpo=get_J_mpo(sites,J)
    for _ in range(T):
        apply_all(mps,W_mpo,J_mpo_p,chi_max=chi)
    apply_all(mps,W_mpo,J_mpo,chi_max=chi)
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f,mps,"/mps")
        f["bond_entropy"]=mps.entanglement_entropy()
        # f["czz"]=get_czz(mps,W_mpo) #needs to be projected, my bad
def magsec_proj(T,M,branch="fw"):
    chinfo=ChargeInfo([1],[branch])
    if branch=="fw":
        leg_p=LegCharge.from_qflat(chinfo,[1,0,1,0])
    else:
        leg_p=LegCharge.from_qflat(chinfo,[1,0,0,1])
    leg_t=LegCharge.from_trivial(1,chinfo)
    leg_pt=LegCharge.from_trivial(4,chinfo)
    leg_rt=LegCharge.from_qflat(chinfo,[M])
    legr=[(max(0,i-T+M+2),min(i+1,M)+1) for i in range(T-2)]
    legs=[leg_t]+[LegCharge.from_qflat(chinfo,list(range(*x))) for x in legr]+[leg_rt]
    Id_a=np.eye(4)
    Id_m=[get_proj(Id_a,ll,lr.conj(),leg_pt,leg_p).drop_charge() for ll,lr in zip(legs[:-1],legs[1:])]
    Id_l=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),np.eye(4)),[leg_t,leg_t.conj(),leg_pt,leg_pt.conj()],labels=["wL","wR","p","p*"],dtype=complex)
    Id_r=npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),np.eye(4)),[leg_rt,leg_t.conj(),leg_pt,leg_pt.conj()],labels=["wL","wR","p","p*"],dtype=complex,qtotal=[M])
    return MPO([BlipSite(False) for t in range(T+1)],[Id_l.drop_charge()]+Id_m+[Id_r.drop_charge()])

@lru_cache(None)
def get_hr_mpo(L):
    sites=[BlipSite(False) for _ in range(L)]
    if sites[0].conserve: #violates conservation
        raise ValueError()
    T=len(sites)-1
    tarr=[0]+list(range(T//2+T%2))+list(range(T//2))[::-1]+[0]
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
def get_magsec(mps,W_mpo,options):
    T=mps.L-1
    msuu=[]
    msud=[]
    pu_a=np.zeros((4,4))
    pu_a[0,0]=1
    pd_a=np.zeros((4,4))
    pd_a[1,1]=1
    leg_p=LegCharge.from_trivial(4)
    pu=npc.Array.from_ndarray(pu_a,[leg_p,leg_p.conj()],labels=["p","p*"],dtype=complex)
    pd=npc.Array.from_ndarray(pd_a,[leg_p,leg_p.conj()],labels=["p","p*"],dtype=complex)
    mpc=mps.copy()
    get_hr_mpo(mps.L).apply(mpc,options)
    mpcuu=mpc.copy()
    W_mpo.apply_naively(mpcuu)
    mpcuu.apply_local_op(0,pu)
    mpcud=mpcuu.copy()

    mpcuu.apply_local_op(T,pu)
    mpcuu.canonical_form(False)

    mpcud.apply_local_op(T,pd)
    mpcud.canonical_form(False)
    normuu=mpcuu.norm*mpc.norm
    normud=mpcud.norm*mpc.norm
    # print(normuu)
    # print(normud)
    for i in range(T):
        msp=magsec_proj(T,i,"fw")
        msp.IdL[0]=0
        msp.IdR[T+1]=0
        msuu.append(MPOEnvironment(mpc,msp,mpcuu).full_contraction(0)*normuu)
        msud.append(MPOEnvironment(mpc,msp,mpcud).full_contraction(0)*normud)
    return msuu,msud#,normuu,normud,normuue,normude
def pattern_to_mps(pattern):
    pdict={"+":[1,0,0,0],"-":[0,1,0,0],"b":[0,0,1,0],"a":[0,0,0,1],"q":[0,0,1,1],"c":[1,1,0,0],"*":[1,1,1,1]}
    state = [pdict[p] for p in pattern]
    sites=[BlipSite(False) for _ in pattern]
    psi = MPS.from_product_state(sites, state)
    return psi
def get_entropy(M):
    ev=la.eigvals(M)
    return -sum(ev*np.log(ev+1e-20))
# import h5py
#
# f=h5py.File("../out/test_5.h5")
# mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
# mps.L
# mutual_information(mps)
# mps
# rho=mps.get_rho_segment([1])
# rs=rho.to_ndarray()[[0,2,3,1],:][:,[0,2,3,1]]
# calc_mps({"_id":"test2","J":0.2,"T":10,"g":0.2,"chi":64})
def mutual_information(mps):
    minf=[]
    for i in range(mps.L):
        rho=mps.get_rho_segment([i]).to_ndarray()[[0,2,3,1],:][:,[0,2,3,1]]
        ce=get_entropy(rho)
        cfw=get_entropy(np.trace(rho.reshape(2,2,2,2),axis1=0,axis2=2))
        cbw=get_entropy(np.trace(rho.reshape(2,2,2,2),axis1=1,axis2=3))
        minf.append(cfw+cbw-ce)
    return minf
def calc_cz(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    sites=[BlipSite(False) for _ in range(T+1)]
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    W_mpo=get_W_mpo(sites,g)
    nop="c"*(T+1)
    czp1="+"+"c"*(T-1)+"+"
    czp2="+"+"c"*(T-1)+"-"
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        f["czz_middle"],f["norm_middle"]=get_czz_norm(mps,W_mpo,get_hr_mpo(mps.L))
        # f["czz_boundary"]=get
        f["norm_sojourn"]=mps.overlap(pattern_to_mps(nop))
def calc_zs(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    options={"trunc_params":{"chi_max":chi},"m_temp":4,"verbose":False,"compression_method":"zip_up"}
    sites=[BlipSite(False) for _ in range(T+1)]
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    W_mpo=get_W_mpo(sites,g)
    msuu,msud=get_magsec(mps,W_mpo,options)
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        f["magsec_uu"],f["magsec_ud"]=msuu,msud
        f["czz"]=get_czz(mps,W_mpo,get_hr_mpo(mps.L))

def calc_overlaps(doc):
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    p_set=doc["overlap_pattern_set"]
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        for ps in p_set:
            ov=[]
            for p in doc[ps]:
                ov.append(mps.overlap(pattern_to_mps(p)))
            f[ps]=ov
        f["mutual_information"]=mutual_information(mps)
def test_method():
    import datastore

    doc=datastore.cl.mps_unit.find_one({"campaign_all.mps_hr_diag_128":"D","g":{"$gt":0.2},"T":34,"chi":128})
    check_convergence(doc)
def check_convergence(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    sites=[BlipSite(False) for _ in range(T+1)]
    W_mpo=get_W_mpo(sites,g)
    J_mpo_p=get_J_mpo_proj(sites,J)
    J_mpo=get_J_mpo(sites,J)
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    apply_all(mps,W_mpo,J_mpo_p,chi_max=chi)
    mpc=mps.copy()
    apply_all(mpc,W_mpo,J_mpo_p,chi_max=chi)
    # return mps.overlap(mpc)/mps.norm/mpc.norm
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        f["convergence_overlap"]=mps.overlap(mpc)

import mps_dis_tenpy as mdt
import numpy as np
import os
import tenpy
from tenpy.networks.mpo import MPO,MPOEnvironment
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
import h5py
import functools
from tenpy.algorithms.dmrg import DMRGEngine,TwoSiteDMRGEngine
import tenpy.linalg.np_conserved as npc

class Model():
    def __init__(self, mpo):
        self.H_MPO=mpo
def check_iter_var():
    J=0.2
    g=0.2
    T=30
    mpo=get_full_mpo_hr(J,g,T)
    psi=mdt.get_it_mps(mpo.sites)
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    mpo.IdL[0]=0
    mpo.IdR[-1]=0
    options={
        "diag_method":"lanczos_nonhermitian",
        "mixer":None,
        "orthogonal_to":[],
        "chi_max":64,
        "max_E_err":1.0,# Not useful criterion
    }
    res=TwoSiteDMRGEngine(psi,Model(mpo),options)
    res=res.run()
    mps=mdt.get_it_mps(sites)
    W_mpo=mdt.get_W_mpo(sites,g)
    J_mpo_p=mdt.get_J_mpo_proj(sites,J)
    J_mpo=mdt.get_J_mpo(sites,J)
    chi=100
    for _ in range(T):
        mdt.apply_all(mps,W_mpo,J_mpo_p,chi_max=chi)
    mpc=mps.copy()
    mdt.apply_all(mpc,W_mpo,J_mpo_p,chi_max=chi)
    mpc.overlap(mps)/mpc.norm/mps.norm
    mpr=res[1].copy()
    mdt.apply_all(mpr,W_mpo,J_mpo_p,chi_max=chi)
    mpr.overlap(res[1])/mpr.norm/res[1].norm
    mps.overlap(res[1])/mps.norm/res[1].norm
    mps.overlap(res[1])/mps.norm



def multiply_W(w1,w2):
    pre=npc.tensordot(w1,w2,axes=[("p*",),("p",)])
    pre=pre.combine_legs([(0,3),(1,4)])
    pre.ireplace_labels(["(?0.?3)","(?1.?4)"],["wL","wR"])
    return pre

def multiply_mpos(mpolist):
    Wps=[[m.get_W(i) for m in mpolist] for i in range(mpolist[0].L)]
    return MPO(mpolist[0].sites,[functools.reduce(multiply_W,Wp) for Wp in Wps])

def get_full_mpo_hr(J,g,T):
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    W_mpo=mdt.get_W_mpo(sites,g)
    J_mpo_p=mdt.get_J_mpo_proj(sites,J)
    return multiply_mpos([J_mpo_p,W_mpo])
# import datastore
# doc=datastore.cl.mps_unit.find_one({"T":20,"chi":64,"g":{"$gt":0.2},"J":{"$gt":0.2},"campaign_all.mps_hr_lowchi":"D"})
# !cp baobab_scratch/mps_hr_lowchi/out/{str(doc["_id"])}.h5 in

# calc_dmrg_from_iter(doc)

def get_dmrg_czz_norm(mps,W_mpo):
    mpc=mps.copy()
    W_mpo.IdL[0]=0
    W_mpo.IdR[-1]=0
    norm=MPOEnvironment(mps,W_mpo,mpc).full_contraction(0)*mps.norm*mpc.norm
    leg_p=mps.sites[0].leg
    zop=npc.Array.from_ndarray(np.diag([1,-1,1,1]),[leg_p,leg_p.conj()],labels=["p","p*"])
    mpc.apply_local_op(0,zop,True)
    mpc.apply_local_op(mps.L-1,zop,True)
    czz=MPOEnvironment(mps,W_mpo,mpc).full_contraction(0)*mps.norm*mpc.norm
    return czz,norm
def calc_dmrg_from_iter(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    W_mpo=mdt.get_W_mpo(sites,g)
    J_mpo_p=mdt.get_J_mpo_proj(sites,J)
    hr_mpo=mdt.get_hr_mpo(T+1)
    options={
        "diag_method":"lanczos_nonhermitian",
        "mixer":None,
        "orthogonal_to":[],
        "chi_list":{0:chi}
    }
    mpo=multiply_mpos([J_mpo_p,W_mpo])
    mpo.IdL[0]=0
    mpo.IdR[-1]=0
    hroptions={"trunc_params":{"chi_max":chi},"m_temp":4,"verbose":False,"compression_method":"zip_up"}
    hr_mpo.apply(mps,hroptions)
    res=TwoSiteDMRGEngine(mps.copy(),Model(mpo),options).run()
    # res=(1,mps.copy())
    nop="c"*(T+1)
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f,res[1],"/mps")
        f["dmrg_bond_entropy"]=res[1].entanglement_entropy()
        f["dmrg_ev"]=res[0]
        f["dmrg_overlap"]=mps.overlap(res[1])/mps.norm/res[1].norm
        f["dmrg_czz_middle"],f["dmrg_norm_middle"]=get_dmrg_czz_norm(res[1],W_mpo)
        f["dmrg_norm_sojourn"]=res[1].overlap(mdt.pattern_to_mps(nop))
def calc_correct_be(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    W_mpo=mdt.get_W_mpo(sites,g)
    J_mpo=mdt.get_J_mpo(sites,J)
    W_mpo.apply_naively(mps)
    J_mpo.apply_naively(mps)
    mps.canonical_form()
    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        f["dmrg_full_bond_entropy"]=mps.entanglement_entropy()
def timestate_op(T,M,g,bd):
    sites=[mdt.BlipSite(False) for _ in range(T+1)]
    wop=mdt.get_W_mpo(sites,g)
    msop=mdt.magsec_proj(T,M)
    leg_t=LegCharge.from_trivial(1,sites[0].chinfo)
    leg_p=LegCharge.from_trivial(4,sites[0].chinfo)
    if bd:
        UDDICT={"u":np.diag([1,0,0,0]),"d":np.diag([0,1,0,0])}
        pops=[UDDICT[bd[0]]]+[np.eye(4) for _ in range(T-1)]+[UDDICT[bd[1]]]
        pops=[npc.Array.from_ndarray(np.einsum("ab,cd->abcd",np.eye(1),M),[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for M in pops]
        pop=MPO(sites,pops)
        rop=multiply_mpos([wop,msop,pop])
    else:
        rop=multiply_mpos([wop,msop])
    rop.IdL[0]=0
    rop.IdR[-1]=0
    return rop
def calc_magsec(doc):
    J=doc["J"]
    g=doc["g"]
    T=doc["T"]
    chi=doc["chi"]
    with h5py.File(os.path.join("in","%s.h5"%(str(doc["_id"]))),"r") as f:
        mps=tenpy.tools.hdf5_io.load_from_hdf5(f,"/mps")
    msuu,msud=[],[]
    for M in range(T):
        opuu=timestate_op(T,M,g,"uu")
        opud=timestate_op(T,M,g,"ud")
        msuu.append(opuu.expectation_value(mps)*mps.norm*mps.norm)
        msud.append(opud.expectation_value(mps)*mps.norm*mps.norm)

    with h5py.File(os.path.join("out","%s.h5"%(str(doc["_id"]))),"w") as f:
        f["dmrg_magsec_uu"]=msuu
        f["dmrg_magsec_ud"]=msud

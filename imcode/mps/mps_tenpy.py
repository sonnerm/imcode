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

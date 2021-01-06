
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

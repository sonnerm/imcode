import tenpy
def folded_temporal_entropy(mps):
    return mps.bond_entropy()

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

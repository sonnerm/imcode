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

def get_zz_mpo(sites):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_qflat(chinfo,[0])
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Za=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,-1.0,0.0,0.0]))
    Z=npc.Array.from_ndarray(Za,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Z]+[Id]*(T-1)+[Z])

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


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

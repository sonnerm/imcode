import tenpy
import numpy as np
from .utils import apply,multiply_mpos
from .fold.utils import FoldSite
from .flat.utils import FlatSite
from tenpy.networks.mpo import MPOEnvironment,MPO
from tenpy.networks.mps import MPS
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc
def boundary_obs(im,obs):
    return im.overlap(obs)/2
def embedded_obs(left_im,obs_mpo,right_im):
    obs_mpo.IdL[0]=0
    obs_mpo.IdR[-1]=0
    return MPOEnvironment(left_im,obs_mpo,right_im).full_contraction(0)*left_im.norm*right_im.norm/2
def zz_operator(t):
    sites=[FoldSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Za=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,-1.0,0.0,0.0]))
    Z=npc.Array.from_ndarray(Za,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Z]+[Id]*(t-1)+[Z])

def zz_state(t):
    sites=[FoldSite() for _ in range(t+1)]
    state=[[1,-1,0,0]]+[[1,1,1,1]]*(t-1)+[[1,-1,0,0]]
    return MPS.from_product_state(sites,state)

def embedded_czz(im,lop):
    t=im.L-1
    op=multiply_mpos([lop,zz_operator(t)])
    return embedded_obs(im,op,im)
def boundary_czz(im,lop):
    t=im.L-1
    st=zz_state(t)
    apply(lop,st)
    return boundary_obs(im,st)
def embedded_norm(im,lop):
    return embedded_obs(im,lop,im)
def boundary_norm(im,lop):
    st=open_boundary_im(t)
    apply(lop,st)
    return boundary_obs(im,lop,im)

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
    return MPO([FoldSite(False) for t in range(T+1)],[Id_l.drop_charge()]+Id_m+[Id_r.drop_charge()])
def embedded_magsec(im,s,lop):
    pass
def boundary_magsec(im,s,lop):
    pass
def lohschmidt(im):
    lstate= [[1,0,0,0]]+[[0,0,1,0]]*(mps.L-2)+[[1,0,0,0]]
    lmps=MPS.from_product_state(mps.sites,lstate)
    return boundary_obs(im,lmps)


# def get_blip_dist1(mps):
#     if mps.L<4:
#         return 0.0
#     lstate= [[1,0,0,0],[0,0,1,0]]+[[1,1,0,0]]*(mps.L-4)+[[0,0,0,1],[1,0,0,0]]
#     lmps=MPS.from_product_state(mps.sites,lstate)
#     return mps.overlap(lmps)
# def get_blip_dist2(mps):
#     if mps.L<6:
#         return 0.0
#     lstate= [[1,0,0,0],[0,0,1,0],[0,0,0,1]]+[[1,1,0,0]]*(mps.L-6)+[[0,0,1,0],[0,0,0,1],[1,0,0,0]]
#     lmps=MPS.from_product_state(mps.sites,lstate)
#     return mps.overlap(lmps)


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
    return MPO([FoldSite(False) for t in range(T+1)],[Id_l.drop_charge()]+Id_m+[Id_r.drop_charge()])

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

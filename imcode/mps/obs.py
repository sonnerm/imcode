import numpy as np
from .utils import apply,multiply_mpos
from .channel import unitary_channel,mpo_to_state
from ..dense import SZ
from . import fold
from . import flat
def zz_operator(t,ti=0,tj=None):
    if tj==None:
        tj=t
    sites=[fold.FoldSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Za=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,-1.0,0.0,0.0]))
    Z=npc.Array.from_ndarray(Za,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]*ti+[Z]+[Id]*(tj-ti-1)+[Z]+[Id]*(t-tj))

def zz_state(t,ti=0,tj=None):
    if tj==None:
        tj=t
    sites=[fold.FoldSite() for _ in range(t+1)]
    state=[[1,-1,0,0]]+[[1,1,1,1]]*(tj-ti-1)+[[1,-1,0,0]]
    if ti!=0:
        state=[[1,1,0,0]]+[[1,1,1,1]]*(ti-1)+state
    if tj!=t:
        state=state+[[1,1,1,1]]*(t-tj-1)+[[1,1,0,0]]
    return MPS.from_product_state(sites,state)

def embedded_czz(im,lop,ti=0,tj=None):
    t=im.L-1
    op=multiply_mpos([lop,zz_operator(t,ti,tj)])
    return embedded_obs(im,op,im)
def boundary_czz(im,lop,ti=0,tj=None):
    t=im.L-1
    st=zz_state(t,ti=0,tj=None)
    apply(lop,st)
    return boundary_obs(im,st)
def embedded_norm(im,lop):
    return embedded_obs(im,lop,im)
def boundary_norm(im,lop):
    t=im.L-1
    st=fold.open_boundary_im(t)
    apply(lop,st)
    return boundary_obs(im,st)


def direct_czz(F,t,i,j,chi=None,options=None,init=None):
    L=F.L
    leg_p=LegCharge.from_trivial(2)
    leg_b=LegCharge.from_trivial(1)
    assert isinstance(F.sites[0],flat.FlatSite)
    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[i]=np.array([[SZ]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    mop=MPO(F.sites,szop)
    if init is not None:
        mop=multiply_mpos([init,mop])
    state=mpo_to_state(mop)
    uchannel=unitary_channel(F)
    for _ in range(t):
        apply(uchannel,state,chi,options)
    szop=[np.array([[np.eye(2)]]) for _ in range(L)]
    szop[j]=np.array([[SZ]])
    szop=[npc.Array.from_ndarray(w,[leg_b,leg_b.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for w in szop]
    ostate=mpo_to_state(MPO(F.sites,szop))
    return state.overlap(ostate)/2**L,state

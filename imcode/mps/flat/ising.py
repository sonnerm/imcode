from .utils import FlatSite
from ..utils import multiply_mpos
def ising_W(t,g):
    sites=[FlatSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(2)
    leg_m=LegCharge.from_trivial(2)
    s=1.0j*np.sin(g)
    c=np.cos(g)
    Wprim=np.array([[c,s],
                    [s,c]])
    W_0a=np.einsum("cd,cb,ac->abcd",np.eye(2),np.eye(2),np.eye(1))
    W_ia=np.einsum("cd,cb,ac->abcd",np.eye(2),np.eye(2),Wprim)
    W_Ta=np.einsum("cd,cb,ac->abcd",np.eye(4),np.eye(1),Wprim)
    W_0=npc.Array.from_ndarray(W_0a,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    W_i=npc.Array.from_ndarray(W_ia,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    W_T=npc.Array.from_ndarray(W_Ta,[leg_m,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[W_0]+[W_i]*(t-1)+[W_T])

def ising_h(t,h):
    sites=[FlatSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(2))
    Hfa=np.einsum("ab,cd->abcd",np.eye(1),np.diag([np.exp(1.0j*h),np.exp(-1.0j*np.conj(h))]))
    Hba=np.einsum("ab,cd->abcd",np.eye(1),np.diag([np.exp(-1.0j*np.conj(h)),np.exp(1.0j*h)]))
    Hf=npc.Array.from_ndarray(Hfw,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Hb=npc.Array.from_ndarray(Hbw,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[Hf]*(t-1)+[Id]+[Hb]*(t-1))

def ising_J(t,J):
    sites=[FlatSite() for _ in range(t+1)]
    leg_t=LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    # Iprim=np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]])/np.sqrt(2)
    Iprim=np.array([[1.0,1.0],[1.0,1.0]])/np.sqrt(2)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    pj=np.exp(1.0j*J)
    mj=np.exp(-1.0j*np.conj(J))
    Jprim=np.array([[pj,mj],
                    [mj,pj]])
    Ja=np.einsum("ab,cd->abcd",np.eye(1),Jprim)
    Jf=npc.Array.from_ndarray(Ja,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Jb=npc.Array.from_ndarray(np.conj(Ja),[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[Jf]*(t-1)+[Id]+[Jb]*(t-1))
def ising_T(t,J,g,h):
    return multiply_mpos([ising_J(t,J),ising_W(t,g),ising_h(t,h)])

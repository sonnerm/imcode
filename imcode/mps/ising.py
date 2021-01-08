def ising_H(J,g,h):
    pass

def ising_F(J,g,h):
    pass
def ising_W(t,g):
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

def ising_h(t,h):
    T=len(sites)-1
    leg_t=tenpy.linalg.charges.LegCharge.from_trivial(1)
    leg_p=sites[0].leg
    Ida=np.einsum("ab,cd->abcd",np.eye(1),np.eye(4))
    Ha=np.einsum("ab,cd->abcd",np.eye(1),np.diag([1.0,1.0,np.exp(2.0j*h),np.exp(-2.0j*h)]))
    H=npc.Array.from_ndarray(Ha,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+[H]*(T-1)+[Id])

def ising_J(t,J):
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
def ising_T(t,J,g,h):
    pass

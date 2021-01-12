from functools import lru_cache
@lru_cache(None)
def Jr_operator(t):
    sites=[BlipSite() for t in range(t+1)]
    # Iprim=np.ones((4,4))/np.sqrt(2)
    Iprim=np.ones((4,4))/np.sqrt(2)
    Iprim=np.array([[1.0,1.0,0.0,0.0],[1.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])/np.sqrt(2)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    inc=[wrap_ndarray(_get_jr_inc(c)) for c in range((t-1)//2)]
    dec=[wrap_ndarray(_get_jr_dec(c)) for c in range((t-1)//2,0,-1)]
    if (t-1)%2:
        return MPO(sites,[Id]+inc+[wrap_ndarray(_get_jr_cen(t//2-1))]+dec+[Id])
    else:
        return MPO(sites,[Id]+inc+dec+[Id])

def _get_jr_cen(c):
    ret=np.zeros((2*c+1,2*c+1,4,4))
    for i in range(2*c+1):
        #No flip
        ret[i,i,0,0]=1.0
        ret[i,i,1,1]=1.0
        ret[i,i,2,2]=1.0
        ret[i,i,3,3]=1.0
        #flip on fw and bw
        ret[i,i,0,1]=1.0
        ret[i,i,1,0]=1.0
        ret[i,i,2,3]=1.0
        ret[i,i,3,2]=1.0
        if i<2*c:
            #flip on fw
            ret[i,i+1,0,3]=1.0
            ret[i,i+1,1,2]=1.0
            ret[i,i+1,2,1]=1.0
            ret[i,i+1,3,0]=1.0
        if i>0:
            #flip on bw
            ret[i,i-1,0,2]=1.0
            ret[i,i-1,1,3]=1.0
            ret[i,i-1,3,1]=1.0
            ret[i,i-1,2,0]=1.0
    return ret
def _get_jr_inc(c):
    ret=np.zeros((2*c+1,2*c+3,4,4))
    for i in range(2*c+1):
        #No flip
        ret[i,i+1,0,0]=1.0
        ret[i,i+1,1,1]=1.0
        ret[i,i+1,2,2]=1.0
        ret[i,i+1,3,3]=1.0
        #flip on fw and bw
        ret[i,i+1,0,1]=1.0
        ret[i,i+1,1,0]=1.0
        ret[i,i+1,2,3]=1.0
        ret[i,i+1,3,2]=1.0
        #flip on fw
        ret[i,i+2,0,3]=1.0
        ret[i,i+2,1,2]=1.0
        ret[i,i+2,2,1]=1.0
        ret[i,i+2,3,0]=1.0
        #flip on bw
        ret[i,i,0,2]=1.0
        ret[i,i,1,3]=1.0
        ret[i,i,3,1]=1.0
        ret[i,i,2,0]=1.0
    return ret

def _get_jr_dec(c):
    ret=np.zeros((2*c+1,2*c-1,4,4))
    for i in range(2*c-1):
        ret[i+1,i,0,0]=1.0
        ret[i+1,i,1,1]=1.0
        ret[i+1,i,2,2]=1.0
        ret[i+1,i,3,3]=1.0
        #flip on fw and bw
        ret[i+1,i,0,1]=1.0
        ret[i+1,i,1,0]=1.0
        ret[i+1,i,2,3]=1.0
        ret[i+1,i,3,2]=1.0

        #flip on fw
        ret[i,i,0,3]=1.0
        ret[i,i,1,2]=1.0
        ret[i,i,2,1]=1.0
        ret[i,i,3,0]=1.0
        #flip on bw
        ret[i+2,i,0,2]=1.0
        ret[i+2,i,1,3]=1.0
        ret[i+2,i,3,1]=1.0
        ret[i+2,i,2,0]=1.0
    return ret
def ising_Jr_T(t,g,h):
    return multiply_mpos([Jr_operator(t),ising_W(t,g),ising_h(t,h)])

def ising_Jhr_T(t,g):
    return multiply_mpos([Jr_operator(t),ising_W(t,g),hr_operator(t)])

def ising_Jhr_Tp(t,g):
    return multiply_mpos([hr_operator(t),Jr_operator(t),ising_W(t,g)])

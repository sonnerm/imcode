
@lru_cache(None)
def Jr_op(t):
    sites=[BlipSite() for _ in range(t+1)]
    tarr=[0]+list(range(t//2+t%2))+list(range(t//2))[::-1]+[0]
    Ida=np.einsum("ab,cd->abcd",np.eye(1),Iprim)
    legs=[LegCharge.from_qflat(BlipSite(True).chinfo,list(range(-i,i+1))) for i in tarr]
    leg_i1=LegCharge.from_trivial(4,BlipSite(True).chinfo)
    leg_i2=BlipSite(True).leg
    leg_p=sites[0].leg
    leg_t=LegCharge.from_trivial(1,sites[0].chinfo)
    Js=[get_proj(Iprim,lc,ln.conj(),leg_i2,leg_i1.conj()).drop_charge() for lc,ln in zip(legs[1:-2],legs[2:-1])]
    Id=npc.Array.from_ndarray(Ida,[leg_t,leg_t,leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) # make sure
    return MPO(sites,[Id]+Js+[Id])

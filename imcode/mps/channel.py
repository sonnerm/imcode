import numpy as np
from tenpy.linalg.charges import LegCharge,LegPipe
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site
class OperatorSite(Site):
    def __init__(self,site):
        self.base_site=site
        super().__init__(LegPipe([site.leg,site.leg.conj()]))

def mpo_to_state(mpo):
    nsites=[OperatorSite(s) for s in mpo.sites]
    Ws=[mpo.get_W(i,False) for i in range(mpo.L)]
    Ws=[W.combine_legs(["p","p*"],pipes=nsites[i].leg) for i,W in enumerate(Ws)]
    for W in Ws:
        W.ireplace_labels(["(p.p*)","wL","wR"],["p","vL","vR"])
    Svs=[np.ones(W.shape[0]) / np.sqrt(W.shape[1]) for W in Ws]
    Svs.append([1.0])
    Svs[0]=[1.0]
    ret=MPS(nsites,Ws,Svs,form=None)
    if mpo.L>2:
        ret.canonical_form(False)
    return ret

def state_to_mpo(state):
    nsites=[s.base_site for s in state.sites]
    normp=state.norm**(1/state.L)
    Bs=[state.get_B(i,copy=True,form=None) for i in range(state.L)]
    for B in Bs:
        B.ireplace_labels(["p","vL","vR"],["(p.p*)","wL","wR"])
    Bs=[B*normp for B in Bs]
    Bs=[B.split_legs() for B in Bs]
    ret=MPO(nsites,Bs)
    return ret
def unitary_channel(F):
    nsites=[OperatorSite(s) for s in F.sites]
    Ws=[F.get_W(i,True) for i in range(F.L)]
    Wcs=[F.get_W(i,True) for i in range(F.L)]
    for Wc in Wcs:
        Wc.ireplace_labels(["p*","p","wL","wR"],["p1","p1*","wL1*","wR1*"])
        Wc.conj(inplace=True)
    Ws=[npc.outer(W,Wc) for W,Wc in zip(Ws,Wcs)]
    Ws=[W.combine_legs([("p","p1"),("p*","p1*"),("wL","wL1"),("wR","wR1")]) for W in Ws]
    for W in Ws:
        W.ireplace_labels(["(p.p1)","(p*.p1*)","(wL.wL1)","(wR.wR1)"],["p","p*","wL","wR"])
    ret=MPO(nsites,Ws)
    return ret

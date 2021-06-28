import numpy as np
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
def im_channel(im,i):
    pass

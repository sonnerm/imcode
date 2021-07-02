from tenpy.linalg.charges import LegCharge
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site
class FoldSite(Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(4),["+","-","b","a"],)
def pattern_to_mps(pattern):
    pdict={"+":[1,0,0,0],"-":[0,1,0,0],"b":[0,0,1,0],"a":[0,0,0,1],"q":[0,0,1,1],"c":[1,1,0,0],"*":[1,1,1,1]}
    state = [pdict[p] for p in pattern]
    sites=[BlipSite() for _ in pattern]
    psi = MPS.from_product_state(sites, state)
    return psi
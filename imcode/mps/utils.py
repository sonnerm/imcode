from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import LegCharge
class BlipSite(tenpy.networks.site.Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(4),["+","-","b","a"],)

def mpo_to_dense():
    pass
def mps_to_dense():
    pass

def apply_all(mps,h_mpo,W_mpo,J_mpo,chi_max=128):
    options={"trunc_params":{"chi_max":chi_max},"verbose":False,"compression_method":"SVD"}
    h_mpo.apply_naively(mps)
    W_mpo.apply_naively(mps)
    J_mpo.apply_naively(mps)
    mps.compress(options)
def get_it_mps(sites):
    state = [[1,1,0,0]] * len(sites) #infinite temperature state
    psi = MPS.from_product_state(sites, state)
    return psi

def get_loc_mps(sites):
    state = [[1/2,1/2,1/2,1/2]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi
def get_open_mps(sites):
    state = [[1,1,1,1]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi

def pattern_to_mps(pattern):
    pdict={"+":[1,0,0,0],"-":[0,1,0,0],"b":[0,0,1,0],"a":[0,0,0,1],"q":[0,0,1,1],"c":[1,1,0,0],"*":[1,1,1,1]}
    state = [pdict[p] for p in pattern]
    sites=[BlipSite(False) for _ in pattern]
    psi = MPS.from_product_state(sites, state)
    return psi

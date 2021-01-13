from tenpy.networks.mps import MPS
from functools import reduce
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc

def wrap_ndarray(ar):
    l1=LegCharge.from_trivial(ar.shape[0])
    l2=LegCharge.from_trivial(ar.shape[1])
    l3=LegCharge.from_trivial(ar.shape[2])
    l4=LegCharge.from_trivial(ar.shape[3])
    return npc.Array.from_ndarray(ar,[l1,l2,l3,l4],labels=["wL","wR","p","p*"])
def normalize_im():
    pass
def _multiply_W(w1,w2):
    pre=npc.tensordot(w1,w2,axes=[("p*",),("p",)])
    pre=pre.combine_legs([(0,3),(1,4)])
    pre.ireplace_labels(["(?0.?3)","(?1.?4)"],["wL","wR"])
    return pre
def multiply_mpos(mpolist):
    Wps=[[m.get_W(i) for m in mpolist] for i in range(mpolist[0].L)]
    return MPO(mpolist[0].sites,[reduce(_multiply_W,Wp) for Wp in Wps])
def apply(mpo,mps,chi=None,options=None):
    mpo.IdL[0]=0
    mpo.IdR[-1]=0 #How is this my job?
    if options is None:
        if chi is None:
            mpo.apply_naively(mps)
            mps.canonical_form(renormalize=False)
            return
        else:
            options={"trunc_params":{"chi_max":chi},"verbose":False,"compression_method":"SVD"}
    mpo.apply(mps,options)

def apply_naively(mpo,mps):
    mpo.apply_naively(mps,options)

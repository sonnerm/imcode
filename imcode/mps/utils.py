from tenpy.networks.mps import MPS
from functools import reduce
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import LegCharge
from tenpy.algorithms.exact_diag import ExactDiag
import tenpy.linalg.np_conserved as npc

def mpo_to_dense(mpo):
    mpo.IdL[0]=0
    mpo.IdR[-1]=0 #How is this my job?
    ed=ExactDiag.from_H_mpo(mpo)
    ed.build_full_H_from_mpo()
    nda=ed.full_H.to_ndarray()
    nda=nda.reshape((2,2,4**(mpo.L-2),2,2,2,2,4**(mpo.L-2),2,2))[0,:,:,0,:,0,:,:,0,:]
    for i in range(mpo.L-2):
        nda = nda.reshape(2*4**i,4,4**(mpo.L*2-4-i)*2)[:,[0,2,3,1],:]
        nda = nda.reshape(4**(mpo.L-1+i)*2,4,4**(mpo.L-3-i)*2)[:,[0,2,3,1],:]
    nda = nda.reshape((2,)*(mpo.L*2-2)+(4**(mpo.L-1),)).transpose([0,]+[x for x in range(1,2*mpo.L-4,2)]+[2*mpo.L-3]+[x for x in list(range(2,2*mpo.L-2,2))[::-1]]+[2*mpo.L-2])
    nda = nda.reshape((4**(mpo.L-1),)+(2,)*(mpo.L*2-2)).transpose([0,1]+[x+1 for x in range(1,2*mpo.L-4,2)]+[2*mpo.L-2]+[x+1 for x in list(range(2,2*mpo.L-2,2))[::-1]])
    return nda.reshape(4**(mpo.L-1),4**(mpo.L-1))


def mps_to_dense(mps):
    psi = mps.get_theta(0, mps.L)
    psi = psi.take_slice([0, 0], ['vL', 'vR'])
    psi = psi.to_ndarray().reshape((2,2,4**(mps.L-2),2,2))[0,:,:,0,:]
    for i in range(mps.L-2):
        psi = psi.reshape(2*4**i,4,4**(mps.L-3-i)*2)[:,[0,2,3,1],:]
    psi = psi.reshape((2,)*(mps.L*2-2)).transpose([0,]+[x for x in range(1,2*mps.L-4,2)]+[2*mps.L-3]+[x for x in list(range(2,2*mps.L-2,2))[::-1]])
    return psi.ravel()*mps.norm
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

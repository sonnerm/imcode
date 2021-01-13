from . import flat
from . import fold
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
    if isinstance(mps.sites[0],fold.FoldSite):
        psi = psi.take_slice([0, 0], ['vL', 'vR'])
        psi = psi.to_ndarray().reshape((2,2,4**(mps.L-2),2,2))[0,:,:,0,:]
        for i in range(mps.L-2):
            psi = psi.reshape(2*4**i,4,4**(mps.L-3-i)*2)[:,[0,2,3,1],:]
        psi = psi.reshape((2,)*(mps.L*2-2)).transpose([0,]+[x for x in range(1,2*mps.L-4,2)]+[2*mps.L-3]+[x for x in list(range(2,2*mps.L-2,2))[::-1]])
        return psi.ravel()*mps.norm
    elif isinstance(mps.sites[0],flat.FlatSite):
        psi = npc.trace(psi,'vL', 'vR')
        psi = psi.to_ndarray()
        return psi.ravel()*mps.norm
    else:
        assert False

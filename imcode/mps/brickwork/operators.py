import numpy as np
from tenpy.linalg.charges import LegCharge
from tenpy.networks.mpo import MPO,MPOEnvironment
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site
import tenpy.linalg.np_conserved as npc
import numpy.linalg as la
from ..utils import multiply_mpos

class BrickworkSite(Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(4),["+","b","a","-"],)
def brickwork_F(gates):
    pass
def brickwork_Sa(t, gate):
    '''
        dual layer of the brickwork transfer matrix without boundary states
    '''
    # gate=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape((16,16))
    gate=np.einsum("abcd,efgh->aebfcgdh",gate.reshape((2,2,2,2)),gate.reshape((2,2,2,2)).conj()).reshape(16,16)
    u,s,v=la.svd(gate)
    gateb=(v).reshape((16,1,4,4))
    gatea=(u*s).T.reshape((1,16,4,4))
    leg_m=LegCharge.from_trivial(16)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    gatean=npc.Array.from_ndarray(gatea,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    gatebn=npc.Array.from_ndarray(gateb,[leg_m,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO([BrickworkSite() for _ in range(2*t)],[gatean,gatebn]*t)
# gate=np.random.random((4,4))+1.0j*np.random.random((4,4))
# import scipy.linalg as sla
# gate=sla.expm(1.0j*gate)
# u,s,v=la.svd(np.kron(gate,gate.conj()))
# dual=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape(16,16)
# np.allclose((u@np.diag(np.sqrt(s)))@(np.diag(np.sqrt(s))@v),np.kron(gate,gate.conj()))



def brickwork_Sb(t, gate,init=np.eye(4),final=np.eye(4)):
    '''
        dual layer of the brickwork transfer matrix with boundary states
    '''
    gate=gate.reshape((2,2,2,2))
    inita=np.einsum("cdab,abef,cdgh->egfh",init.reshape((2,2,2,2)),gate,gate.conj()) #No idea why init.T but works
    inita=inita.reshape((1,1,4,4))
    final=np.einsum("abcd->acbd",final.reshape((2,2,2,2))).reshape((4,4))
    finala=final.reshape((1,1,4,4))
    gate=np.einsum("abcd,efgh->aebfcgdh",gate.reshape((2,2,2,2)),gate.reshape((2,2,2,2)).conj()).reshape(16,16)
    u,s,v=la.svd(gate)
    gateb=(v).reshape((16,1,4,4))
    gatea=(u*s).T.reshape((1,16,4,4))
    leg_m=LegCharge.from_trivial(16)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    initan=npc.Array.from_ndarray(inita,[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    finalan=npc.Array.from_ndarray(finala,[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    gatean=npc.Array.from_ndarray(gatea,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    gatebn=npc.Array.from_ndarray(gateb,[leg_m,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO([BrickworkSite() for _ in range(2*t)],[initan]+[gatean,gatebn]*(t-1)+[finalan])
def brickwork_T(t,even,odd,init=np.eye(4),final=np.eye(4)):
    return multiply_mpos([brickwork_Sa(t,even),brickwork_Sb(t,odd,init,final)])
def brickwork_La(t):
    leg_m=LegCharge.from_trivial(4)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    M1a=np.eye(4).reshape(1,4,4)
    M2a=np.eye(4).reshape(4,1,4)
    M1an=npc.Array.from_ndarray(M1a,[leg_t,leg_m.conj(),leg_p],labels=["vL","vR","p"])
    M2an=npc.Array.from_ndarray(M2a,[leg_m,leg_t.conj(),leg_p],labels=["vL","vR","p"])
    Ws=[M1an,M2an]*t
    Svs=[np.ones(W.shape[0]) / np.sqrt(W.shape[1]) for W in Ws]
    Svs.append([1.0])
    Svs[0]=[1.0]
    return MPS([BrickworkSite() for _ in range(2*t)],Ws,Svs)

def brickwork_Lb(t,lop,init=np.eye(2),final=np.eye(2)):
    leg_m=LegCharge.from_trivial(4)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    lop=np.kron(lop,lop.conj()).T
    M1a=lop.reshape(1,4,4)
    M2a=np.eye(4).reshape(4,1,4)
    inita=(lop@init.T.ravel()).reshape(1,1,4)
    finala=final.reshape(1,1,4)
    initan=npc.Array.from_ndarray(inita,[leg_t,leg_t.conj(),leg_p],labels=["vL","vR","p"])
    finalan=npc.Array.from_ndarray(finala,[leg_t,leg_t.conj(),leg_p],labels=["vL","vR","p"])
    M1an=npc.Array.from_ndarray(M1a,[leg_t,leg_m.conj(),leg_p],labels=["vL","vR","p"])
    M2an=npc.Array.from_ndarray(M2a,[leg_m,leg_t.conj(),leg_p],labels=["vL","vR","p"])
    Ws=[initan]+[M1an,M2an]*(t-1)+[finalan]
    Svs=[np.ones(W.shape[0]) / np.sqrt(W.shape[1]) for W in Ws]
    Svs.append([1.0])
    Svs[0]=[1.0]
    return MPS([BrickworkSite() for _ in range(2*t)],Ws,Svs)

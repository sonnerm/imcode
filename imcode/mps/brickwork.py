from tenpy.linalg.charges import LegCharge
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from .utils import multiply_mpos

class BrickworkSite(Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(4),["+","b","a","-"],)
def brickwork_S(t, gate):
    '''
        "gate" portion of the brickwork transfer matrix
    '''
    gate=np.einsum("abcd,efgh->aecgbfdh",gate.reshape((2,2,2,2)),gate.conj().reshape((2,2,2,2))).reshape((16,16))
    u,s,v=la.svd(gate)
    gatea=(u@np.diag(np.sqrt(s))).T.reshape((1,16,4,4))
    gateb=(np.diag(np.sqrt(s))@v).reshape((16,1,4,4))
    leg_m=LegCharge.from_trivial(16)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    gatean=npc.Array.from_ndarray(gatea,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    gatebn=npc.Array.from_ndarray(gateb,[leg_m,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO([BrickworkSite(x) for x in range(2*t)],[gatean,gatebn]*t)

def brickwork_L(t,lop,init=(0.5,0,0,0.5),final=(1,0,0,1)):
    '''
        "local operator" portion of the brickwork transfer matrix
    '''
    lop=np.kron(lop,lop.conj())
    lop=np.einsum("ab,cd->cabd",np.eye(4),lop)
    init=np.einsum("ab,d->abd",np.eye(4),init).reshape((1,4,4,4))
    final=np.einsum("ab,d->adb",lop,final).reshape((4,1,4,4))
    leg_m=LegCharge.from_trivial(4)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    lopn=npc.Array.from_ndarray(lop,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    initn=npc.Array.from_ndarray(init,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    finaln=npc.Array.from_ndarray(final,[leg_m,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO([BrickworkSite(x) for x in range(2*t)],[initn]+[lopn]*(2*t-2)+[finaln])

def brickwork_Lt(t,lop,init=(0.5,0,0,0.5),final=(1,0,0,1)):
    '''
        "local operator" portion of the brickwork transfer matrix
    '''
    lop=np.kron(lop,lop.conj())
    lop=np.einsum("ab,cd->cabd",np.eye(4),lop)
    init=np.einsum("ab,d->adb",np.eye(4),init).reshape((1,4,4,4))
    final=np.einsum("ab,d->abd",lop,final).reshape((4,1,4,4))
    leg_m=LegCharge.from_trivial(4)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    lopn=npc.Array.from_ndarray(lop,[leg_m,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    initn=npc.Array.from_ndarray(init,[leg_t,leg_m.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    finaln=npc.Array.from_ndarray(final,[leg_m,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"])
    return MPO([BrickworkSite(x) for x in range(2*t)],[initn]+[lopn]*(2*t-2)+[finaln])
def brickwork_open_boundary_im(t):
    return ret

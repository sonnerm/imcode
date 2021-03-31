from tenpy.linalg.charges import LegCharge
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from .utils import multiply_mpos
def brickwork_F(gates,lops):
    assert len(lops)%2==0
    ls=[npc.Array.from_ndarray(l,[leg_t,leg_t.conj(),leg_p,leg_p.conj()],labels=["wL","wR","p","p*"]) for l in lops]
    evs=dense_kron([g for g in gates[::2]])
    if len(gates)==len(lops)-1:
        gates=list(gates)+[np.eye(4)]
    ods=dense_kron([g for g in gates[1::2]])
    ods=ods.reshape((ods.shape[0]//4,2,2,ods.shape[1]//4,2,2))
    ods=ods.transpose([2,0,1,5,3,4]).reshape((evs.shape[0],evs.shape[1]))
    return ods@ls@evs@ls

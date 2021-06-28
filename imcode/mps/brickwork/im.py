from .utils import BrickworkSite
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge
import numpy as np
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
def brickwork_pd(t):
    sites=[BrickworkSite() for _ in range(2*t)]
    state = [[1,0,0,1]] * len(sites)
    psi = MPS.from_product_state(sites, state)
    return psi

def brickwork_dephaser(t,gamma):
    leg_m=LegCharge.from_trivial(4)
    leg_t=LegCharge.from_trivial(1)
    leg_p=LegCharge.from_trivial(4)
    M1a=np.eye(4).reshape(1,4,4)
    M2a=np.array([[1,0,0,0],[0,1-gamma,0,0],[0,0,1-gamma,0],[0,0,0,1]]).reshape(4,1,4)
    M1an=npc.Array.from_ndarray(M1a,[leg_t,leg_m.conj(),leg_p],labels=["vL","vR","p"])
    M2an=npc.Array.from_ndarray(M2a,[leg_m,leg_t.conj(),leg_p],labels=["vL","vR","p"])
    Ws=[M1an,M2an]*t
    Svs=[np.ones(W.shape[0]) / np.sqrt(W.shape[1]) for W in Ws]
    Svs.append([1.0])
    Svs[0]=[1.0]
    return MPS([BrickworkSite() for _ in range(2*t)],Ws,Svs)

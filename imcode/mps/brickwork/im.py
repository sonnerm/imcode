from .utils import BrickworkSite
from .. import MPS
import numpy as np
def brickwork_Lb(t,lop,init=np.eye(2),final=np.eye(2)):
    lop=np.kron(lop,lop.conj()).T
    M1a=lop.reshape(1,4,4)
    M2a=np.eye(4).reshape(4,1,4)
    inita=(lop@init.T.ravel()).reshape(1,1,4)
    finala=final.reshape(1,1,4)
    Ws=[inita]+[M1a,M2a]*(t-1)+[finala]
    return MPS.from_matrices(Ws)
def brickwork_La(t):
    M1a=np.eye(4).reshape(1,4,4)
    M2a=np.eye(4).reshape(4,1,4)
    Ws=[M1a,M2a]*t
    return MPS.from_matrices(Ws)
def brickwork_pd(t):
    state = [[1,0,0,1]] * len(sites)
    psi = MPS.from_product_state(state)
    return psi

def brickwork_dephaser(t,gamma):
    M1a=np.eye(4).reshape(1,4,4)
    M2a=np.array([[1,0,0,0],[0,1-gamma,0,0],[0,0,1-gamma,0],[0,0,0,1]]).reshape(4,1,4)
    Ws=[M1a,M2a]*t
    return MPS.from_matrices(Ws)

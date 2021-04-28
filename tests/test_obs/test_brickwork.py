import pytest
import numpy as np
import imcode.dense as dense
import imcode.mps.brickwork as bw
import scipy.linalg as la
from tenpy.networks.mpo import MPOEnvironment,MPO
import numpy.linalg as nla
from ..utils import seed_rng

SZ=np.array([[1.0,0.0],[0.0,-1.0]])
SZ2=np.kron(SZ,np.eye(2))/2
SZ3=np.kron(SZ,np.eye(4))/np.sqrt(8)

def test_dense_brickwork_L2():
    seed_rng("bw_L2")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop1=lop1+lop1.T.conj()
    lop1=la.expm(1.0j*lop1)
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=lop2+lop2.T.conj()
    lop2=la.expm(1.0j*lop2)
    U=gop@np.kron(lop1,lop2)
    assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,3):
        S=dense.brickwork_Sb(t,U)
        B=dense.brickwork_La(t)
        print(B.shape,S.shape,t)

        assert B@S@B==pytest.approx(1.0)

        S=dense.brickwork_Sa(t,gop)
        B1=dense.brickwork_Lb(t,lop1)
        B2=dense.brickwork_Lb(t,lop2)
        assert B1@S@B2==pytest.approx(1.0)

        S=dense.brickwork_Sb(t,U,init=np.kron(SZ,np.eye(2)),final=np.kron(SZ,np.eye(2)))
        czz=B@S@B
        print(czz)
        assert czz==pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))

def test_mps_brickwork_L2():
    seed_rng("bw_L2")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    U=la.expm(1.0j*gop)
    for t in range(1,3):
        pass
        # S=bw.brickwork_Sb(t,U)
        # B1=bw.brickwork_La(t)
        # B2=bw.brickwork_La(t)
        # S.IdL[0]=0
        # S.IdR[-1]=0
        # for i in range(B1.L):
        #     B1.get_B(i).conj(True,True).conj(False,True)
        # assert MPOEnvironment(B1,S,B2).full_contraction(0)*B1.norm*B2.norm==pytest.approx(1.0)

        # S=bw.brickwork_Sa(t,gop)
        # B1=bw.brickwork_Lb(t,lop1)
        # B2=bw.brickwork_Lb(t,lop2)
        # assert B1@S@B2==pytest.approx(1.0)
        #
        # S=bw.brickwork_Sb(t,U,init=np.kron(SZ,np.eye(2)),final=np.kron(SZ,np.eye(2)))
        # czz=B@S@B
        # print(czz)
        # assert czz==pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))

import pytest
import numpy as np
import imcode.dense as dense
import imcode.mps.brickwork as bw
import imcode.mps as mps
import scipy.linalg as la
from tenpy.networks.mpo import MPOEnvironment,MPO
import numpy.linalg as nla
from ..utils import seed_rng
MAX_T=3
SZ=np.array([[1.0,0.0],[0.0,-1.0]])
SX=np.array([[0.0,1.0],[1.0,0.0]])
SZ2=np.kron(SZ,np.eye(2))
SZ3=np.kron(SZ,np.eye(4))

def test_dense_brickwork_L2_obc():
    seed_rng("bw_L2_obc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    U=gop
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,MAX_T):
        S=dense.brickwork_Sb(t,U)
        B=dense.brickwork_La(t)
        assert B@S@B==pytest.approx(4.0)
        S=dense.brickwork_Sb(t,U,init=SZ2,final=SZ2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        czz=B@S@B
        print(czz)
        assert czz==czzc
        S=dense.brickwork_Sb(t,U,init=init,final=final)
        czz=B@S@B
        print(czz)
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))
def test_dense_brickwork_L2_bbc():
    seed_rng("bw_L2_bbc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop1=lop1+lop1.T.conj()
    lop1=la.expm(1.0j*lop1)
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=lop2+lop2.T.conj()
    lop2=la.expm(1.0j*lop2)
    init1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    # lop1=np.eye(2)
    # lop2=np.eye(2)
    U=np.kron(lop1,lop2)@gop
    assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,3):
        S=dense.brickwork_Sa(t,gop)
        B1=dense.brickwork_Lb(t,lop1)
        B2=dense.brickwork_Lb(t,lop2)
        assert B1@S@B2==pytest.approx(4.0)

        B1=dense.brickwork_Lb(t,lop1,init=SZ,final=SZ)
        B2=dense.brickwork_Lb(t,lop2)
        czz=B1@S@B2
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        print((czz,czzc))
        assert czz==czzc

        B1=dense.brickwork_Lb(t,lop1,init=init1,final=final1)
        B2=dense.brickwork_Lb(t,lop2,init=init2,final=final2)
        czz=B1@S@B2
        czzc=np.trace(np.kron(init1,init2)@nla.matrix_power(U,t)@np.kron(final1,final2)@nla.matrix_power(U.T.conj(),t))
        assert czz==pytest.approx(czzc)

def test_mps_brickwork_L2_bbc():
    seed_rng("bw_L2_bbc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    lop1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop1=lop1+lop1.T.conj()
    lop1=la.expm(1.0j*lop1)
    lop2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    lop2=lop2+lop2.T.conj()
    lop2=la.expm(1.0j*lop2)
    init1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    init2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final1=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final2=np.random.random((2,2))+np.random.random((2,2))*1.0j
    U=np.kron(lop1,lop2)@gop
    # assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,10):
        S=bw.brickwork_Sa(t,gop)
        B1=bw.brickwork_Lb(t,lop1)
        B2=bw.brickwork_Lb(t,lop2)
        assert mps.embedded_obs(B1,S,B2) == pytest.approx(4.0)

        B1=bw.brickwork_Lb(t,lop1,init=SZ,final=SZ)
        B2=bw.brickwork_Lb(t,lop2)
        czz=mps.embedded_obs(B1,S,B2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        print((czz,czzc))
        assert czz==czzc

        B1=bw.brickwork_Lb(t,lop1,init=init1,final=final1)
        B2=bw.brickwork_Lb(t,lop2,init=init2,final=final2)
        czz=mps.embedded_obs(B1,S,B2)
        czzc=np.trace(np.kron(init1,init2)@nla.matrix_power(U,t)@np.kron(final1,final2)@nla.matrix_power(U.T.conj(),t))
        assert czz==pytest.approx(czzc)
def test_mps_brickwork_L2_obc():
    seed_rng("bw_mps_L2_obc")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    gop=gop+gop.T.conj()
    gop=la.expm(1.0j*gop)
    # gop=np.eye(4)
    U=gop
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    # assert U==pytest.approx(dense.brickwork_F([U]))
    for t in range(1,10):
        S=bw.brickwork_Sb(t,U)
        B=bw.brickwork_La(t)
        assert mps.embedded_obs(B,S,B) == pytest.approx(4.0)
        S=bw.brickwork_Sb(t,U,init=SZ2,final=SZ2)
        czzc=pytest.approx(np.trace(SZ2@nla.matrix_power(U,t)@SZ2@nla.matrix_power(U.T.conj(),t)))
        czz=mps.embedded_obs(B,S,B)
        print((t,czz,czzc))
        assert czz==czzc
        S=bw.brickwork_Sb(t,U,init=init,final=final)
        czz=mps.embedded_obs(B,S,B)
        assert czz==pytest.approx(np.trace(init@nla.matrix_power(U,t)@final@nla.matrix_power(U.T.conj(),t)))

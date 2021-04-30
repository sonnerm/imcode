import numpy as np
# from imcode import sparse
import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
import imcode.mps.brickwork as bw
from ..utils import seed_rng
import pytest

@pytest.fixture(scope="module")
def dense_brickwork_Sb():
    T=2
    seed_rng("dense_brickwork_Sb")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    return (dense.brickwork_Sb(T,gop,init=init,final=final),(T,gop,init,final))

@pytest.fixture(scope="module")
def dense_brickwork_Sa():
    T=2
    seed_rng("dense_brickwork_Sa")
    gop=np.random.random((4,4))+np.random.random((4,4))*1.0j
    return (dense.brickwork_Sa(T,gop),(T,gop))

@pytest.fixture(scope="module")
def dense_brickwork_Lb():
    T=2
    lop=np.random.random((2,2))+np.random.random((2,2))*1.0j
    init=np.random.random((2,2))+np.random.random((2,2))*1.0j
    final=np.random.random((2,2))+np.random.random((2,2))*1.0j
    return (dense.brickwork_Lb(T,lop,init=init,final=final),(T,lop,init,final))

@pytest.fixture(scope="module")
def dense_brickwork_La():
    T=2
    return (dense.brickwork_La(T),(T,))

@pytest.fixture(scope="module")
def dense_brickwork_T():
    T=2
    seed_rng("dense_brickwork_T")
    even=np.random.random((4,4))+np.random.random((4,4))*1.0j
    odd=np.random.random((4,4))+np.random.random((4,4))*1.0j
    init=np.random.random((4,4))+np.random.random((4,4))*1.0j
    final=np.random.random((4,4))+np.random.random((4,4))*1.0j
    return (dense.brickwork_T(T,even,odd,init,final),(T,even,odd,init,final))


def test_mps_brickwork_Sa(dense_brickwork_Sa):
    mih=bw.brickwork_Sa(*dense_brickwork_Sa[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_brickwork_Sa[0])
    assert mih.chi==[1,16]*(mih.L//2)+[1]

def test_mps_brickwork_Sb(dense_brickwork_Sb):
    mih=bw.brickwork_Sb(*dense_brickwork_Sb[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_brickwork_Sb[0])
    assert mih.chi==[1]+[1,16]*(mih.L//2-1)+[1,1]

def test_mps_brickwork_La(dense_brickwork_La):
    mih=bw.brickwork_La(*dense_brickwork_La[1])
    assert mps.mps_to_dense(mih)==pytest.approx(dense_brickwork_La[0])
#
# def test_mps_brickwork_Lb(dense_brickwork_Lb):
#     mih=bw.brickwork_Lb(*dense_brickwork_Lb[1])
#     assert mps.mps_to_dense(mih)==pytest.approx(dense_brickwork_Lb[0])
#
# def test_mps_brickwork_T(dense_brickwork_T):
#     mih=bw.brickwork_T(*dense_brickwork_T[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_brickwork_T[0])
#     assert mih.chi==[1]+[16]*(mih.L-1)+[1]

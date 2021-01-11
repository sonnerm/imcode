import numpy as np
import numpy.linalg as la
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import seed_rng
import pytest

@pytest.fixture(scope="module")
def dense_ising_Jr_T():
    T=3
    seed_rng("dense_ising_Jr_T")
    g=np.random.normal()
    h=np.random.normal()
    return (dense.ising_Jr_T(T,g,h),(T,g,h))

def test_dense_ising_Jr_T(dense_ising_Jr_T):
    diT=dense_ising_Jr_T[0]
    assert diT.dtype==np.complex_
    proj=la.matrix_power(diT,dense_ising_Jr_T[1][0]*2)
    assert proj==pytest.approx(diT@proj)

@pytest.mark.xfail
def test_sparse_ising_Jr_T(dense_ising_Jr_T):
    sih=sparse.ising_Jr_T(*dense_ising_Jr_T[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_Jr_T[0])

def test_mps_ising_Jr_T(dense_ising_Jr_T):
    mih=mps.ising_Jr_T(*dense_ising_Jr_T[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_Jr_T[0])
    #assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]

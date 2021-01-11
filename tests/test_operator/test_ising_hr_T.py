import numpy as np
import numpy.linalg as la
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import seed_rng
import pytest

@pytest.fixture(scope="module")
def dense_ising_hr_T():
    T=3
    seed_rng("dense_ising_hr_T")
    J=np.random.normal()
    g=np.random.normal()
    return (dense.ising_hr_T(T,J,g),(T,J,g))

@pytest.mark.skip()
def test_ising_hr_disorder(dense_ising_hr_T):
    SAMPLE=1000
    seed_rng("dense_ising_hr_T_dis")
    Tmd=np.zeros_like(dense_ising_hr_T[0])
    t,J,g=dense_ising_hr_T[1]
    for i in range(SAMPLE):
        Tmd+=dense.ising_T(t,J,g,np.random.uniform(0,2*np.pi))
    Tmd/=SAMPLE
    print(Tmd)
    print(dense_ising_hr_T[0])
    assert Tmd==pytest.approx(dense_ising_hr_T[0],rel=1e-2,abs=1e-2)

def test_dense_ising_hr_T(dense_ising_hr_T):
    diT=dense_ising_hr_T[0]
    assert diT.dtype==np.complex_
    proj=la.matrix_power(diT,dense_ising_hr_T[1][0]*2)
    assert proj==pytest.approx(diT@proj)
def test_sparse_ising_hr_T(dense_ising_hr_T):
    sih=sparse.ising_hr_T(*dense_ising_hr_T[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_hr_T[0])

def test_mps_ising_hr_T(dense_ising_hr_T):
    mih=mps.ising_hr_T(*dense_ising_hr_T[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_T[0])
    #assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]

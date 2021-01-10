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
    seed_rng("dense_ising_T")
    J=np.random.normal()
    g=np.random.normal()
    return (dense.ising_hr_T(T,J,g),(T,J,g))

def test_dense_ising_hr_T(dense_ising_hr_T):
    diT=dense_ising_hr_T[0]
    assert diT.dtype==np.complex_
    proj=la.matrix_power(diT,dense_ising_hr_T[1][0]*2)
    assert proj==pytest.approx(diT@proj)
@pytest.mark.skip()
def test_sparse_ising_hr_T(dense_ising_hr_T):
    sih=sparse.ising_hr_T(*dense_ising_hr_T[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_hr_T[0])

@pytest.mark.skip()
def test_mps_ising_hr_T(dense_ising_hr_T):
    mih=mps.ising_hr_T(*dense_ising_hr_T[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_T[0])
    assert mih.chi==[1]+[4]*(mih.L-1)+[1]

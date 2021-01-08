import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from .utils import sparse_eq
import pytest
@pytest.fixture(scope="module")
def dense_Jr_operator():
    T=2
    return (dense.Jr_operator(T),(T))


def test_dense_Jr_operator(dense_Jr_operator):
    dih=dense_Jr_operator[0]
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.0,1.0} #only zero and one's


def test_sparse_Jr_operator(dense_Jr_operator):
    sih=sparse.Jr_operator(dense_Jr_operator[1])
    sparse_eq(sih,dense_Jr_operator[0])
@pytest.mark.skip("Needs reimplementation")
def test_mps_Jr_operator(dense_Jr_operator):
    mih=mps.Jr_operator(dense_Jr_operator[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_Jr_operator[0])

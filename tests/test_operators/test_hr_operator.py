import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
import pytest
@pytest.fixture(scope="module")
def dense_hr_operator():
    T=3
    return (dense.hr_operator(T),(T))


def test_dense_hr_operator(dense_hr_operator):
    dih=dense_hr_operator[0]
    assert dih.dtype==np.float_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert np.diag(dih)**2==pytest.approx(np.diag(dih)) #projection


def test_sparse_hr_operator(dense_hr_operator):
    sih=sparse.hr_operator(dense_hr_operator[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_hr_operator[0])
def test_mps_hr_operator(dense_hr_operator):
    mih=mps.hr_operator(dense_hr_operator[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_hr_operator[0])

import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
import pytest
@pytest.fixture(scope="module")
def dense_hr_operator_odd():
    T=3
    return (dense.hr_operator(T),(T))

@pytest.fixture(scope="module")
def dense_hr_operator_even():
    T=4
    return (dense.hr_operator(T),(T))


def test_dense_hr_operator_odd(dense_hr_operator_odd):
    dih=dense_hr_operator_odd[0]
    assert dih.dtype==np.float_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert np.diag(dih)**2==pytest.approx(np.diag(dih)) #projection

def test_dense_hr_operator_even(dense_hr_operator_even):
    dih=dense_hr_operator_even[0]
    assert dih.dtype==np.float_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert np.diag(dih)**2==pytest.approx(np.diag(dih)) #projection



def test_sparse_hr_operator_odd(dense_hr_operator_odd):
    sih=sparse.hr_operator(dense_hr_operator_odd[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_hr_operator_odd[0])

def test_sparse_hr_operator_even(dense_hr_operator_even):
    sih=sparse.hr_operator(dense_hr_operator_even[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_hr_operator_even[0])
def test_mps_hr_operator_odd(dense_hr_operator_odd):
    mih=mps.hr_operator(dense_hr_operator_odd[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_hr_operator_odd[0])

def test_mps_hr_operator_even(dense_hr_operator_even):
    mih=mps.hr_operator(dense_hr_operator_even[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_hr_operator_even[0])

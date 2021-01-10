import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
import pytest
@pytest.fixture(scope="module")
def dense_Jr_operator_odd():
    T=3
    return (dense.Jr_operator(T),(T))

@pytest.fixture(scope="module")
def dense_Jr_operator_even():
    T=4
    return (dense.Jr_operator(T),(T))


def test_dense_Jr_operator_odd(dense_Jr_operator_odd):
    dih=dense_Jr_operator_odd[0]
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.0,1.0} #only zero and one's

def test_dense_Jr_operator_even(dense_Jr_operator_even):
    dih=dense_Jr_operator_even[0]
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.0,1.0} #only zero and one's


@pytest.mark.skip("Needs reimplementation")
def test_sparse_Jr_operator_odd(dense_Jr_operator_odd):
    sih=sparse.Jr_operator(dense_Jr_operator_odd[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_Jr_operator_odd[0])

@pytest.mark.skip("Needs reimplementation")
def test_sparse_Jr_operator_even(dense_Jr_operator_even):
    sih=sparse.Jr_operator(dense_Jr_operator_even[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_Jr_operator_even[0])

def test_mps_Jr_operator_odd(dense_Jr_operator_odd):
    mih=mps.Jr_operator(dense_Jr_operator_odd[1])
    print(np.array(np.nonzero(mps.mpo_to_dense(mih)[1,:])))
    print(np.array(np.nonzero(dense_Jr_operator_odd[0][1,:])))
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_Jr_operator_odd[0])
def test_mps_Jr_operator_even(dense_Jr_operator_even):
    mih=mps.Jr_operator(dense_Jr_operator_even[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_Jr_operator_even[0])

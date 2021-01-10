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
    assert set(list(np.ravel(dih)))=={0.0,0.5} #only zero and one's

def test_dense_Jr_operator_even(dense_Jr_operator_even):
    dih=dense_Jr_operator_even[0]
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.0,0.5} #only zero and one's


@pytest.mark.skip("Needs reimplementation")
def test_sparse_Jr_operator_odd(dense_Jr_operator_odd):
    sih=sparse.Jr_operator(dense_Jr_operator_odd[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_Jr_operator_odd[0])

@pytest.mark.skip("Needs reimplementation")
def test_sparse_Jr_operator_even(dense_Jr_operator_even):
    sih=sparse.Jr_operator(dense_Jr_operator_even[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_Jr_operator_even[0])

def test_mps_Jr_operator_odd(dense_Jr_operator_odd):
    t=dense_Jr_operator_odd[1]
    mih=mps.Jr_operator(t)
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_Jr_operator_odd[0])
    assert mih.chi==[1]+[2*i+1 for i in range(0,(t+1)//2)]+[2*i-1 for i in range((t-1)//2,0,-1)]+[1]
def test_mps_Jr_operator_even(dense_Jr_operator_even):
    t=dense_Jr_operator_even[1]
    mih=mps.Jr_operator(t)
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_Jr_operator_even[0])
    assert mih.chi==[1]+[2*i+1 for i in range((t+1)//2)]+[2*i-1 for i in range((t+1)//2,0,-1)]+[1]

def test_dense_Jr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.Jr_operator(1)
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.5} #in this case it is all ones
    dih=dense.Jr_operator(2)
    assert dih.dtype==np.float_#real
    assert dih.T==pytest.approx(dih) #symmetric
    assert set(list(np.ravel(dih)))=={0.0,0.5} #only zero and one's

@pytest.mark.skip("Needs reimplementation")
def test_sparse_Jr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.Jr_operator(1)
    sih=sparse.Jr_operator(1)
    assert sparse.sparse_to_dense(sih)==pytest.approx(dih)
    dih=dense.Jr_operator(2)
    sih=dense.Jr_operator(2)
    assert sparse.sparse_to_dense(sih)==pytest.approx(dih)

def test_mps_Jr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.Jr_operator(1)
    mih=mps.Jr_operator(1)
    assert mps.mpo_to_dense(mih)==pytest.approx(dih)
    dih=dense.Jr_operator(2)
    mih=mps.Jr_operator(2)
    assert mps.mpo_to_dense(mih)==pytest.approx(dih)

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

@pytest.fixture(scope="module")
def dense_hr_operator_T1():
    T=1
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
    t=dense_hr_operator_odd[1]
    mih=mps.hr_operator(t)
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_hr_operator_odd[0])
    assert mih.chi==[1]+[2*i+1 for i in range(0,(t+1)//2)]+[2*i-1 for i in range((t-1)//2,0,-1)]+[1]

def test_mps_hr_operator_even(dense_hr_operator_even):
    t=dense_hr_operator_even[1]
    mih=mps.hr_operator(dense_hr_operator_even[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_hr_operator_even[0])
    assert mih.chi==[1]+[2*i+1 for i in range((t+1)//2)]+[2*i-1 for i in range((t+1)//2,0,-1)]+[1]

def test_dense_hr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.hr_operator(1)
    assert dih.dtype==np.float_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert np.diag(dih)**2==pytest.approx(np.diag(dih)) #projection
    dih=dense.hr_operator(2)
    assert dih.dtype==np.float_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert np.diag(dih)**2==pytest.approx(np.diag(dih)) #projection

def test_sparse_hr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.hr_operator(1)
    sih=sparse.hr_operator(1)
    assert sparse.sparse_to_dense(sih)==pytest.approx(dih)
    dih=dense.hr_operator(2)
    sih=dense.hr_operator(2)
    assert sparse.sparse_to_dense(sih)==pytest.approx(dih)
def test_mps_hr_operator_short():
    # Essentially check whether code fails for short times
    dih=dense.hr_operator(1)
    mih=mps.hr_operator(1)
    assert mps.mpo_to_dense(mih)==pytest.approx(dih)
    dih=dense.hr_operator(2)
    mih=mps.hr_operator(2)
    assert mps.mpo_to_dense(mih)==pytest.approx(dih)

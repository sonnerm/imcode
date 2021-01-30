import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import seed_rng
import pytest

@pytest.fixture(scope="module")
def dense_dephaser_operator():
    T=3
    seed_rng("dense_dephaser_operator")
    gamma=np.random.random() # between 0 and 1
    return (dense.dephaser_operator(T,gamma),(T,gamma))

def test_dense_dephaser_operator(dense_dephaser_operator):
    diJ=dense_dephaser_operator[0]
    assert np.diag(np.diag(diJ))==pytest.approx(diJ) #diagonal

def test_sparse_dephaser_operator(dense_dephaser_operator):
    sih=sparse.dephaser_operator(*dense_dephaser_operator[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_dephaser_operator[0])


def test_fold_dephaser_operator(dense_dephaser_operator):
    mih=mps.fold.dephaser_operator(*dense_dephaser_operator[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_dephaser_operator[0])
    assert mih.chi==[1]*(mih.L+1)

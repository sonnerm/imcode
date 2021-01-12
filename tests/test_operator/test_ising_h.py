import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import seed_rng
import pytest
# @pytest.fixture(scope="module")
# def dense_ising_h_complex():
#     T=2
#     seed_rng("dense_ising_h_complex")
#     h=np.random.normal()+np.random.normal()*1.0j
#     return (dense.ising_h(T,h),(T,h))

@pytest.fixture(scope="module")
def dense_ising_h_real():
    T=3
    seed_rng("dense_ising_h_real")
    h=np.random.normal()
    return (dense.ising_h(T,h),(T,h))

def test_dense_ising_h_real(dense_ising_h_real):
    dih=dense_ising_h_real[0]
    assert dih.dtype==np.complex_
    assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
    assert dih.conj()*dih==pytest.approx(np.eye(dih.shape[0])) #unitary
    # Not degenerate case
    assert dih.conj()!=pytest.approx(dih)

# def test_dense_ising_h_complex(dense_ising_h_complex):
#     dih=dense_ising_h_complex[0]
#     assert dih.dtype==np.complex_
#     assert np.diag(np.diag(dih))==pytest.approx(dih) #diagonal
#     # Not degenerate case
#     assert dih.conj()!=pytest.approx(dih)
#     assert dih.conj()*dih!=pytest.approx(np.eye(dih.shape[0]))

def test_sparse_ising_h_real(dense_ising_h_real):
    sih=sparse.ising_h(*dense_ising_h_real[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_h_real[0])

# def test_sparse_ising_h_complex(dense_ising_h_complex):
#     sih=sparse.ising_h(*dense_ising_h_complex[1])
#     sparse_eq(sih,dense_ising_h_complex[0])

def test_fold_ising_h_real(dense_ising_h_real):
    mih=mps.fold.ising_h(*dense_ising_h_real[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_h_real[0])
    assert mih.chi==[1]*(mih.L+1)

def test_flat_ising_h_real(dense_ising_h_real):
    mih=mps.flat.ising_h(*dense_ising_h_real[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_h_real[0])
    assert mih.chi==[1]*(mih.L+1)

# def test_mps_ising_h_complex(dense_ising_h_complex):
#     mih=mps.fold.ising_h(*dense_ising_h_complex[1])
#     print(np.diag(mps.mpo_to_dense(mih)))
#     print(np.diag(dense_ising_h_complex[0]))
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_h_complex[0])
#     assert mih.chi==[1]*(mih.L+1)

import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import seed_rng
import pytest

@pytest.fixture(scope="module")
def dense_ising_W_complex():
    T=3
    seed_rng("dense_ising_W_complex")
    g=np.random.normal()+np.random.normal()*1.0j
    return (dense.ising_W(T,g),(T,g))

@pytest.fixture(scope="module")
def dense_ising_W_real():
    T=3
    seed_rng("dense_ising_W_real")
    g=np.random.normal()
    return (dense.ising_W(T,g),(T,g))

def test_dense_ising_W_real(dense_ising_W_real):
    diW=dense_ising_W_real[0]
    assert diW.dtype==np.complex_
    assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
    assert diW.conj()==pytest.approx(diW)#real since there is always an even number of flips in total
    # Not degenerate case
    assert diW.conj()*diW!=pytest.approx(np.eye(diW.shape[0]))

def test_dense_ising_W_complex(dense_ising_W_complex):
    diW=dense_ising_W_complex[0]
    assert diW.dtype==np.complex_
    assert np.diag(np.diag(diW))==pytest.approx(diW) #diagonal
    # Not degenerate case
    assert diW.conj()!=pytest.approx(diW)
    assert diW.conj()*diW!=pytest.approx(np.eye(diW.shape[0]))

def test_sparse_ising_W_real(dense_ising_W_real):
    sih=sparse.ising_W(*dense_ising_W_real[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_W_real[0])

def test_sparse_ising_W_complex(dense_ising_W_complex):
    sih=sparse.ising_W(*dense_ising_W_complex[1])
    assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_W_complex[0])

def test_mps_ising_W_real(dense_ising_W_real):
    mih=mps.ising_W(*dense_ising_W_real[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_W_real[0])
    assert mih.chi==[1]+[4]*(mih.L-1)+[1]

def test_mps_ising_W_complex(dense_ising_W_complex):
    print(np.diag(dense_ising_W_complex[0]))
    mih=mps.ising_W(*dense_ising_W_complex[1])
    print(np.diag(mps.mpo_to_dense(mih)))
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_W_complex[0])
    assert mih.chi==[1]+[4]*(mih.L-1)+[1]

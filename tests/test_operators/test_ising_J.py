import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from ..utils import sparse_eq,seed_rng
import pytest
pytestmark=pytest.mark.skip("for now")
@pytest.fixture(scope="module")
def dense_ising_J_complex():
    T=2
    seed_rng("dense_ising_J_complex")
    J=np.random.normal()+np.random.normal()*1.0j
    return (dense.ising_J(T,J),(T,J))

@pytest.fixture(scope="module")
def dense_ising_J_real():
    T=2
    seed_rng("dense_ising_J_real")
    J=np.random.normal()
    return (dense.ising_J(T,J),(T,J))

def test_dense_ising_J_real(dense_ising_J_real):
    diJ=dense_ising_J_real[0]
    assert diJ.dtype==np.complex_
    assert diJ.T==pytest.approx(diJ) #symmetric
    # Not degenerate case
    assert diJ.conj()!=pytest.approx(diJ)
    assert diJ.conj()@diJ!=pytest.approx(np.eye(diJ.shape[0]))

def test_dense_ising_J_complex(dense_ising_J_complex):
    diJ=dense_ising_J_complex[0]
    assert diJ.dtype==np.complex_
    assert diJ.T==pytest.approx(diJ) #symmetric
    # Not degenerate case
    assert diJ.conj()!=pytest.approx(diJ)
    assert diJ.conj()*diJ!=pytest.approx(np.eye(diJ.shape[0]))

def test_sparse_ising_J_real(dense_ising_J_real):
    sih=sparse.ising_J(*dense_ising_J_real[1])
    sparse_eq(sih,dense_ising_J_real[0])

def test_sparse_ising_J_complex(dense_ising_J_complex):
    sih=sparse.ising_J(*dense_ising_J_complex[1])
    sparse_eq(sih,dense_ising_J_complex[0])

def test_mps_ising_J_real(dense_ising_J_real):
    mih=mps.ising_J(*dense_ising_J_real[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_J_real[0])
    assert mih.chi==[1]*(mih.L+1)

def test_mps_ising_J_complex(dense_ising_J_complex):
    mih=mps.ising_J(*dense_ising_J_complex[1])
    assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_J_complex[0])
    assert mih.chi==[1]*(mih.L+1)

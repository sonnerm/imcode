import numpy as np
from imcode import dense
from imcode import sparse
from imcode import mps
from .utils import sparse_eq
import pytest
pytestmark=pytest.mark.skip("skip everything")
@pytest.fixture(scope="module")
def dense_ising_H_complex():
    L=5
    np.random.seed(hash("dense_ising_H_complex")%(2**32))
    J=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    g=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    h=np.random.normal(size=(L,))+np.random.normal(size=(L,))*1.0j
    return (dense.ising_H(J,g,h),(J,g,h))

@pytest.fixture(scope="module")
def dense_ising_H_real():
    L=5
    np.random.seed(hash("dense_ising_H_real")%(2**32))
    J=np.random.normal(size=(L,))
    g=np.random.normal(size=(L,))
    h=np.random.normal(size=(L,))
    return (dense.ising_H(J,g,h),(J,g,h))
def test_dense_ising_H_real(dense_ising_H_real):
    diH=dense_ising_H_real[0]
    assert diH.dtype==np.float_
    assert diH.conj()==pytest.approx(diH)
    assert diH.T.conj()==pytest.approx(diH)
    assert diH.T==pytest.approx(diH)

def test_dense_ising_H_complex(dense_ising_H_complex):
    diH=dense_ising_H_complex[0]
    assert diH.dtype==np.complex_
    #This model is always symmetric
    assert diH.T==pytest.approx(diH)
    #ensure that tests cover generic case
    assert diH.conj()!=pytest.approx(diH)
    assert diH.T.conj()!=pytest.approx(diH)
def test_sparse_ising_H_real(dense_ising_H_real):
    siH=sparse.ising_H(*dense_ising_H_real[1])
    assert siH.dtype==np.float_
    sparse_eq(siH,dense_ising_H_real[0])

def test_sparse_ising_H_complex(dense_ising_H_complex):
    siH=sparse.ising_H(*dense_ising_H_complex[1])
    assert siH.dtype==np.complex_
    sparse_eq(siH,dense_ising_H_complex[0])

def test_mps_ising_H_real(dense_ising_H_real):
    miH=mps.ising_H(*dense_ising_H_real[1])
    assert mps.mps_to_dense(siH)==pytest.approx(dense_ising_H_real[0])

def test_mps_ising_H_complex(dense_ising_H_complex):
    miH=mps.ising_H(*dense_ising_H_complex[1])
    assert mps.mps_to_dense(siH)==pytest.approx(dense_ising_H_complex[0])

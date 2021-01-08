import numpy as np
import pytest
from pytest import mark
from utils import sparse_eq
import imcode.sparse as sparse
import imcode.dense as dense
def test_sparse_ising_diag():
    L=5
    np.random.seed(hash("sparse_ising_diag")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    sdi=sparse.ising_diag(J,h)
    assert sdi.dtype==np.complex_
    assert np.diag(sdi)==pytest.approx(dense.ising_H(J,[0.0]*L,h))
    J=np.random.normal(size=L)
    h=np.random.normal(size=L)
    sdi=sparse.ising_diag(J,h)
    assert sdi.dtype==np.float_
    assert np.diag(sdi)==pytest.approx(dense.ising_H(J,[0.0]*L,h))
def test_sparse_sxdiagonallinearoperator():
    sxd=sparse.SxDiagonalLinearOperator(np.array([1,-1]))
    sparse_eq(sxd,dense.SX)
    sxd=sparse.SxDiagonalLinearOperator(np.array([1,1,1,1]))
    sparse_eq(sxd,np.eye(4))

@mark.skip("Not written yet")
def test_sparse_diagonallinearoperator():
    pass

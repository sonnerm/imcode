import numpy as np
import pytest
from pytest import mark
from .utils import seed_rng
import imcode.sparse as sparse
import imcode.dense as dense
def test_sparse_ising_diag():
    L=5
    seed_rng("sparse_ising_diag")
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
    assert sparse.sparse_to_dense(sxd)==pytest.approx(dense.SX)
    sxd=sparse.SxDiagonalLinearOperator(np.array([1,1,1,1]))
    assert sparse.sparse_to_dense(sxd)==pytest.approx(np.eye(4))

@mark.skip("Not written yet")
def test_sparse_diagonallinearoperator():
    pass

def test_sparse_algebra():
    seed_rng("sparse_algebra")
    L=64
    sx1=sparse.SxDiagonalLinearOperator(np.random.normal((L,))+1.0j*np.random.normal((L,)))
    sx2=sparse.SxDiagonalLinearOperator(np.random.normal((L,))+1.0j*np.random.normal((L,)))
    sz1=sparse.DiagonalLinearOperator(np.random.normal((L,))+1.0j*np.random.normal((L,)))
    sz2=sparse.DiagonalLinearOperator(np.random.normal((L,))+1.0j*np.random.normal((L,)))
    sp=sx1@(sz2+sx2)@sz1
    de=sparse.sparse_to_dense(sx1)@(sparse.sparse_to_dense(sz2)+sparse.sparse_to_dense(sx2))@sparse.sparse_to_dense(sz1)
    assert sparse.sparse_to_dense(sp)==pytest.approx(de)
    assert sparse.sparse_to_dense(sp.T)==pytest.approx(de.T)
    assert sparse.sparse_to_dense(sp.adjoint())==pytest.approx(de.T.conj())
    assert sparse.sparse_to_dense(sp.T.adjoint())==pytest.approx(de.conj())

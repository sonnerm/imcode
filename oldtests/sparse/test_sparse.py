import numpy as np
import scipy.linalg as scla
import pytest
from pytest import mark
import imcode.sparse as sparse
import imcode.dense as dense
def test_sparse_sxdiagonallinearoperator(seed_rng):
    sxd=sparse.SxDiagonalLinearOperator(np.array([1,-1]))
    assert sparse.sparse_to_dense(sxd)==pytest.approx(dense.SX)
    sxd=sparse.SxDiagonalLinearOperator(np.array([1,1,1,1]))
    assert sparse.sparse_to_dense(sxd)==pytest.approx(np.eye(4))


def test_sparse_algebra(seed_rng):
    L=64
    sx1=sparse.SxDiagonalLinearOperator(np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,)))
    sx2=sparse.SxDiagonalLinearOperator(np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,)))
    sz1=sparse.DiagonalLinearOperator(np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,)))
    sz2=sparse.DiagonalLinearOperator(np.random.normal(size=(L,))+1.0j*np.random.normal(size=(L,)))
    sp=sx1@(sz2+sx2)@sz1
    de=sparse.sparse_to_dense(sx1)@(sparse.sparse_to_dense(sz2)+sparse.sparse_to_dense(sx2))@sparse.sparse_to_dense(sz1)
    assert sparse.sparse_to_dense(sp)==pytest.approx(de)
    assert sparse.sparse_to_dense(sp.T)==pytest.approx(de.T)
    assert sparse.sparse_to_dense(sp.adjoint())==pytest.approx(de.T.conj())
    assert sparse.sparse_to_dense(sp.T.adjoint())==pytest.approx(de.conj())
def test_fwht(seed_rng):
    L=5
    ar1=np.random.normal(size=(2**L,))
    ar2=ar1.copy()
    sparse.fwht(ar2)
    ar1=scla.hadamard(2**L)@ar1
    assert ar1==pytest.approx(ar2)

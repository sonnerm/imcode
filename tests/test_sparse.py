import numpy as np
import pytest
from pytest import mark
import imcode.sparse as sparse
import imcode.dense as dense
def test_ising_H():
    # Testing sparse implementation of ising_H against dense implementation of ising_H
    L=5
    np.random.seed(hash("sparse_test_ising_H")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    assert (sparse.ising_H(J,g,h)@np.eye(2**L)).dtype==np.complex_
    assert sparse.ising_H(J,g,h)@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h))
    assert sparse.ising_H(J,g,h).conj()@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).conj())
    assert sparse.ising_H(J,g,h).T.conj()@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).T.conj())
    assert sparse.ising_H(J,g,h).T@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).T)
    # Test hermiticity, realness
    J=np.random.normal(size=L)
    g=np.random.normal(size=L)
    h=np.random.normal(size=L)
    assert (sparse.ising_H(J,g,h)@np.eye(2**L)).dtype==np.float_
    assert sparse.ising_H(J,g,h)@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h))
    assert sparse.ising_H(J,g,h).conj()@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).conj())
    assert sparse.ising_H(J,g,h).T.conj()@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).T.conj())
    assert sparse.ising_H(J,g,h).T@np.eye(2**L)==pytest.approx(dense.ising_H(J,g,h).T)
def test_ising_diag():
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
def test_ising_F():
    L=5
    np.random.seed(hash("sparse_ising_F")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    assert sparse.ising_F(J,g,h)@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h))
    assert sparse.ising_F(J,g,h).conj()@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).conj())
    assert sparse.ising_F(J,g,h).T.conj()@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).T.conj())
    assert sparse.ising_F(J,g,h).T@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).T)
@mark.skip("Not written")
def test_ising_T():
    t=3
    np.random.seed(hash("sparse_ising_T")%2**32)
    J=np.random.normal()+1.0j*np.random.normal()
    g=np.random.normal()+1.0j*np.random.normal()
    h=np.random.normal()+1.0j*np.random.normal()
    assert sparse.ising_F(J,g,h)@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h))
    assert sparse.ising_F(J,g,h).conj()@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).conj())
    assert sparse.ising_F(J,g,h).T.conj()@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).T.conj())
    assert sparse.ising_F(J,g,h).T@np.eye(2**L)==pytest.approx(dense.ising_F(J,g,h).T)

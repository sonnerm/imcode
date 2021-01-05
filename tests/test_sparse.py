import numpy as np
import pytest
import imcode.sparse as sparse
import imcode.dense as dense
def test_ising_H():
    # Testing sparse implementation of ising_H against dense implementation of ising_H
    L=5
    np.random.seed(hash("sparse_test_ising_H")%2**32)
    J=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    g=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    h=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    assert sparse.ising_H(J,g,h).to_dense()==pytest.approx(dense.ising_H(J,g,h))
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
def test_ising_linear_operator():
    #Test the linear operator class
    np.random.seed(hash("sparse_test_ising_linear_operator")%2**32)
    L=2**5
    D1=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    D2=np.random.normal(size=L)+1.0j*np.random.normal(size=L)
    op_sp=sparse.IsingLinearOperator(D1,D2)
    op_de=op_sp.to_dense()
    assert (op_de == op_sp@np.eye(L)).all() # should work exactly, no tolerances
    assert (np.eye(L) == sparse.IsingLinearOperator(np.ones(L),np.ones(L)).to_dense()).all()

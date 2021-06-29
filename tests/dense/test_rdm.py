import numpy as np
import numpy.linalg as la
import imcode.dense as dense
import pytest

def test_fixed_rdm():
    vec=np.zeros((8,))
    vec[0]=1.0
    assert dense.rdm(vec,[0])==pytest.approx(np.array([[1,0],[0,0]]))
    assert dense.rdm(vec,[1])==pytest.approx(np.array([[1,0],[0,0]]))
    assert dense.rdm(vec,[2])==pytest.approx(np.array([[1,0],[0,0]]))
    vec[0]=0.0
    vec[1]=np.sqrt(2)/2
    vec[2]=np.sqrt(2)/2
    assert dense.rdm(vec,[0])==pytest.approx(np.array([[1,0],[0,0]]))
    assert dense.rdm(vec,[1])==pytest.approx(np.array([[0.5,0],[0,0.5]]))
    assert dense.rdm(vec,[2])==pytest.approx(np.array([[0.5,0],[0,0.5]]))
    assert dense.rdm(vec,[1,2])==pytest.approx(np.array([[0,0,0,0],[0,0.5,0.5,0],[0,0.5,0.5,0],[0,0,0,0]]))
    assert dense.rdm(vec,[0,2])==pytest.approx(np.array([[0.5,0,0,0],[0,0.5,0,0],[0,0,0,0],[0,0,0,0]]))

def check_rdm(rdm):
    assert np.trace(rdm)==pytest.approx(1.0)
    assert np.trace(rdm@rdm)<=1.0+1e-15
    assert rdm==pytest.approx(rdm.T.conj())
    assert (la.eigvalsh(rdm)>-1e-15).all()

def test_real_rdm(seed_rng):
    seed_rng("real_rdm")
    vec=np.random.normal(size=(16,))
    vec/=np.sqrt(np.sum(vec.conj()*vec))
    check_rdm(dense.rdm(vec,[0]))
    check_rdm(dense.rdm(vec,[0,1]))
    check_rdm(dense.rdm(vec,[0,2]))
    check_rdm(dense.rdm(vec,[0,1,2]))
    check_rdm(dense.rdm(vec,[0,1,2,3]))
    check_rdm(dense.rdm(vec,[0,2,3]))
    check_rdm(dense.rdm(vec,[3,0]))

def test_complex_rdm(seed_rng):
    seed_rng("complex_rdm")
    vec=np.random.normal(size=(16,))+1.0j*np.random.normal(size=(16,))
    vec/=np.sqrt(np.sum(vec.conj()*vec))
    check_rdm(dense.rdm(vec,[0]))
    check_rdm(dense.rdm(vec,[0,1]))
    check_rdm(dense.rdm(vec,[0,2]))
    check_rdm(dense.rdm(vec,[0,1,2]))
    check_rdm(dense.rdm(vec,[0,1,2,3]))
    check_rdm(dense.rdm(vec,[0,2,3]))
    check_rdm(dense.rdm(vec,[3,0]))

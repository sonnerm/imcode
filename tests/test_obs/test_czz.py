import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
import pytest
from ..utils import seed_rng
import numpy as np

@pytest.fixture
def dense_ising_czz():
    t=3
    seed_rng("dense_ising_czz")
    J=np.random.normal()
    g=np.random.normal()
    h=np.random.normal()
    T=dense.ising_T(t,J,g,h)
    im=dense.im_iterative(T)
    lop=dense.ising_W(t,g)@dense.ising_h(t,h)
    return (dense.embedded_czz(im,lop),dense.boundary_czz(im,lop),(t,J,g,h))



def test_dense_direct_ising_czz(dense_ising_czz):
    t,J,g,h=dense_ising_czz[-1]
    L=7
    F=dense.ising_F([J]*(L-1),[g]*L,[h]*L)
    assert dense.direct_czz(F,t,3,3)==pytest.approx(dense_ising_czz[0])
    assert dense.direct_czz(F,t,0,0)==pytest.approx(dense_ising_czz[1])

def test_sparse_direct_ising_czz(dense_ising_czz):
    pass

@pytest.mark.xfail
def test_mps_direct_ising_czz(dense_ising_czz):
    pass

def test_mps_ising_czz():
    pass

def test_sparse_ising_czz():
    pass

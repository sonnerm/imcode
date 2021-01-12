import pytest
import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
from .utils import check_mps_im,check_dense_im
from ..utils import seed_rng
import numpy as np
@pytest.fixture(scope="module")
def dense_ising_im():
    seed_rng("ising_im")
    t=3
    J=np.random.normal()
    g=np.random.normal()
    h=np.random.normal()
    dt=dense.ising_T(t,J,g,h)
    im=dense.im_iterative(dt)
    return (im,(t,J,g,h))
def test_dense_ising_im_iterative(dense_ising_im):
    check_dense_im(dense_ising_im[0])

def test_dense_ising_im_diag(dense_ising_im):
    print(dense_ising_im[0])
    print(dense.im_diag(dense.ising_T(*dense_ising_im[1]))[0])
    assert dense.im_diag(dense.ising_T(*dense_ising_im[1]))[0]==pytest.approx(dense_ising_im[0])

def test_sparse_ising_im_iterative(dense_ising_im):
    assert sparse.im_iterative(sparse.ising_T(*dense_ising_im[1]))==pytest.approx(dense_ising_im[0])

def test_sparse_ising_im_diag(dense_ising_im):
    assert sparse.im_diag(sparse.ising_T(*dense_ising_im[1]))[0]==pytest.approx(dense_ising_im[0])

def test_mps_ising_im_iterative(dense_ising_im):
    assert mps.mps_to_dense(mps.im_iterative(mps.fold.ising_T(*dense_ising_im[1])))==pytest.approx(dense_ising_im[0])
@pytest.mark.xfail
def test_mps_ising_im_dmrg(dense_ising_im):
    assert mps.mps_to_dense(mps.im_dmrg(mps.fold.ising_T(*dense_ising_im[1])))==pytest.approx(dense_ising_im[0])

import pytest
import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
from .utils import check_mps_im,check_dense_im
from ..utils import seed_rng
import numpy as np
@pytest.fixture(scope="module")
def dense_ising_hr_imp():
    seed_rng("ising_hr_imp")
    t=3
    J=np.random.normal()
    g=np.random.normal()
    dt=dense.ising_hr_Tp(t,J,g)
    im=dense.im_iterative(dt)
    return (im,(t,J,g))
def test_dense_ising_hr_imp_iterative(dense_ising_hr_imp):
    check_dense_im(dense_ising_hr_imp[0])

def test_dense_ising_hr_imp_diag(dense_ising_hr_imp):
    print(dense_ising_hr_imp[0])
    print(dense.im_diag(dense.ising_hr_Tp(*dense_ising_hr_imp[1]))[0])
    assert dense.im_diag(dense.ising_hr_Tp(*dense_ising_hr_imp[1]))[0]==pytest.approx(dense_ising_hr_imp[0])

def test_sparse_ising_hr_imp_iterative(dense_ising_hr_imp):
    assert sparse.im_iterative(sparse.ising_hr_Tp(*dense_ising_hr_imp[1]))==pytest.approx(dense_ising_hr_imp[0])

def test_sparse_ising_hr_imp_diag(dense_ising_hr_imp):
    assert sparse.im_diag(sparse.ising_hr_Tp(*dense_ising_hr_imp[1]))[0]==pytest.approx(dense_ising_hr_imp[0])

def test_mps_ising_hr_imp_iterative(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_iterative(mps.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])
@pytest.mark.xfail
def test_mps_ising_hr_imp_dmrg(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_dmrg(mps.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])

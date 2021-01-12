import pytest
import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
from .utils import check_mps_im,check_dense_im,check_mps_imp,check_dense_imp
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
def test_dense_ising_hr_imp_expand(dense_ising_hr_imp):
    im=dense.im_iterative(dense.ising_hr_T(*dense_ising_hr_imp[1]))
    im2=dense.im_finite([dense.ising_hr_T(*dense_ising_hr_imp[1])],boundary=dense_ising_hr_imp[0])
    assert im == pytest.approx(im2)

def test_dense_ising_hr_imp_diag(dense_ising_hr_imp):
    assert dense.im_diag(dense.ising_hr_Tp(*dense_ising_hr_imp[1]))[0]==pytest.approx(dense_ising_hr_imp[0])

def test_sparse_ising_hr_imp_iterative(dense_ising_hr_imp):
    assert sparse.im_iterative(sparse.ising_hr_Tp(*dense_ising_hr_imp[1]))==pytest.approx(dense_ising_hr_imp[0])

def test_sparse_ising_hr_imp_diag(dense_ising_hr_imp):
    assert sparse.im_diag(sparse.ising_hr_Tp(*dense_ising_hr_imp[1]))[0]==pytest.approx(dense_ising_hr_imp[0])

def test_fold_ising_hr_imp_iterative(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_iterative(mps.fold.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])

def test_flat_ising_hr_imp_iterative(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_iterative(mps.flat.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])
@pytest.mark.xfail
def test_fold_ising_hr_imp_dmrg(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_dmrg(mps.fold.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])

@pytest.mark.xfail
def test_flat_ising_hr_imp_dmrg(dense_ising_hr_imp):
    assert mps.mps_to_dense(mps.im_dmrg(mps.flat.ising_hr_Tp(*dense_ising_hr_imp[1])))==pytest.approx(dense_ising_hr_imp[0])

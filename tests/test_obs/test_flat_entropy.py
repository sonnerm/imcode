import imcode.dense as dense
import imcode.sparse as sparse
import imcode.mps as mps
import pytest
from ..utils import seed_rng
import numpy as np

@pytest.fixture
def dense_ising_flat_entropy():
    t=3
    seed_rng("dense_ising_flat_entropy")
    J=np.random.normal()
    g=np.random.normal()
    h=np.random.normal()
    T=dense.ising_T(t,J,g,h)
    im=dense.im_iterative(T)
    return (dense.flat_entropy(im),(t,J,g,h))


def test_fold_ising_flat_entropy(dense_ising_flat_entropy):
    t,J,g,h=dense_ising_flat_entropy[-1]
    T=mps.fold.ising_T(t,J,g,h)
    im=mps.im_iterative(T)
    with pytest.raises(Exception):
        pytest.raises(mps.flat_entropy(im))

def test_flat_ising_flat_entropy(dense_ising_flat_entropy):
    t,J,g,h=dense_ising_flat_entropy[-1]
    T=mps.fold.ising_T(t,J,g,h)
    im=mps.im_iterative(T)
    assert mps.flat_entropy(im) == pytest.approx(dense_ising_flat_entropy[0])

def test_sparse_ising_flat_entropy(dense_ising_flat_entropy):
    t,J,g,h=dense_ising_flat_entropy[-1]
    T=sparse.ising_T(t,J,g,h)
    im=sparse.im_iterative(T)
    assert dense.flat_entropy(im) == pytest.approx(dense_ising_flat_entropy[0])

@pytest.mark.xfail
def test_flat_ising_flat_entropy(dense_ising_flat_entropy):
    t,J,g,h=dense_ising_flat_entropy[-1]
    T=mps.flat.ising_T(t,J,g,h)
    im=mps.im_iterative(T)
    assert mps.flat_entropy(im) == pytest.approx(dense_ising_flat_entropy[0])

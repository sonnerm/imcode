import numpy as np
import pytest
import imcode.mps as mps
import imcode.dense as dense

def test_fold_open_boundary_im():
    t=3
    assert mps.mps_to_dense(mps.fold.open_boundary_im(t))==pytest.approx(np.ones((2**(2*t))))

def test_flat_open_boundary_im():
    t=3
    assert mps.mps_to_dense(mps.flat.open_boundary_im(t))==pytest.approx(np.ones((2**(2*t))))

def test_dense_open_boundary_im():
    t=3
    assert dense.open_boundary_im(t)==pytest.approx(np.ones((2**(2*t))))

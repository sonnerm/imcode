import numpy as np
import pytest
import imcode.dense as dense
import imcode.mps as mps
import imcode.mps.fold as fold

def test_mps_perfect_dephaser_im():
    t=3
    assert mps.mps_to_dense(fold.perfect_dephaser_im(t))==pytest.approx(dense.perfect_dephaser_im(t))

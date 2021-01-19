import imcode.dense as dense
import imcode.mps.fold as fold
import imcode.mps as mps
from ..utils import seed_rng
import numpy as np
import pytest
def test_mps_dephaser_im():
    t=3
    seed_rng("mps_dephaser_im")
    gamma=np.abs(np.random.normal())
    assert mps.mps_to_dense(fold.dephaser_im(t,gamma))==pytest.approx(dense.dephaser_im(t,gamma))

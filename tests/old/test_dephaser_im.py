# import imcode.dense as dense
# import imcode.mps.fold as fold
# import imcode.mps as mps
#
# import numpy as np
# import pytest
# def test_mps_dephaser_im():
#     t=3
#     seed_rng("mps_dephaser_im")
#     gamma=np.abs(np.random.normal())
#     assert mps.mps_to_dense(fold.dephaser_im(t,gamma))==pytest.approx(dense.dephaser_im(t,gamma))
#
# def test_dephaser_perfect_dephaser_im():
#     t=3
#     assert dense.dephaser_im(t,1.0)==pytest.approx(dense.perfect_dephaser_im(t))
#
# def test_dephaser_open_boundary_im():
#     t=3
#     assert dense.dephaser_im(t,0.0)==pytest.approx(dense.open_boundary_im(t))

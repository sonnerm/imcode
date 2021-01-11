import numpy as np
import pytest
import imcode.dense as dense
import imcode.mps as mps

# @pytest.mark.skip()
# def test_mps_perfect_dephaser_im():
#     t=3
#     assert mps.mps_to_dense(mps.perfect_dephaser_im(t))==pytest.approx(dense.perfect_dephaser_im(t))
# @pytest.mark.skip()
# def test_dense_perfect_dephaser_im():
#     pass

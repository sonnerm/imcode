# import pytest
# import imcode.dense as dense
# import imcode.sparse as sparse
# import imcode.mps as mps
# from .utils import check_mps_im,check_dense_im
#
# import numpy as np
# @pytest.fixture(scope="module")
# def mps_ising_im_triangle():
#     seed_rng("mps_ising_im_triangle")
#     t=4
#     J=np.random.normal()
#     g=np.random.normal()
#     h=np.random.normal()
#     dt=mps.fold.ising_T(t,J,g,h)
#     im=mps.im_iterative(dt)
#     return (im,(t,J,g,h))
# def test_mps_ising_im_triangle(mps_ising_im_triangle):
#     im,(t,J,g,h) = mps_ising_im_triangle
#     Ts=[mps.fold.ising_T(x,J,g,h) for x in range(2,t+1,2)]
#     imt=mps.im_triangle(Ts)
#     assert im.overlap(imt)/im.norm/imt.norm==pytest.approx(1.0)

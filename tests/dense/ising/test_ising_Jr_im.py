# import pytest
# import imcode.dense as dense
# import imcode.sparse as sparse
# import imcode.mps as mps
# from .utils import check_mps_im,check_dense_im
#
# import numpy as np
# @pytest.fixture(scope="module")
# def dense_ising_Jr_im():
#     seed_rng("ising_Jr_im")
#     t=3
#     g=np.random.normal()
#     h=np.random.normal()
#     dt=dense.ising_Jr_T(t,g,h)
#     im=dense.im_iterative(dt)
#     return (im,(t,g,h))
#
# def test_dense_ising_Jr_im_iterative(dense_ising_Jr_im):
#     check_dense_im(dense_ising_Jr_im[0])
#
# @pytest.mark.slow
# def test_ising_Jr_im_disorder(dense_ising_Jr_im):
#     SAMPLE=1000
#     seed_rng("ising_Jr_im_disorder")
#     t,g,h=dense_ising_Jr_im[1]
#     vec=dense.open_boundary_im(t)
#     for i in range(2*t):
#         vec=dense.ising_T(t,np.random.uniform(0,2*np.pi),g,h)@vec
#     vav=np.zeros_like(vec)
#     for i in range(SAMPLE):
#         vec=dense.ising_T(t,np.random.uniform(0,2*np.pi),g,h)@vec
#         vav+=vec
#     vav/=SAMPLE
#     assert vav == pytest.approx(dense_ising_Jr_im[0],rel=1e-3,abs=1e-3)
#
# def test_dense_ising_Jr_im_diag(dense_ising_Jr_im):
#     print(dense_ising_Jr_im[0])
#     print(dense.im_diag(dense.ising_Jr_T(*dense_ising_Jr_im[1]))[0])
#     assert dense.im_diag(dense.ising_Jr_T(*dense_ising_Jr_im[1]))[0]==pytest.approx(dense_ising_Jr_im[0])
# @pytest.mark.xfail
# def test_sparse_ising_Jr_im_iterative(dense_ising_Jr_im):
#     assert sparse.im_iterative(sparse.ising_Jr_T(*dense_ising_Jr_im[1]))==pytest.approx(dense_ising_Jr_im[0])
#
# @pytest.mark.xfail
# def test_sparse_ising_Jr_im_diag(dense_ising_Jr_im):
#     assert sparse.im_diag(sparse.ising_Jr_T(*dense_ising_Jr_im[1]))[0]==pytest.approx(dense_ising_Jr_im[0])
#
# def test_fold_ising_Jr_im_iterative(dense_ising_Jr_im):
#     assert mps.mps_to_dense(mps.im_iterative(mps.fold.ising_Jr_T(*dense_ising_Jr_im[1])))==pytest.approx(dense_ising_Jr_im[0])
#
# def test_fold_ising_Jr_im_iterative(dense_ising_Jr_im):
#     assert mps.mps_to_dense(mps.im_zipup(mps.fold.ising_Jr_T(*dense_ising_Jr_im[1]),chi=64))==pytest.approx(dense_ising_Jr_im[0])
#
# # @pytest.mark.xfail
# # def test_flat_ising_Jr_im_iterative(dense_ising_Jr_im):
# #     assert mps.mps_to_dense(mps.im_iterative(mps.flat.ising_Jr_T(*dense_ising_Jr_im[1])))==pytest.approx(dense_ising_Jr_im[0])
# @pytest.mark.xfail
# def test_fold_ising_Jr_im_dmrg(dense_ising_Jr_im):
#     assert mps.mps_to_dense(mps.im_dmrg(mps.fold.ising_Jr_T(*dense_ising_Jr_im[1])))==pytest.approx(dense_ising_Jr_im[0])
#
# # @pytest.mark.xfail
# # def test_flat_ising_Jr_im_dmrg(dense_ising_Jr_im):
# #     assert mps.mps_to_dense(mps.im_dmrg(mps.flat.ising_Jr_T(*dense_ising_Jr_im[1])))==pytest.approx(dense_ising_Jr_im[0])
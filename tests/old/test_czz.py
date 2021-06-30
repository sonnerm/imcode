# import imcode.dense as dense
# import imcode.sparse as sparse
# import imcode.mps as mps
# import pytest
#
# import numpy as np
#
# @pytest.fixture
# def dense_ising_czz():
#     t=3
#     seed_rng("dense_ising_czz")
#     J=np.random.normal()
#     g=np.random.normal()
#     h=np.random.normal()
#     T=dense.ising_T(t,J,g,h)
#     im=dense.im_iterative(T)
#     lop=dense.ising_W(t,g)@dense.ising_h(t,h)
#     return (dense.embedded_czz(im,lop),dense.boundary_czz(im,lop),(t,J,g,h))
#
#
#
# def test_dense_direct_ising_czz(dense_ising_czz):
#     t,J,g,h=dense_ising_czz[-1]
#     L=7
#     F=dense.ising_F([J]*(L-1),[g]*L,[h]*L)
#     assert dense.direct_czz(F,t,3,3)==pytest.approx(dense_ising_czz[0])
#     assert dense.direct_czz(F,t,0,0)==pytest.approx(dense_ising_czz[1])
# # @pytest.mark.xfail
# # def test_sparse_direct_ising_czz(dense_ising_czz):
# #     t,J,g,h=dense_ising_czz[-1]
# #     L=7
# #     F=sparse.ising_F([J]*(L-1),[g]*L,[h]*L)
# #     assert sparse.direct_czz(F,t,3,3)==pytest.approx(dense_ising_czz[0])
# #     assert sparse.direct_czz(F,t,0,0)==pytest.approx(dense_ising_czz[1])
#
# def test_mps_direct_ising_czz(dense_ising_czz):
#     t,J,g,h=dense_ising_czz[-1]
#     L=7
#     F=mps.ising_F([J]*(L-1),[g]*L,[h]*L)
#     # evop=mps.evolve_operator(zz_op,F)
#     assert mps.direct_czz(F,t,3,3)[0]==pytest.approx(dense_ising_czz[0])
#     assert mps.direct_czz(F,t,0,0)[0]==pytest.approx(dense_ising_czz[1])
#
# def test_sparse_ising_czz(dense_ising_czz):
#     t,J,g,h=dense_ising_czz[-1]
#     T=sparse.ising_T(t,J,g,h)
#     im=sparse.im_iterative(T)
#     lop=sparse.ising_W(t,g)@sparse.ising_h(t,h)
#     assert sparse.embedded_czz(im,lop) == pytest.approx(dense_ising_czz[0])
#     assert sparse.boundary_czz(im,lop) == pytest.approx(dense_ising_czz[1])
#
# def test_mps_ising_czz(dense_ising_czz):
#     t,J,g,h=dense_ising_czz[-1]
#     T=mps.fold.ising_T(t,J,g,h)
#     im=mps.im_iterative(T)
#     lop=mps.multiply_mpos([mps.fold.ising_W(t,g),mps.fold.ising_h(t,h)])
#     assert mps.embedded_czz(im,lop) == pytest.approx(dense_ising_czz[0])
#     assert mps.boundary_czz(im,lop) == pytest.approx(dense_ising_czz[1])

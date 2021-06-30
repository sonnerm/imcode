# import imcode.dense as dense
# import imcode.sparse as sparse
# import imcode.mps as mps
# import pytest
#
# import numpy as np
#
# @pytest.fixture
# def dense_ising_czz_pol():
#     t=3
#     seed_rng("dense_ising_czz_pol")
#     J=np.random.normal()
#     g=np.random.normal()
#     h=np.random.normal()
#     T=dense.ising_T(t,J,g,h,(1.0,0.0))
#     im=dense.im_iterative(T)
#     lop=dense.ising_W(t,g,(1.0,0.0))@dense.ising_h(t,h)
#     return (dense.embedded_czz(im,lop),dense.boundary_czz(im,lop),(t,J,g,h))
#
#
#
# def test_dense_direct_ising_czz_pol(dense_ising_czz_pol):
#     t,J,g,h=dense_ising_czz_pol[-1]
#     L=7
#     F=dense.ising_F([J]*(L-1),[g]*L,[h]*L)
#     init=dense.dense_kron([np.array([[2.0,0.0],[0.0,0.0]])]*7)
#     assert dense.direct_czz(F,t,3,3,init=init)==pytest.approx(dense_ising_czz_pol[0])
#     assert dense.direct_czz(F,t,0,0,init=init)==pytest.approx(dense_ising_czz_pol[1])
# # @pytest.mark.xfail
# # def test_sparse_direct_ising_czz(dense_ising_czz):
# #     t,J,g,h=dense_ising_czz[-1]
# #     L=7
# #     F=sparse.ising_F([J]*(L-1),[g]*L,[h]*L)
# #     assert sparse.direct_czz(F,t,3,3)==pytest.approx(dense_ising_czz[0])
# #     assert sparse.direct_czz(F,t,0,0)==pytest.approx(dense_ising_czz[1])
#
# # def test_mps_direct_ising_czz_pol(dense_ising_czz_pol):
# #     t,J,g,h=dense_ising_czz_pol[-1]
# #     L=7
# #     F=mps.ising_F([J]*(L-1),[g]*L,[h]*L)
# #
# #     # evop=mps.evolve_operator(zz_op,F)
# #     assert mps.direct_czz(F,t,3,3)[0]==pytest.approx(dense_ising_czz_pol[0])
# #     assert mps.direct_czz(F,t,0,0)[0]==pytest.approx(dense_ising_czz_pol[1])
#
# def test_sparse_ising_czz_pol(dense_ising_czz_pol):
#     t,J,g,h=dense_ising_czz_pol[-1]
#     T=sparse.ising_T(t,J,g,h,(1.0,0.0))
#     im=sparse.im_iterative(T)
#     lop=sparse.ising_W(t,g,(1.0,0.0))@sparse.ising_h(t,h)
#     assert sparse.embedded_czz(im,lop) == pytest.approx(dense_ising_czz_pol[0])
#     assert sparse.boundary_czz(im,lop) == pytest.approx(dense_ising_czz_pol[1])
#
# def test_mps_ising_czz_pol(dense_ising_czz_pol):
#     t,J,g,h=dense_ising_czz_pol[-1]
#     T=mps.fold.ising_T(t,J,g,h,(1.0,0.0))
#     im=mps.im_iterative(T)
#     lop=mps.multiply_mpos([mps.fold.ising_W(t,g,(1.0,0.0)),mps.fold.ising_h(t,h)])
#     # st=mps.zz_state(t)
#     # mps.apply(lop,st)
#     # assert mps.mpo_to_dense(lop)@dense.zz_state(t) == pytest.approx(mps.mps_to_dense(st))
#     # dst=st.get_theta(0, st.L).take_slice([0, 0], ['vL', 'vR']).to_ndarray().ravel()*st.norm
#     # dim=im.get_theta(0, im.L).take_slice([0, 0], ['vL', 'vR']).to_ndarray().ravel()*im.norm
#     # assert st.overlap(im) == pytest.approx(dst.conj()@dim)
#     # assert st.overlap(im) == pytest.approx(mps.mps_to_dense(st).conj()@mps.mps_to_dense(im))
#     # assert mps.mps_to_dense(mps.zz_state(t)) == pytest.approx(dense.zz_state(t))
#     assert mps.boundary_czz(im,lop) == pytest.approx(dense_ising_czz_pol[1])
#     assert mps.embedded_czz(im,lop) == pytest.approx(dense_ising_czz_pol[0])

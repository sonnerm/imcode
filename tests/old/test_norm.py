# import imcode.dense as dense
# import imcode.sparse as sparse
# import imcode.mps as mps
# import pytest
#
# import numpy as np
# def test_dense_direct_ising_norm():
#     t=3
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     L=7
#     F=dense.ising_F([J]*(L-1),[g]*L,[h]*L)
#     assert dense.direct_norm(F,t,3,3)==pytest.approx(1.0)
#     assert dense.direct_norm(F,t,0,0)==pytest.approx(1.0)
#
# def test_dense_ising_norm():
#     t=3
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     T=dense.ising_T(t,J,g,h)
#     im=dense.im_iterative(T)
#     lop=dense.ising_W(t,g)@dense.ising_h(t,h)
#     assert dense.embedded_norm(im,lop) == pytest.approx(1.0)
#     assert dense.boundary_norm(im,lop) == pytest.approx(1.0)
#
# @pytest.mark.xfail
# def test_sparse_direct_ising_norm(dense_ising_czz):
#     t=3
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     L=7
#     F=sparse.ising_F([J]*(L-1),[g]*L,[h]*L)
#     assert sparse.direct_norm(F,t,3,3)==pytest.approx(1.0)
#     assert sparse.direct_norm(F,t,0,0)==pytest.approx(1.0)
#
# @pytest.mark.xfail
# def test_mps_direct_ising_norm(dense_ising_czz):
#     t=3
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     L=7
#     F=mps.ising_F([J]*(L-1),[g]*L,[h]*L)
#     assert mps.direct_norm(F,t,3,3)==pytest.approx(1.0)
#     assert mps.direct_norm(F,t,0,0)==pytest.approx(1.0)
#
# def test_sparse_ising_norm():
#     t=6
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     T=sparse.ising_T(t,J,g,h)
#     im=sparse.im_iterative(T)
#     lop=sparse.ising_W(t,g)@sparse.ising_h(t,h)
#     assert sparse.embedded_norm(im,lop) == pytest.approx(1.0)
#     assert sparse.boundary_norm(im,lop) == pytest.approx(1.0)
#
# def test_fold_ising_norm():
#     t=10
#     seed_rng("ising_norm")
#     J,g,h=np.random.normal(size=3)
#     T=mps.fold.ising_T(t,J,g,h)
#     im=mps.im_iterative(T,chi=64)
#     lop=mps.multiply_mpos([mps.fold.ising_W(t,g),mps.fold.ising_h(t,h)])
#     assert mps.embedded_norm(im,lop) == pytest.approx(1.0)
#     assert mps.boundary_norm(im,lop) == pytest.approx(1.0)

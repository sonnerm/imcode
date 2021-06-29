# import numpy as np
# import numpy.linalg as la
# from imcode import dense
# from imcode import sparse
# from imcode import mps
#
# import pytest
#
# @pytest.fixture(scope="module")
# def dense_ising_hr_Tp():
#     T=3
#     seed_rng("dense_ising_hr_Tp")
#     J=np.random.normal()
#     g=np.random.normal()
#     return (dense.ising_hr_Tp(T,J,g),(T,J,g))
#
# def test_dense_ising_hr_Tp(dense_ising_hr_Tp):
#     diT=dense_ising_hr_Tp[0]
#     assert diT.dtype==np.complex_
#     proj=la.matrix_power(diT,dense_ising_hr_Tp[1][0]*2)
#     assert proj==pytest.approx(diT@proj)
# def test_sparse_ising_hr_Tp(dense_ising_hr_Tp):
#     sih=sparse.ising_hr_Tp(*dense_ising_hr_Tp[1])
#     assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_hr_Tp[0])
#
# def test_fold_ising_hr_Tp(dense_ising_hr_Tp):
#     mih=mps.fold.ising_hr_Tp(*dense_ising_hr_Tp[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_Tp[0])
# @pytest.mark.skip()
# def test_flat_ising_hr_Tp(dense_ising_hr_Tp):
#     mih=mps.flat.ising_hr_Tp(*dense_ising_hr_Tp[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_Tp[0])

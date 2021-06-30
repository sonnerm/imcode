# import numpy as np
# import numpy.linalg as la
# from imcode import dense
# from imcode import sparse
# from imcode import mps
#
# import pytest
#
# @pytest.fixture(scope="module")
# def dense_ising_dephase_hr_Tp():
#     T=3
#     seed_rng("dense_ising_dephase_hr_Tp")
#     J=np.random.normal()
#     g=np.random.normal()
#     gamma=np.random.random()
#     return (dense.ising_dephase_hr_Tp(T,J,g,gamma),(T,J,g,gamma))
#
# def test_dense_ising_dephase_hr_Tp(dense_ising_dephase_hr_Tp):
#     diT=dense_ising_dephase_hr_Tp[0]
#     assert diT.dtype==np.complex_
#     proj=la.matrix_power(diT,dense_ising_dephase_hr_Tp[1][0]*2)
#     assert proj==pytest.approx(diT@proj)
# def test_sparse_ising_dephase_hr_Tp(dense_ising_dephase_hr_Tp):
#     sih=sparse.ising_dephase_hr_Tp(*dense_ising_dephase_hr_Tp[1])
#     assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_dephase_hr_Tp[0])
#
# def test_fold_ising_dephase_hr_Tp(dense_ising_dephase_hr_Tp):
#     mih=mps.fold.ising_dephase_hr_Tp(*dense_ising_dephase_hr_Tp[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_dephase_hr_Tp[0])
#     #assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]
#
# @pytest.mark.skip()
# def test_flat_ising_dephase_hr_Tp(dense_ising_dephase_hr_Tp):
#     mih=mps.flat.ising_dephase_hr_Tp(*dense_ising_dephase_hr_Tp[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_dephase_hr_Tp[0])
#     assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]

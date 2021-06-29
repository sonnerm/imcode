# import numpy as np
# import numpy.linalg as la
# from imcode import dense
# from imcode import sparse
# from imcode import mps
#
# import pytest
#
# @pytest.fixture(scope="module")
# def dense_ising_hr_T():
#     T=3
#     seed_rng("dense_ising_hr_T")
#     J=np.random.normal()
#     g=np.random.normal()
#     return (dense.ising_hr_T(T,J,g),(T,J,g))
#
# @pytest.mark.slow
# def test_ising_hr_disorder(dense_ising_hr_T):
#     SAMPLE=1000
#     seed_rng("dense_ising_hr_T_dis")
#     Tmd=np.zeros_like(dense_ising_hr_T[0])
#     t,J,g=dense_ising_hr_T[1]
#     for i in range(SAMPLE):
#         Tmd+=dense.ising_T(t,J,g,np.random.uniform(0,2*np.pi))
#     Tmd/=SAMPLE
#     print(Tmd)
#     print(dense_ising_hr_T[0])
#     assert Tmd==pytest.approx(dense_ising_hr_T[0],rel=1e-3,abs=1e-3)
#
# def test_dense_ising_hr_T(dense_ising_hr_T):
#     diT=dense_ising_hr_T[0]
#     assert diT.dtype==np.complex_
#     proj=la.matrix_power(diT,dense_ising_hr_T[1][0]*2)
#     assert proj==pytest.approx(diT@proj)
# def test_sparse_ising_hr_T(dense_ising_hr_T):
#     sih=sparse.ising_hr_T(*dense_ising_hr_T[1])
#     assert sparse.sparse_to_dense(sih)==pytest.approx(dense_ising_hr_T[0])
#
# def test_fold_ising_hr_T(dense_ising_hr_T):
#     mih=mps.fold.ising_hr_T(*dense_ising_hr_T[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_T[0])
#     #assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]
#
# @pytest.mark.skip()
# def test_flat_ising_hr_T(dense_ising_hr_T):
#     mih=mps.flat.ising_hr_T(*dense_ising_hr_T[1])
#     assert mps.mpo_to_dense(mih)==pytest.approx(dense_ising_hr_T[0])
#     #assert mih.chi==[1]+[4*i for i in range(1,(mih.L)//2)]+[4*i for i in range((mih.L)//2,0,-1)]+[1]